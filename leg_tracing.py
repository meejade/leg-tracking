# 文件: multi_cam_color_3d.py
import cv2
import numpy as np
import socket
import json
import time
from collections import deque

# ----------------------------
# 用户需配置的部分
# ----------------------------
# 每台相机的索引或视频流 URL（按顺序）
CAM_SOURCES = [0, 1]  # 示例：两台本地 webcam；可替换为 RTSP/USB id
# 相机内参与外参（应由标定得到）
# camera_params[i] = dict(K=3x3, dist= (k1,k2,p1,p2,k3...), R=3x3, t=3x1)
camera_params = [
    # 示例占位（务必用实际标定结果替换）
    {

        "K": np.array([[1160.5, 0., 934.6532],
                       [0., 1169.6, 574.8938],
                       [0., 0., 1.]]),
        "dist": np.zeros(5),
        # 示例外参：第二台相机绕y轴平移 200mm
        "R": np.array([[0.9984, 0.0504, -0.0258],
                       [-0.0358, 0.9145, 0.4029],
                       [0.0439, -0.4014, 0.9149]]),
        "t": np.array([80.146, 430.1, 839.5])  # 单位：米（或与K相同尺度）
    },
    {
        "K": np.array([[1012.9, 0., 926.4134],
                       [0., 1030.4, 614.3430],
                       [0., 0., 1.]]),
        "dist": np.zeros(5),
        "R": np.array([[0.9995, 0.0074, 0.0304],
                       [-0.0152, 0.9643, 0.2643],
                       [-0.0274, -0.2646, 0.9640]]),
        "t": np.array([9.5, -304.788, 937.249])
    }
]

# 颜色阈值设置（HSV），需要在你的实验光照下调参
# 示例：6 种颜色 (h_low,h_high), (s_low,s_high), (v_low,v_high)
color_ranges = {
    "blue":    [(90,110,100,255,100,255)],  # 红色可能要双区间
    "green":  [(35,85,55,255,100,255)],
    "orange":  [(0,20,100,255,100,255)],
    "yellow": [(20,30,55,255,55,255)],
    "pink":[(160,179,100,255,100,255)],
    "violet":   [(110,130,55,255,100,255)],
     # "1":  [(90,110,100,255,0,50)],
     # "2":  [(90,110,100,255,50,100)],
     # "3":  [(90,110,100,255,150,200)],
     # "4":  [(90,110,100,255,200,255)],

    # "5":  [(120,150,0,255,0,255)],
    # "6":  [(150,180,0,255,0,255)],
}

# UDP 配置（将 3D 点发给 Unity）
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# Kalman / 平滑参数
SMOOTH_WINDOW = 5  # 简单移动平均缓存大小（可用 Kalman 替换为更好效果）

# ----------------------------
# 工具函数: 投影矩阵
# ----------------------------
def make_projection_matrix(K, R, t):
    """返回 3x4 投影矩阵 P = K [R | t]"""
    Rt = np.hstack((R, t.reshape(3,1)))
    P = K.dot(Rt)
    return P

# 线性多视图三角化（DLT）
def triangulate_multiview(points_uv, proj_mats):
    """
    points_uv: list of (u,v) for each view (only include views where marker可见)
    proj_mats: list of corresponding 3x4 projection matrices
    返回: 3D 点 (x,y,z)（齐次归一化）
    """
    # 构造 A 矩阵，Ax = 0
    A = []
    for (u,v), P in zip(points_uv, proj_mats):
        # 每个视角产生两行
        A.append(u * P[2,:] - P[0,:])
        A.append(v * P[2,:] - P[1,:])
    A = np.array(A)  # (2m x 4)
    # SVD 求解
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]

# ----------------------------
# 颜色检测函数
# ----------------------------
def detect_color_centroids(frame, color_ranges):
    """
    frame: BGR 图像
    color_ranges: dict color -> list of HSV tuples (h1,h2,s1,s2,v1,v2), 支持多个区间（如红）
    返回: dict color -> (u,v) 或 None
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = {}
    for color, ranges in color_ranges.items():
        mask_total = None
        for (h1,h2,s1,s2,v1,v2) in ranges:
            lower = np.array([h1, s1, v1], dtype=np.uint8)
            upper = np.array([h2, s2, v2], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            if mask_total is None:
                mask_total = mask
            else:
                mask_total = cv2.bitwise_or(mask_total, mask)
        # 形态学去噪
        kernel = np.ones((3,3), np.uint8)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_total = cv2.morphologyEx(mask_total, cv2.MORPH_CLOSE, kernel, iterations=1)
        # 找轮廓并取最大区域的质心
        contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # 选面积最大的轮廓
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < 5:  # 过滤极小噪声
                out[color] = None
            else:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])
                    out[color] = (cx, cy)
                else:
                    out[color] = None
        else:
            out[color] = None
    return out

# ----------------------------
# 简单的 3D 点平滑（移动平均）
# ----------------------------
class MovingAverage:
    def __init__(self, window=5):
        self.window = window
        self.buff = deque(maxlen=window)
    def update(self, pt):
        if pt is None:
            return None
        self.buff.append(pt)
        arr = np.array(self.buff)
        return arr.mean(axis=0)
hsv_frame = None
def pick_color(event, x, y, flags, param):
    global hsv_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_value = hsv_frame[y,x]
        print(f"Clicked: {hsv_value}")
current_frame = None
def pick_points(event, x, y, flags, param):
    global current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        p = camera_params[0]
        proj_matrix = [make_projection_matrix(p["K"], p["R"], p["t"])]
        result = triangulate_multiview([(x,y)], proj_matrix)  # 3D (x,y,z)
        print(f"2D: {x,y}, 3D: {result}")

# ----------------------------
# 主循环：多相机检测 + 三角化 + 平滑 + UDP 发送
# ----------------------------
def main():
    # 打开摄像头
    global hsv_frame
    global current_frame
    cv2.namedWindow("cam1")

    cv2.setMouseCallback("cam1", pick_points)
    caps = [cv2.VideoCapture(src) for src in CAM_SOURCES]
    for i, cap in enumerate(caps):
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            print(f"Open camera {i} src={CAM_SOURCES[i]}")
        if not cap.isOpened():
            print(f"Failed to open camera {i} src={CAM_SOURCES[i]}")
            return

    # 准备投影矩阵
    proj_mats = [make_projection_matrix(p["K"], p["R"], p["t"]) for p in camera_params]

    # 平滑器（每个颜色一个）
    smoothers = {c: MovingAverage(window=SMOOTH_WINDOW) for c in color_ranges.keys()}

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        while True:
            # 读取所有相机的一帧
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    frame = None
                frames.append(frame)

            # 如果有任何 frame None，可跳过或用上次帧（这里我们跳过）
            if any(f is None for f in frames):
                print("Some frame missing, skipping")
                time.sleep(0.01)
                continue

            # 对每台相机进行颜色检测，保存结果为 detections[cam_idx][color] = (u,v) 或 None
            detections = []
            for frame in frames:
                det = detect_color_centroids(frame, color_ranges)
                detections.append(det)

            # 按颜色聚合视图
            results_3d = {}
            for color in color_ranges.keys():
                # 收集在各视角可见的像素点和对应投影矩阵
                pts_uv = []
                P_list = []
                for cam_idx, det in enumerate(detections):
                    uv = det.get(color)
                    if uv is not None:
                        # 可选择先做去畸变：cv2.undistortPoints；这里直接用原始像素，后续用于 P
                        pts_uv.append((float(uv[0]), float(uv[1])))
                        P_list.append(proj_mats[cam_idx])
                if len(pts_uv) >= 2:
                    # 多视角三角化
                    X = triangulate_multiview(pts_uv, P_list)  # 3D (x,y,z)
                    # 平滑
                    Xs = smoothers[color].update(X)
                    results_3d[color] = None if Xs is None else Xs.tolist()
                else:
                    # 可见视角不足，使用 None 或者上帧预测（移动平均无法预测）
                    results_3d[color] = None
            for color in results_3d:
                print(f"{color}: {results_3d[color]}")
            # 发送到 Unity
            packet = {
                "timestamp": time.time(),
                "points": results_3d
            }
            sock.sendto(json.dumps(packet).encode('utf-8'), (UDP_IP, UDP_PORT))

            # 可视化（调试用）：在第一台相机画出检测结果和投影（若需要）
            for i, (frame, det) in enumerate(zip(frames, detections)):
                vis = frame.copy()
                for color, uv in det.items():
                    if uv is not None:
                        cv2.circle(vis, (int(uv[0]), int(uv[1])), 6, (0, 255, 0), -1)
                        cv2.putText(vis, color, (int(uv[0]) + 8, int(uv[1]) + 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.imshow(f"cam{i}", vis)
                current_frame = vis
                #hsv_frame = cv2.cvtColor(vis, cv2.COLOR_BGR2HSV)
                #cv2.imshow(f"color{i}", hsv_frame)
                # frame = cv2.cvtColor(vis, cv2.COLOR_HSV2BGR)
                # cv2.imshow(f"BGR{i}", frame)

            # 只需要一个 waitKey，就能刷新所有窗口
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
