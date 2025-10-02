import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# 假设棋盘 9x6 内角点，每格 25mm
pattern_size = (8, 6)
square_size = 0.03  # 30mm
objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
objp *= square_size
#print(objp)
# 读取两台相机拍摄的同一个棋盘图像
img1 = cv2.imread("cam0_ori.jpg")
img2 = cv2.imread("cam1_ori.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 已知的相机内参
# K1 = np.array([[1.19278960e+03, 0.00000000e+00, 9.59500000e+02],
#  [0.00000000e+00, 1.18743072e+03, 5.39500000e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# dist1 = np.array([[ 0.28357869, -0.58097189,  0,  0,  0]])

K1 = np.array([[1.19278960e+03, 0.00000000e+00, 9.59500000e+02],
 [0.00000000e+00, 1.18743072e+03, 5.39500000e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist1 = np.array([[ 0.28357869, -0.58097189,  0,  0,  0]])

K2 = np.array([[1.17741831e+03, 0.00000000e+00, 9.59500000e+02],
 [0.00000000e+00, 1.18009358e+03, 5.39500000e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist2 = np.array([[ 0.28725371, -0.69132378,  0,0,0]])

# 找角点
ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)

#print(corners1)
# print(corners2)
if ret1:
    # cam1 的外参
    _, rvec1, tvec1 = cv2.solvePnP(objp, corners1, K1, dist1)
    R1, _ = cv2.Rodrigues(rvec1)
    r1 = R.from_matrix(R1)
    euler1 = r1.as_euler('xyz', degrees=True)  # 你可以改 'xyz', 'zyx' 等顺序
    print("Euler1 (xyz, degree):", euler1)
    # cam2 的外参
    _, rvec2, tvec2 = cv2.solvePnP(objp, corners2, K2, dist2)
    R2, _ = cv2.Rodrigues(rvec2)
    r2 = R.from_matrix(R2)
    euler2 = r2.as_euler('xyz', degrees=True)
    print("Euler2 (xyz, degree):", euler2)

    print("Cam1: R=\n", R1, "\nT=\n", tvec1)
    print("Cam2: R=\n", R2, "\nT=\n", tvec2)

    C_chess_1 = -R1.T @ tvec1
    C_chess_2 = -R2.T @ tvec2

    T = np.diag([1, 1, -1])  # 反转Z轴
    C_world_1 = T @ C_chess_1
    C_world_2 = T @ C_chess_2
    print("Cam1: C_world_1=\n", C_world_1)
    print("Cam2: C_world_2=\n", C_world_2)
