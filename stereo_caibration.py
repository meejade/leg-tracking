import cv2
import numpy as np

# 假设棋盘 9x6 内角点，每格 25mm
pattern_size = (8, 6)
square_size = 0.03  # 30mm
objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
objp *= square_size
print(objp)
# 读取两台相机拍摄的同一个棋盘图像
img1 = cv2.imread("cam0_ori.jpg")
#img2 = cv2.imread("cam1_3.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 已知的相机内参
K1 = np.array([[960, 0.00000000e+00, 960],
 [0.00000000e+00, 540, 540],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist1 = np.array([[ 0,  0,  0,  0,  0]])
# K2 = np.array([[3.35816080e+03, 0.00000000e+00, 9.45815977e+02],
#  [0.00000000e+00, 3.35209607e+03, 4.30019147e+02],
#  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# dist2 = np.array([[ 5.60586943e-01, -7.71429132e+00, -1.14857662e-02,  3.50552900e-03,
#    3.92616041e+01]])

# 找角点
ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
# ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)
corners1_rev = corners1[::-1]
# corners2_rev = corners2[::-1]
print(corners1)
# print(corners2)
if ret1:
    # cam1 的外参
    _, rvec1, tvec1 = cv2.solvePnP(objp, corners1_rev, K1, dist1)
    R1, _ = cv2.Rodrigues(rvec1)

    # cam2 的外参
    # _, rvec2, tvec2 = cv2.solvePnP(objp, corners2_rev, K2, dist2)
    # R2, _ = cv2.Rodrigues(rvec2)

    print("Cam1: R=\n", R1, "\nT=\n", tvec1)
    # print("Cam2: R=\n", R2, "\nT=\n", tvec2)
