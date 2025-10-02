import cv2
import numpy as np
import glob

# 定义棋盘格规格
#cap = cv2.VideoCapture(0)
pattern_size = (8, 6)  # 8x6 内角点
square_size = 0.03    # 每个格子的边长（米），你自己测

# 世界坐标系下的棋盘格 3D 点
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
objp *= square_size

objpoints = []  # 3D 点
imgpoints = []  # 2D 点

images = glob.glob('D:/Meiqi/leg-tracking-master/leg-tracking-master/*.jpg')  # 存放棋盘格图片路径

for fname in images:
    print("for")
    img = cv2.imread(fname)
    #print(img)
    #cv2.imshow('image', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    print(ret)
    print(corners)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

# 标定
print(objpoints)
print(imgpoints)
print(gray.shape)
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(gray.shape[::-1])
print("K=", K)
print("dist=", dist)
