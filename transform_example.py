import cv2
import numpy as np

# image = cv2.imread(r'C:\xingshizheng\params\a.jpg')
# img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# def get_h(path):
#     img1 = cv2.imread(path, 0)
#     _, thresh = cv2.threshold(img1, 200, 255, 0)
#     image, contours = cv2.findContours(thresh, 2, 1)
#     area = 0
#     for c in image:
#         rect = cv2.minAreaRect(c)
#         box = np.int0(cv2.boxPoints(rect))
#         y_max = np.max(box[:, 1])
#         x_max = np.max(box[:, 0])
#         y_min = np.min(box[:, 1])
#         x_min = np.min(box[:, 0])
#         if (y_max - y_min) * (x_max - x_min) > area:
#             area = (y_max - y_min) * (x_max - x_min)
#             yy_max = y_max
#             xx_max = x_max
#             yy_min = y_min
#             xx_min = x_min
#     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     #左上
#     img1 = cv2.circle(img1, (int(xx_min), int(yy_min)), 10, (0, 0, 255))
#     #右上
#     img1 = cv2.circle(img1, (int(xx_max), int(yy_min)), 10, (0, 0, 255))
#     #左下
#     img1 = cv2.circle(img1, (int(xx_min), int(yy_max)), 10, (0, 0, 255))
#     #右下
#     img1 = cv2.circle(img1, (int(xx_max), int(yy_max)), 10, (0, 0, 255))
#     cv2.imshow('',img1)
#     cv2.waitKey(0)

# def get_h(path):
#     img1 = cv2.imread(path, 0)
#     _, thresh = cv2.threshold(img1, 200, 255, 0)
#     image, contours = cv2.findContours(thresh, 2, 1)
#     area = 0
#     for c in image:
#         rect = cv2.minAreaRect(c)
#         box = np.int0(cv2.boxPoints(rect))
#         y_max = np.max(box[:, 1])
#         x_max = np.max(box[:, 0])
#         y_min = np.min(box[:, 1])
#         x_min = np.min(box[:, 0])
#         if (y_max - y_min) * (x_max - x_min) > area:
#             area = (y_max - y_min) * (x_max - x_min)
#             bbox = box
#             yy_max = y_max
#             xx_max = x_max
#             yy_min = y_min
#             xx_min = x_min
#     cv2.drawContours(img1, [bbox], 0, (100, 100, 100), 4)
#     cv2.imshow('a', img1)
#     cv2.waitKey(0)
# import os
# for dir in os.listdir(r'C:\xingshizheng\train'):
#     try:
#         get_h(r'C:\xingshizheng\train\{}'.format(dir))
#     except:
#         pass

import cv2 as cv
import numpy as np

# 凸包
# 1.先找到轮廓
img = cv.imread(r'C:\xingshizheng\params\a.jpg', 0)
_, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
contours, hierarchy = cv.findContours(thresh, 3, 2)
cnt = contours[0]

# 2.寻找凸包，得到凸包的角点
hull = cv.convexHull(cnt)

# 3.绘制凸包
image = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
cv.polylines(image, [hull], True, (0, 0, 255), 1)

cv.imshow('convex hull', image)
cv.waitKey(0)
cv.destroyAllWindows()