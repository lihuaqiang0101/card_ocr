import numpy as np
import cv2
from sklearn.cluster import KMeans

# def get_points(img1):
#     # img1 = cv2.imread(path, 0)
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
#     Harris = []
#     Harris.append((int(xx_min), int(yy_min)))
#     Harris.append((int(xx_max), int(yy_min)))
#     Harris.append((int(xx_min), int(yy_max)))
#     Harris.append((int(xx_max), int(yy_max)))
#     return np.array(Harris)
"""
最小外接矩形
"""
# def get_points(name,img1):
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
#     img1 = np.zeros(shape=img1.shape)
#     cv2.drawContours(img1, [bbox], 0, (255, 255, 255), 1)
#     cv2.imwrite(name,img1)
#     return cv2.imread(name,0)

"""
凸包
"""
def get_points(name,img1):
    _, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, 3, 2)
    cnt = contours[0]

    # 2.寻找凸包，得到凸包的角点
    hull = cv2.convexHull(cnt)

    # 3.绘制凸包
    image = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img = np.zeros(shape=image.shape)
    cv2.polylines(img, [hull], True, (255, 255, 255), 1)
    cv2.imwrite(name,img)
    return cv2.imread(name,0)

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def getHarris(gray):
    # 2.灰度化
    # gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # 3.角点检测
    corners = cv2.cornerHarris(gray, 7, 5, 0.04)
    corner = np.argwhere(corners > corners.max() * 0.01)
    original = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    mixture = original.copy()
    mixture[corners > corners.max() * 0.01] = [0, 0, 255]

    # cv2.imshow('c',mixture)
    # cv2.waitKey(0)

    n_clusters = 4
    cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(corner)
    centroid = cluster.cluster_centers_[:,[1,0]]
    centroid = centroid[np.lexsort(centroid.T)]#按最后一列排序
    up = centroid[:2,:]
    down = centroid[2:, :]
    #按第一列排序
    up = up[np.lexsort(up[:,::-1].T)]
    down = down[np.lexsort(down[:,::-1].T)]
    Harris = []
    Harris.append((int(up[0][0]),int(up[0][1])))
    Harris.append((int(up[1][0]), int(up[1][1])))
    Harris.append((int(down[0][0]), int(down[0][1])))
    Harris.append((int(down[1][0]), int(down[1][1])))
    # for i in range(4):
    #     Harris.append(tuple(corner[len(corner)//8 + len(corner)//4*i]))
    return np.array(Harris)