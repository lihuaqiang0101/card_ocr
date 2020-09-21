import torch
import numpy as np
from PIL import Image
from skimage.transform import resize
from CardOcr import deeplabv3_resnet50
import cv2
import torch.nn.functional as F
from transform import four_point_transform
from transform import getHarris
from transform import get_points
from math import *

def Transformer(image,pts):
    warped = four_point_transform(image, pts)
    return warped

def getCardsMask(path):
    model_card = deeplabv3_resnet50(num_classes=1)
    model_card.cuda()
    model_card.load_state_dict(torch.load(r'C:\xingshizheng\params\paramdeeplab_card74.pth', map_location='cuda'))
    model_card.eval()

    image = Image.open(path)
    image = np.array(image)
    img_h, img_w, _ = image.shape
    dim_diff = np.abs(img_h - img_w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1, pad2), (0, 0), (0, 0)) if img_h <= img_w else ((0, 0), (pad1, pad2), (0, 0))
    card = np.pad(image, pad, 'constant', constant_values=128)
    input_img = card / 255.
    padded_h, padded_w, _ = input_img.shape

    input_img1 = resize(input_img, (490, 490, 3), mode='reflect')
    input_data1 = np.expand_dims(np.transpose(input_img1, (2, 0, 1)), axis=0)
    imagedata1 = torch.Tensor(np.array(input_data1, dtype='float32'))
    imagedata1 = imagedata1.cuda()
    cardMask = F.sigmoid(model_card(imagedata1)['out'])
    cardMask = cardMask.cpu().detach().numpy()
    cardMask = np.reshape(cardMask, (490, 490))

    mask_card = cardMask > 0.98
    mask_nocard = cardMask < 0.98
    indexs_card = np.array(np.where(mask_card == True))
    indexs_card = np.stack(indexs_card, axis=1)
    cardMask[indexs_card] = 255
    cardMask[mask_nocard] = 0
    card = cv2.cvtColor(card,cv2.COLOR_BGR2RGB)
    return card,cardMask#resize(card, (490, 490, 3), mode='reflect')

def r(image,degree):
    height, width = image.shape[:2]
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    image = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
    return image

def getCards(path):
    name = path.split('\\')[-1]
    name = r'C:\xingshizheng\boxes\{}'.format(name)
    image, label = getCardsMask(path)
    cv2.imwrite(name, label)
    label = cv2.imread(name, 0)
    label = get_points(name,label)
    pts = getHarris(label) * image.shape[0] / 490
    Cards = Transformer(image, pts)
    h,w,_ = Cards.shape
    if h > w:
        Cards = r(Cards,90)
    Cards = cv2.resize(Cards, (800, 500), cv2.INTER_CUBIC)
    return name,Cards

def getText(name,image):
    plate_number = image[75:130,115:340]
    vehicle_type = image[80:130,442:780]
    holder = image[128:188,120:784]
    address = image[190:250,120:780]
    Nature_of_use = image[247:310,120:285]
    motorcycle_type = image[250:310,390:790]
    VIN = image[310:373,357:785]
    Engine_number = image[370:432,335:792]
    registration_date = image[427:500,303:515]
    date_of_issue = image[424:500,585:798]
    images = []
    images.append(plate_number)
    images.append(vehicle_type)
    images.append(holder)
    images.append(address)
    images.append(Nature_of_use)
    images.append(motorcycle_type)
    images.append(VIN)
    images.append(Engine_number)
    images.append(registration_date)
    images.append(date_of_issue)
    image = np.ones(shape=(800, 800, 3)) * 255
    y = 0
    for img in images:
        shape = img.shape
        image[y:shape[0]+y,0:shape[1]] = img
        y += shape[0]
        y += 10
    cv2.imwrite(name,image)
    return cv2.imread(name)

# name,Cards = getCards(r'C:\xingshizheng\train\Roxuiwo7RkiBluNDnDVfzQ.jpg')
# name,Cards = getCards(r'C:\xingshizheng\images\1.jpg')
# Cards = getText(name,Cards)
# cv2.imshow('',Cards)
# cv2.waitKey(0)

# import os
# for dir in os.listdir(r'D:\MASKpicture\datas\行驶证\薛文雯（完成）'):
#     image,label = getCardsMask(r'D:\MASKpicture\datas\行驶证\薛文雯（完成）\{}'.format(dir))
#     cv2.imwrite(r'C:\xingshizheng\params\a.jpg',label)
#     label = cv2.imread(r'C:\xingshizheng\params\a.jpg',0)
#
#     """
#     利用角点检测找出四个点方案
#     """
#     # label= cv2.medianBlur(label, ksize=5)
#     # label= cv2.blur(label, (25,25))
#     # index = label > 0
#     # label[index] = 255
#     label = get_points(name,label)
#     pts = getHarris(label) * image.shape[0] / 490
#
#     """
#     利用最小外接矩形找四个点方案
#     """
#     # pts = get_points(label)
#
#     """
#     透视变换
#     """
#     Transformer(image,pts)