# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:29:47 2019

@author: SAI
"""

import cv2
import math
import numpy as np
import os
# pdb仅仅用于调试，不用管它
import pdb

#旋转图像的函数
def rotate_image(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    # 仿射变换
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

#对应修改xml文件
def rotate_xml(src, xmin, ymin, xmax, ymax, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    # 获取旋转后图像的长和宽
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    # rot_mat是最终的旋转矩阵
    # 获取原始矩形的四个中点，然后将这四个点转换到旋转后的坐标系下
    point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
    point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
    point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
    point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
    # 合并np.array
    concat = np.vstack((point1, point2, point3, point4))
    # 改变array类型
    concat = concat.astype(np.int32)
    print concat
    rx, ry, rw, rh = cv2.boundingRect(concat)
    return rx, ry, rw, rh

# 使图像旋转60,90,120,150,210,240,300度
for angle in (60, 90, 120, 150, 210, 240, 300):
    # 指向图片所在的文件夹
    for i in os.listdir("/home/username/image"):
        # 分离文件名与后缀
        a, b = os.path.splitext(i)
        # 如果后缀名是“.jpg”就旋转图像
        if b == ".jpg":
            img_path = os.path.join("/home/username/image", i)
            img = cv2.imread(img_path)
            rotated_img = rotate_image(img, angle)
            # 写入图像
            cv2.imwrite("/home/yourname/rotate/" + a + "_" + str(angle) +"d.jpg", rotated_img)
            print "log: [%sd] %s is processed." % (angle, i)
        else:
            xml_path = os.path.join("/home/username/xml", i)
            img_path = "/home/guoyana/varied_pose/" + a + ".jpg"
            src = cv2.imread(img_path)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for box in root.iter('bndbox'):
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                x, y, w, h = rotate_xml(src, xmin, ymin, xmax, ymax, angle)
                # 改变xml中的人脸坐标值
                box.find('xmin').text = str(x)
                box.find('ymin').text = str(y)
                box.find('ymax').text = str(x+w)
                box.find('ymax').text = str(y+h)
                box.set('updated', 'yes')
            # 写入新的xml
            tree.write("/home/username/xml/" + a + "_" + str(angle) +".xml")
            print "[%s] %s is processed." % (angle, i)
