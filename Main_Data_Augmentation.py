# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:08:30 2019

@author: SAI
"""

import cv2
import os
import xml.etree.ElementTree as ET
import pdb

for img in os.listdir("D:\PythonWorkSpace\VOCMaker\Data_Augmentation\VOCdevkit1\JPEGImages"):
    a, b = os.path.splitext(img)
    if b == ".jpg":
        img = cv2.imread("D:\PythonWorkSpace\VOCMaker\Data_Augmentation\VOCdevkit1\JPEGImages/" + img)
        tree = ET.parse("D:\PythonWorkSpace\VOCMaker\Data_Augmentation\VOCdevkit1\Annotations/" + a + ".xml")
        root = tree.getroot()
        for box in root.iter('bndbox'):
            x1 = float(box.find('xmin').text)
            y1 = float(box.find('ymin').text)
            x2 = float(box.find('xmax').text)
            y2 = float(box.find('ymax').text)

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), [0,255,0], 2)
            cv2.imshow("test", img)
        if 1 == cv2.waitKey(0):
            pass
"""
--------------------- 
作者：木_凌 
来源：CSDN 
原文：https://blog.csdn.net/u014540717/article/details/53301195 
版权声明：本文为博主原创文章，转载请附上博文链接！
"""