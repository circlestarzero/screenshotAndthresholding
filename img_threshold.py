import cv2 as cv
import numpy as np
from cv2 import bitwise_and, bitwise_not, bitwise_or,bitwise_xor
from math import sqrt
import os
def turn_over_gray(img):
    dst = np.ones(img.shape,np.uint8)*255
    dst = dst - img
    return dst
def threaholding_image_1(img):
    sharpen = cv.GaussianBlur(img, (0,0), 3)
    sharpen = cv.addWeighted(img, 1.5, sharpen, -0.5, 0)
    gray = cv.adaptiveThreshold(sharpen, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 15)
    return gray
def threaholding_image(img):
    test=cv.threshold(img, 0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
    img=bitwise_or(test,threaholding_image_1(img))
    return img
def totalwt(test_img):
    nzero = cv.countNonZero(test_img)
    scale = nzero / test_img.size
    return scale
def background_avg(img):
    result = cv.medianBlur(img,int(sqrt(img.size)/20)|1)
    mask = np.ones(img.shape,np.uint8)*int(255)
    img=cv.add(img,(mask-result))
    # cv.imwrite('background.jpg',result)
    return img

def edge_detect(img):
    mask = np.zeros(img.shape,np.uint8)
    contours = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[0]
    maxarea = 0
    maxindex = 0
    for i in range(len(contours)):
        area=cv.contourArea(contours[i])
        if(area>maxarea):
            maxarea = area
            maxindex = i
    mask=cv.drawContours(mask,contours,maxindex,color=255)
    # cv.imwrite('mask.jpg',mask)
    return
def edge_blur(img):
    mask = np.zeros(img.shape,np.uint8)
    contours = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[0]
    for i in range(len(contours)):
        mask=cv.drawContours(mask,contours,i,color=255)
    
    mask=cv.GaussianBlur(mask,(3,3),0)
    mask = cv.bitwise_not(mask)
    mask=cv.threshold(mask,180,255,cv.THRESH_BINARY)[1]
    return mask
def img_threshold(img):
    test_img=cv.threshold(img, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
    if totalwt(test_img) <0.5:
        img=turn_over_gray(img)
    img=background_avg(img)
    # edge_detect(img)
    ke = cv.threshold(img, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)[0]
    t1=threaholding_image(img)
    t2 = cv.threshold(img, min(ke+20,250) , 255,cv.THRESH_BINARY)[1]
    blur = cv.GaussianBlur(t1,(5,5),0)
    t3 = cv.threshold(blur,254,255,cv.THRESH_BINARY)[1]
    tt=bitwise_xor(t1,t2)
    tt=bitwise_not(tt)
    edge0=bitwise_or(t3,tt)
    edge1 = edge_blur(t1)
    edge=bitwise_or(edge1,edge0)
    fn=bitwise_and(edge,t1)
    fn=cv.resize(fn,(int(fn.shape[1]*1.2),int(fn.shape[0]*1.2) ), interpolation = cv.INTER_CUBIC)
    return fn

if __name__ =="__main__":
    root = 'lpic'
    files = os.listdir(root)
    for f in files:
        img = cv.imread(root+'/'+f,0)
        img = img_threshold(img)
        # img = img_horizon_adjust(img)[1]
        cv.imwrite(root+'/'+f,img)