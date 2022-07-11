import cv2 as cv
from cv2 import cvtColor
from cv2 import CV_8U
from matplotlib.pyplot import contour
import numpy as np
from cv2 import bitwise_and, bitwise_not, bitwise_or,bitwise_xor
from math import sqrt
# from edge_detection import get_area
import pandas as pd
from sklearn.cluster import KMeans
# from edge_detection import img_horizon_adjust
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
    result = cv.medianBlur(img,int(sqrt(img.size)/30)|1)

    mask = np.ones(img.shape,np.uint8)*int(255)
    img=cv.add(img,(mask-result))
    return img
def edge_blur(img):
    mask = np.zeros(img.shape,np.uint8)
    contours = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)[0]
    for i in range(len(contours)):
        mask=cv.drawContours(mask,contours,i,color=255)
    mask=cv.GaussianBlur(mask,(3,3),0)
    mask = cv.bitwise_not(mask)
    mask=cv.threshold(mask,180,255,cv.THRESH_BINARY)[1]
    return mask
def edge_detect(img):
    img_blur = cv.GaussianBlur(img, (5, 5), 1)
    edges = cv.Canny(img_blur,128, 256)
    # show('canny', edges)
    kernel = np.ones((2, 2), np.uint8)
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (int(sqrt(img.size)/3), int(sqrt(img.size)/3)))
    edges_close = cv.morphologyEx(edges, cv.MORPH_CLOSE, rectKernel)
    edges_dilate = cv.dilate(edges_close, kernel, iterations=3)
    contours, hierarchy = cv.findContours(edges_dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda cnts: cv.contourArea(cnts, True), reverse=False)
    img_copy = np.zeros(img.shape,np.uint8)*255
    # res = cv.drawContours(img_copy, contours, 0, (255, 255, 255), 2)
    # show('res', res)
    # img_copy = img.copy()
    cnt = contours[0]
    epsilon = 0.03 * cv.arcLength(cnt, True)  # epsilon占周长的比例
    approx = cv.approxPolyDP(cnt, epsilon, True)
    res2 = cv.drawContours(img_copy, [approx], -1, (255, 255, 255), 5)
    return res2
def img_threshold(img):
    # img = get_area(img)
    test_img=cv.threshold(img, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
    if totalwt(test_img) <0.5:
        img=turn_over_gray(img)
    img=background_avg(img)
    ke = cv.threshold(img, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)[0]
    t1=threaholding_image(img)
    # kernel = np.ones((5,5),np.uint8)
    # t1 = cv.morphologyEx(t1,cv.MORPH_OPEN,kernel)
    t2 = cv.threshold(img, min(ke+5,250) , 255,cv.THRESH_BINARY)[1]
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

def color_split(img,n=5):
    test=cv.threshold(img, 0, 255,cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
    if totalwt(test) <0.5:
        img=turn_over_gray(img)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    # dst = np.zeros((height,width),np.uint8)
    d1 = img.reshape(-1,1)
    lth = d1.shape[0]
    km =KMeans(n_clusters=n)
    km.fit(d1)
    lss  = []
    ls = [0,1,2,3,4,5,6,7,8,9,10]
    for i in range(n):
        lss.append((km.cluster_centers_[i][0],i))
    lss.sort()
    for i in range(n):
        ls[lss[i][1]]=i
    d1 = np.array(pd.Series(km.labels_).to_list())
    for i in range(lth):
        d1[i] = int(255.0*ls[d1[i]]/(n-1))
    d1 = d1.reshape((height,width))
    return d1
if __name__ =="__main__":
    img = cv.imread('test/ui.jpg',0)
    # img = get_area(img)
    img = img_threshold(img)
    # img = cvtColor(img,colorgr)
    avg_scale= 1
    # while(abs(avg_scale)>0.1):
    # avg_scale,img=img_horizon_adjust(img)
    cv.imwrite('test/s.jpg',img)
    # cap = cv.VideoCapture('test/test.mp4')
    # while cap.isOpened():
    #     frame = cap.read()[1]
    #     img= get_area(frame)
    #     cv.imshow('test.jpg',img)
    #     cv.waitKey(1000)
    # cv.imwrite('aa.jpg',img_processed)