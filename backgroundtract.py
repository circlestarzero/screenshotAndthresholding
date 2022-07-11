from matplotlib.pyplot import contour
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('test1.mp4')
fgbg = cv.createBackgroundSubtractorMOG2(history=5, varThreshold=10, detectShadows=False)
while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv.threshold(fgmask,220,255,cv.THRESH_BINARY)[1]
    background = fgbg.getBackgroundImage()
    kernel = np.ones((3,3),np.uint8)
    rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2), (-1, -1))
    # edge_dialect= cv.dilate(fgmask,kernel,iterations=1)
    # edge_dialect= cv.dilate(fgmask,kernel,iterations=1)
    fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, rectKernel)
    # cny = cv.Canny(fgmask,0,256)
    cnts = cv.findContours(fgmask,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        min_poly=cv.approxPolyDP(cnt,10,closed=True)
        out_poly=cv.convexHull(cnt)
        if cv.arcLength(min_poly,closed=True)< 100:
            continue
        # rect = cv.minAreaRect(cnt)
        # cv.ellipse(frame, rect, (0, 255, 0), 2, 8)
        # cv.circle(frame, (np.int32(rect[0][0]), np.int32(rect[0][1])), 2, (255, 0, 0), 2, 8, 0)
        # print(cv.contourArea(out_poly)/(frame.shape[0]*frame.shape[1]))
        cv.drawContours(frame,[min_poly],-1, 255, -1)
    # cv.drawContours(frame,cnt,-1,(255,0,0),2)
    cv.imshow('input', frame)
    cv.imshow('mask',fgmask)
    cv.imshow('background', background)
    cv.waitKey(1)
cap.release()
cv.destroyAllWindows()