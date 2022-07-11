# from turtle import shape
from curses import KEY_HELP
import os,time
from multiprocessing.connection import wait
import cv2
from cv2 import bitwise_not
from cv2 import rectangle
from cv2 import cvtColor
from matplotlib.pyplot import sca
import numpy as np
from math import sqrt
import math
# from edgecnn import edge_detection_cnn
from img_threshold import img_threshold

from sympy import true

from img_threshold import turn_over_gray


from collections import defaultdict
def segment_by_angle_kmeans(lines, k=2, **kwargs):

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in enumerate(lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented
def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections
def intersection_divide():
    return 
def draw_contours(img):  # conts = contours
    thresh = cv2.Canny(img, 128, 256)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = np.copy(img)
    img = cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)
    return img

def dis_pt(poly,i,j):
    a=poly[i][0]
    b=poly[j][0]
    return sqrt( (a[0]-b[0])**2+ (a[1]-b[1])**2)
def dis_line (poly,i,j,k):
    a=poly[i][0]
    b=poly[j][0]
    c=poly[k][0]
    A = b[1]-a[1]
    B = a[0]-b[0]
    C = a[1]*b[0]-a[0]*b[1]
    dst = abs(A*c[0]+B*c[1]+C)/sqrt(A*A+B*B)
    return dst
def get_max_dis_l(poly,i,j):
    maxdst = 0
    maxk = -1
    for k in range(i+1,j):
        dst=dis_line(poly,i,j,k)
        if(dst>maxdst):
            maxdst = dst
            maxk= k
    return maxk
def get_max_dis_r(poly,i,j):
    lth = len(poly)
    maxdst = 0
    maxk = -1
    for k in range(j+1,j+1+lth-(j-i+1)):
        dst=dis_line(poly,i,j,k%lth)
        if(dst>maxdst):
            maxdst = dst
            maxk= k%lth
    return maxk
def get_max_quadrilateral(poly):
    lth = len(poly)
    max_area = 0
    max_pt = 0
    for i in range(0,lth):
        for j in range(i+2,lth):
            kl = get_max_dis_l(poly,i,j)
            kr = get_max_dis_r(poly,i,j)
            xl = dis_line(poly,i,j,kl)+dis_line(poly,i,j,kr)
            area = xl*dis_pt(poly,i,j)
            if area>max_area:
                max_area=area
                max_pt= [poly[i][0],poly[kl][0],poly[j][0],poly[kr][0]]
    return max_pt
def order_points_new(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0,1]!=leftMost[1,1]:
        leftMost=leftMost[np.argsort(leftMost[:,1]),:]
    else:
        leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
    (tl, bl) = leftMost
    if rightMost[0,1]!=rightMost[1,1]:
        rightMost=rightMost[np.argsort(rightMost[:,1]),:]
    else:
        rightMost=rightMost[np.argsort(rightMost[:,0])[::-1],:]
    (tr,br)=rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], np.uint8)
def trans_img(img,max_rect):
    max_rect=sorted(max_rect,key=(lambda x:[x[1],x[0]]),reverse=False)
    if (max_rect[0][0]>max_rect[1][0]) : max_rect[0],max_rect[1]= max_rect[1],max_rect[0]
    if (max_rect[2][0]>max_rect[3][0]) : max_rect[2],max_rect[3]= max_rect[3],max_rect[2]
    a,b,c,d= (max_rect[0][0],max_rect[0][1]),(max_rect[1][0],max_rect[1][1]),(max_rect[2][0],max_rect[2][1]),(max_rect[3][0],max_rect[3][1])
    pts1 = np.float32([a,b,c,d])
    # x,y = img.shape[:2]
    w = max(dis(max_rect[0],max_rect[1]),dis(max_rect[2],max_rect[3]))
    h = max(dis(max_rect[1],max_rect[3]),dis(max_rect[0],max_rect[2]))
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    # print(pts1,pts2)
    # print(w,h)
    dst = cv2.warpPerspective(img,M,(w,h))
    return dst

def calculate(a,b,c):
    v1 = a-b
    v2 = c-b
    cc=(v1[0]*v2[0]+v1[1]*v2[1])/(sqrt(v1[0]**2+v1[1]**2))/(sqrt(v2[0]**2+v2[1]**2))
    return math.acos(cc)/math.pi*180
def angel_qualify(pts):
    angel_min = 180
    # angel_max = 0
    a=calculate(pts[0],pts[1],pts[2])
    b=calculate(pts[1],pts[2],pts[3])
    c=calculate(pts[2],pts[3],pts[0])
    d=calculate(pts[3],pts[0],pts[1])
    angel_min=min(min(a,b),min(c,d))
    print(a,b,c,d)
    if angel_min<40:
        return False
    return True
def draw_min_rect_circle(img):  # conts = contours
    # img = cv2.GaussianBlur(img,(3,3),0)
    # img = cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh=edge_detection_cnn(img)
    thresh = cv2.Canny(img, 230, 256)
    # thresh=cv2.threshold(thresh, 0, 256, cv2.THRESH_BINARY)[1]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.uint8)
    cv2.imshow('thresh',thresh)
    cv2.waitKey(5)
    # thresh = cv2.erode(thresh,kernel,iterations=1)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rectKernel)
    edge_dialect= cv2.dilate(thresh,kernel,iterations=1)
    edge_dialect= cv2.dilate(edge_dialect,kernel,iterations=1)
    # cv2.imshow('a',img)
    # cv2.waitKey(0)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img = np.copy(img)
    maxarea=0
    max_rect=0
    
    for cnt in cnts:
        # x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue
        min_rect = cv2.minAreaRect(cnt)  # min_area_rectangle
        min_rect = np.int0(cv2.boxPoints(min_rect))
        w = max(dis(min_rect[0],min_rect[1]),dis(min_rect[2],min_rect[3]))
        h = max(dis(min_rect[1],min_rect[2]),dis(min_rect[0],min_rect[3]))
        
        if(cv2.contourArea(min_rect)/(img.shape[0]*img.shape[1])<0.1): continue
        if(abs(w/h)>10 or abs(h/w)>10): continue
        if (abs(min_rect[1][0]-min_rect[0][0]) >= 1 ): 
            # scale = ((min_rect[1][1]-min_rect[0][1])/(min_rect[1][0]-min_rect[0][0]))
            # if(abs(scale)>10): continue
            # print(scale)
            # print(min_rect)
            maxarea=cv2.contourArea(min_rect)
            max_rect=min_rect
            # print(w,h)
            mask = np.zeros(img.shape[:2],np.uint8)
            mask2 = np.zeros(img.shape[:2],np.uint8)
            mask =cv2.drawContours(mask, cnt, -1, (255, 255, 255), 1)
            rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51,51))
            edges_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, rectKernel)
            # edges_close = cv2.Canny(edges_close,128,256)
            min_poly=cv2.approxPolyDP(cnt,0,closed=True)
            out_poly=cv2.convexHull(cnt)
            # if (len(out_poly)<4): continue
            pits=get_max_quadrilateral(out_poly)
            w1= min(dis(pits[0],pits[1]),dis(pits[2],pits[3]))
            w2= max(dis(pits[0],pits[1]),dis(pits[2],pits[3]))
            if w1==0 : continue
            if w2/w1>2: continue
            h1 = min(dis(pits[1],pits[3]),dis(pits[0],pits[2]))
            h2 = max(dis(pits[1],pits[3]),dis(pits[0],pits[2]))
            if h1==0 : continue
            if h2/h1>2: continue
            if h2/w2>2 or w2/h2>2: continue
            if max(h2,w2)/min(w1,h1)>5: continue
            if angel_qualify(pits) == 0 : continue
            # angel_qualify(pits)
            # print(w1,w2,h1,h2)
            # cv2.waitKey(0)
            dst=trans_img(img,pits)
            # if (cv2.meanStdDev(dst)[1][0][0])>50: continue
            # print(abs(img_horizon_degree(dst)>0.3))
            # if abs(img_horizon_degree(dst))>0.1: continue
            # dst = turn_over_gray(dst)
            # dst = img_threshold(dst)
            # dst = turn_over_gray(dst)
            # dst=cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
            cv2.imshow('te',img_threshold(dst))
            cv2.waitKey(1)
            for i in pits:
                cv2.circle(img,i,5,(255,255,255),5)
            # print(out_poly)
            img=cv2.drawContours(img, [out_poly], 0, (255, 255, 255),1)  # green
            # cv2.imshow('edge_close',edges_close)
            # cv2.waitKey(500)
            # cv2.imshow('mask2',mask2)
            # cv2.waitKey(500)
            # img_hough_p = cv2.HoughLinesP(edges_close, 0.1, math.pi / 180, 100, minLineLength = 100, maxLineGap = 10)
            # if img_hough_p is not None:
            #     mask2=draw_line(mask2,img_hough_p)
            # # edge_dialect= cv2.dilate(edges_close,kernel,iterations=1)
            #     cv2.imshow('hough_p',mask2)
            #     cv2.waitKey(1000)
            # hough
            # img_hough = cv2.HoughLines(thresh, 1, math.pi / 180,150)
            # if img_hough is not None:
            #     mask=draw_houghline(mask,img_hough)
            #     pts=segmented_intersections(segment_by_angle_kmeans(img_hough))
            #     for pt in pts:
            #         cv2.circle(img,pt[0],10,(255,255,255),10)
            #     cv2.imshow('hough',img)
            #     cv2.waitKey(1000)
    # print(max_rect)

    # maxarea/(img.shape[0]*img.shape[1])
    # cv2.drawContours(img, [max_rect], 0, (0, 0, 255))  # red
    return img

def dis(a,b):
    return int(sqrt((a[1]-b[1])**2 + (a[0]-b[0])**2))
def get_area(img):  # conts = contours
    thresh = cv2.Canny(img, 128, 256)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxarea=0
    max_rect=0
    for cnt in cnts:
        min_rect = cv2.minAreaRect(cnt)  # min_area_rectangle
        min_rect = np.int0(cv2.boxPoints(min_rect))
        if (abs(min_rect[1][0]-min_rect[0][0]) >= 1 ): 
            scale = ((min_rect[1][1]-min_rect[0][1])/(min_rect[1][0]-min_rect[0][0]))
            if(abs(scale)>0.5): continue
        if (cv2.contourArea(min_rect)>maxarea):
            maxarea=cv2.contourArea(min_rect)
            max_rect=min_rect
    if maxarea/(img.shape[0]*img.shape[1])>0.2:
        a,b,c,d= (max_rect[0][1],max_rect[0][0]),(max_rect[3][1],max_rect[3][0]),(max_rect[1][1],max_rect[1][0]),(max_rect[2][1],max_rect[2][0])
        pts1 = np.float32([a,b,c,d])
        x,y = img.shape[:2]
        w = max(dis(max_rect[0],max_rect[1]),dis(max_rect[2],max_rect[3]))
        h = max(dis(max_rect[1],max_rect[2]),dis(max_rect[0],max_rect[3]))
        pts2 = np.float32([[0,0],[h,0],[0,w],[h,w]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        print(pts1,pts2)
        print(w,h)
        dst = cv2.warpPerspective(img,M,(w,h))
        return dst
    else:
        return img

def draw_approx_hull_polygon(img, cnts):
    # img = np.copy(img)
    img = np.zeros(img.shape, dtype=np.uint8)

    # cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)  # blue

    min_side_len = img.shape[0] / 32  # 多边形边长的最小值 the minimum side length of polygon
    min_poly_len = img.shape[0] / 16  # 多边形周长的最小值 the minimum round length of polygon
    min_side_num = 3  # 多边形边数的最小值
    approxs = [cv2.approxPolyDP(cnt, min_side_len, True) for cnt in cnts]  # 以最小边长为限制画出多边形
    approxs = [approx for approx in approxs if cv2.arcLength(approx, True) > min_poly_len]  # 筛选出周长大于 min_poly_len 的多边形
    approxs = [approx for approx in approxs if len(approx) > min_side_num]  # 筛选出边长数大于 min_side_num 的多边形
    # Above codes are written separately for the convenience of presentation.
    cv2.polylines(img, approxs, True, (0, 255, 0), 2)  # green

    hulls = [cv2.convexHull(cnt) for cnt in cnts]
    # cv2.polylines(img, hulls, True, (0, 0, 255), 2)  # red

    # for cnt in cnts:
    #     cv2.drawContours(img, [cnt, ], -1, (255, 0, 0), 2)  # blue
    #
    #     epsilon = 0.02 * cv2.arcLength(cnt, True)
    #     approx = cv2.approxPolyDP(cnt, epsilon, True)
    #     cv2.polylines(img, [approx, ], True, (0, 255, 0), 2)  # green
    #
    #     hull = cv2.convexHull(cnt)
    #     cv2.polylines(img, [hull, ], True, (0, 0, 255), 2)  # red
    return img
def edge_detect(img):
    img_blur = cv2.GaussianBlur(img, (5, 5), 1)
    edges = cv2.Canny(img_blur,128, 256)
    # show('canny', edges)
    kernel = np.ones((2, 2), np.uint8)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(sqrt(img.size)/3), int(sqrt(img.size)/3)))
    edges_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rectKernel)
    edges_dilate = cv2.dilate(edges_close, kernel, iterations=3)
    contours, hierarchy = cv2.findContours(edges_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda cnts: cv2.contourArea(cnts, True), reverse=False)
    img_copy = np.zeros(img.shape,np.uint8)*255
    # res = cv.drawContours(img_copy, contours, 0, (255, 255, 255), 2)
    # show('res', res)
    # img_copy = img.copy()
    cnt = contours[0]
    epsilon = 0.03 * cv2.arcLength(cnt, True)  # epsilon占周长的比例
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    res2 = cv2.drawContours(img_copy, [approx], -1, (255, 255, 255), 5)
    return res2
def get_line_cap(a,b):
    return
def draw_line(frame,img_hough):
    for i in range(len(img_hough)):
        for x1, y1, x2, y2 in img_hough[i]:
            dx=0
            dy=0
            lth = 1
            if (x2-x1==0):
                dx = 0
                dy = 10
            elif (y2-y1==0):
                dy= 0
                dx = 10
            else:
                k=(y2-y1)/(x2-x1)
                dx = sqrt(lth**2/(k**2 + 1))
                dy = k*dx
            dx = int(dx)
            dy = int(dy)
            cv2.line(frame, (x1-dx, y1-dy), (x2+dx, y2+dy), (255, 255, 255), 1)
    return frame
def draw_houghline(frame,img_hough):
    for i in range(len(img_hough)):
        for rho,theta in img_hough[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 3000*(-b))
            y1 = int(y0 + 3000*(a))
            x2 = int(x0 - 3000*(-b))
            y2 = int(y0 - 3000*(a))
            cv2.line(frame,(x1,y1),(x2,y2),(255,255,255),3)
    return frame
def img_horizon_degree(frame):
    frame=cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    frame=bitwise_not(frame)
    kernel = np.ones((5,5),np.uint8)
    
    # frame = cv2.dilate(frame,kernel,iterations=1)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    edges_close = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, rectKernel)
    edge_dialect= cv2.dilate(edges_close,kernel,iterations=1)
    edge_dialect= cv2.dilate(edge_dialect,kernel,iterations=1)
    # cv2.imshow('test',frame)
    # thresh = cv2.Canny(frame, 128, 256,apertureSize = 3)
    mask = np.zeros(frame.shape[:2],np.uint8)
    # print(frame.shape[1]/5)
    img_hough_p = cv2.HoughLinesP(edge_dialect, 1, math.pi / 180, 100, minLineLength = int(frame.shape[1]/8), maxLineGap = 15)
    draw_line(mask,img_hough_p)
    # cv2.imshow('ts',mask)
    # cv2.waitKey(0)
    total_scale = 0
    total_angel = 0
    total_zero = 0
    total =0
    for line in img_hough_p:
        for x1,y1,x2,y2 in line:
            if abs(x2-x1)<=1: continue
            scale = (y2-y1)/(x2-x1)
            # if abs(scale<1e-6): continue
            if abs(scale)<0.4:
                if abs(scale)<1e-4: total_zero+=1
                total_scale+=scale
                total_angel+=math.atan(scale)
                total+=1
    # if (total): total-=10
    if total==0: return 0
    avg_scale = total_scale/total
    return avg_scale

def img_horizon_adjust(frame):
    frame=bitwise_not(frame)
    kernel = np.ones((5,5),np.uint8)
    
    # frame = cv2.dilate(frame,kernel,iterations=1)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    edges_close = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, rectKernel)
    edge_dialect= cv2.dilate(edges_close,kernel,iterations=1)
    edge_dialect= cv2.dilate(edge_dialect,kernel,iterations=1)
    # cv2.imshow('test',frame)
    # thresh = cv2.Canny(frame, 128, 256,apertureSize = 3)
    mask = np.zeros(frame.shape[:2],np.uint8)
    # print(frame.shape[1]/5)
    img_hough_p = cv2.HoughLinesP(edge_dialect, 1, math.pi / 180, 100, minLineLength = int(frame.shape[1]/8), maxLineGap = 15)
    draw_line(mask,img_hough_p)
    # cv2.imshow('ts',mask)
    # cv2.waitKey(0)
    total_scale = 0
    total_angel = 0
    total_zero = 0
    total =0
    for line in img_hough_p:
        for x1,y1,x2,y2 in line:
            if abs(x2-x1)<=1: continue
            scale = (y2-y1)/(x2-x1)
            # if abs(scale<1e-6): continue
            if abs(scale)<0.4:
                if abs(scale)<1e-4: total_zero+=1
                total_scale+=scale
                total_angel+=math.atan(scale)
                total+=1
    # if (total): total-=10
    print(total_zero)
    avg_scale = total_scale/total
    # print(avg_scale)
    angel=math.atan(avg_scale)/math.pi*180
    print(angel)
    print(total_angel/total*180/math.pi)
    print(avg_scale)
    avg_scale = math.tan(total_angel/total)
    print(avg_scale)
    dw=avg_scale*frame.shape[1]*(math.cos(math.atan(avg_scale)))
    dh=avg_scale*frame.shape[0]*(math.cos(math.atan(avg_scale)))
    # print(dw,dh)
    dw,dh=math.floor(abs(dw)),math.floor(abs(dh))
    # dw*=10
    # dh*=10
    h,w= frame.shape[:2]
    M = cv2.getRotationMatrix2D((int(frame.shape[1]/2), int(frame.shape[0]/2)), angel, 1.0)
    # frame = bitwise_not(frame)
    rotated = cv2.warpAffine(frame, M, (w+dw,h+dh))
    # rotated = rotated[dh:h-dh,dw:w-dw]
    rotated = bitwise_not(rotated)
    return avg_scale,rotated
def run():
    files = os.listdir('lpic')
    # print(files)
    for f in files:
        img = cv2.imread('lpic/'+f,0)
        img = img_threshold(img)
        # img = img_horizon_adjust(img)[1]
        cv2.imwrite('lpic/'+f,img)
    return
    # cap = cv2.VideoCapture('test.mp4')
    # while cap.isOpened():
    #     img = cap.read()[1]
    #     # img = cv2.imread('test/hn.jpg',0)
    #     # img = cvtColor(img,cv2.COLOR_BGR2GRAY)
    #     cv2.imshow('test',draw_min_rect_circle(img))
    #     cv2.waitKey(1)
        # thresh = cv2.Canny(img, 128, 256,apertureSize = 3)
        # kernel = np.ones((5,5),np.uint8)
    # img=draw_min_rect_circle(img)
    # test1=draw_contours(img)
        # test2=draw_min_rect_circle(img)
    # rotated= img_horizon_adjust(frame)[1]
    # rotated = draw_contours(frame)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # img_hough = cv2.HoughLinesP(thresh, 1, math.pi / 180, 100, minLineLength = 100, maxLineGap = 15)
    # # img_hough = cv2.HoughLines(thresh, 1, math.pi / 180,200)

    # (x, y, w, h) = (np.amin(img_hough, axis = 0)[0,0], np.amin(img_hough, axis = 0)[0,1], np.amax(img_hough, axis = 0)[0,0] - np.amin(img_hough, axis = 0)[0,0], np.amax(img_hough, axis = 0)[0,1] - np.amin(img_hough, axis = 0)[0,1])
    # frame = frame[y:y+h,x:x+w]
    # thresh = cv2.Canny(frame, 128, 256,apertureSize = 3)
    
    # for dot in intersections:
        # cv2.circle(mask,dot[0],1,(255,255,255),1)
    # draw_houghline(mask,img_hough)
    # kernel = np.ones((2, 2), np.uint8)
    # rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    # edges_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, rectKernel)
    # edges_dilate = cv2.dilate(edges_close, kernel, iterations=3)
    # con=cv2.findContours(edges_close,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # cv2.drawContours(frame,con,-1,(0,255,0),2)
    # cv2.drawContours(frame,img_hough_p,(255,255,255),2)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # frame=draw_min_rect_circle(frame, contours)
    # cv2.rectangle(frame,[x,y],[x+w,y+h], (0, 0, 0),2)
    # cv2.imshow('test.jpg',mask)
    # cv2.imshow('a',test2)
    # cv2.waitKey(0)
   


if __name__ == '__main__':
    run()
pass