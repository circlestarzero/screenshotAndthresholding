from ast import BitAnd
from cProfile import label
from cgi import test
import sys
from cv2 import bitwise_and, bitwise_not, bitwise_or
import pandas as pd
from sklearn.cluster import KMeans
import keyboard
import cv2
import numpy as np
import subprocess
import urllib
import urllib.request as ur
from PIL import Image, ImageGrab
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, qAbs, QRect
from PyQt5.QtGui import QPen, QPainter, QColor, QGuiApplication,QImage
from img_threshold import img_threshold
import time
localtime = time.localtime(time.time())
file_date_name=str(localtime.tm_year)+'_'+str(localtime.tm_mon)+'_'+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+'_'+str(localtime.tm_min)+'_'+str(localtime.tm_sec)+'.jpg'
def three_split(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros((height,width),np.uint8)
    d1 = img.reshape(-1,1)
    lth = d1.shape[0]
    n=3
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

def mask_over(img,imgbg):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros((height,width),np.uint8)
    d1 = img.reshape(-1,1)
    d2 = imgbg.reshape(-1,1)
    lth = d1.shape[0]
    for i in range(lth):
        if d2[i] > 250:
            d1[i] = 255
    n=2
    km =KMeans(n_clusters=n)
    km.fit(d1)
    lss  = []
    ls = [0,1,2,3,4,5,6,7,8,9,10]
    for i in range(n):
        lss.append((km.cluster_centers_[i][0],i))
    lss.sort()
    for i in range(n):
        ls[lss[i][1]]=i

    # ser=pd.Series(km.labels_)
    # res0Series = pd.Series(km.labels_)
    # res0 = res0Series[km.values == 1]
    d1 = np.array(pd.Series(km.labels_).to_list())
    for i in range(lth):
        d1[i] = int(255.0*ls[d1[i]]/(n-1))
    d1 = d1.reshape((height,width))
    print(d1.shape)
    return d1
def turn_over_gray(img):
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros((height,width),np.uint8)
    dst[:] = 255
    dst = dst - img
    return dst
def totalwt(img):
    x,y= img.shape
    wt = 0
    for i in range(x):
        for j in range(y):
            if img[i,j]>250:
                wt+=1
    return  wt/(x*y)

def threaholding_image_1(img):
    sharpen = cv2.GaussianBlur(img, (0,0), 3)
    sharpen = cv2.addWeighted(img, 1.5, sharpen, -0.5, 0)
    gray = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
    return gray
def threaholding_image(img):
    test=cv2.threshold(img, 0, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    nzero = cv2.countNonZero(test)
    scale = nzero / img.size
    if scale <0.5:
        img=turn_over_gray(img)
        test=bitwise_not(test)
    img=bitwise_or(test,threaholding_image_1(img))
    return img
def gray_image(img):
    imgbackup = img
    img = three_split(img)
    if totalwt(img) <0.5:
        img=turn_over_gray(img)
        imgbackup=turn_over_gray(imgbackup)
    img=mask_over(imgbackup,img)
    return img
def color_split(img,n=5):
    test=cv2.threshold(img, 0, 255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    if totalwt(test) <0.5:
        img=turn_over_gray(img)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    dst = np.zeros((height,width),np.uint8)
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
class CaptureScreen(QWidget):
    # 初始化变量
    beginPosition = None
    endPosition = None
    fullScreenImage = None
    captureImage = None
    isMousePressLeft = None
    scale=None
    pix=None
    painter = QPainter()
    def __init__(self):
        super(QWidget, self).__init__()
        self.initWindow()   # 初始化窗口
        self.captureFullScreen()    # 获取全屏

    def initWindow(self):
        self.setMouseTracking(True)     # 鼠标追踪
        self.setCursor(Qt.CrossCursor)  # 设置光标
        self.setWindowFlag(Qt.FramelessWindowHint)  # 窗口无边框
        self.setWindowState(Qt.WindowFullScreen)    # 窗口全屏

    def captureFullScreen(self):
        self.fullScreenImage = QGuiApplication.primaryScreen().grabWindow(QApplication.desktop().winId())
        pix = self.fullScreenImage.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        self.scale=int(pix.width()/QApplication.desktop().screenGeometry().width())
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.beginPosition = event.pos()
            self.isMousePressLeft = True
        if event.button() == Qt.RightButton:
            # 如果选取了图片,则按一次右键开始重新截图
            if self.captureImage is not None:
                self.captureImage = None
                self.paintBackgroundImage()
                self.update()
            else:
                self.close()

    def mouseMoveEvent(self, event):
        if self.isMousePressLeft is True:
            self.endPosition = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.endPosition = event.pos()
        self.isMousePressLeft = False
        if self.captureImage is not None:
            self.saveImage()
            self.close()

    def mouseDoubleClickEvent(self, event):
        if self.captureImage is not None:
            self.saveImage()
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            if self.captureImage is not None:
                self.saveImage()
                self.close()

    def paintBackgroundImage(self):
        shadowColor = QColor(0, 0, 0, 100)  # 黑色半透明
        self.painter.drawPixmap(0, 0, self.fullScreenImage)
        self.painter.fillRect(self.fullScreenImage.rect(), shadowColor)     # 填充矩形阴影

    def paintEvent(self, event):
        self.painter.begin(self)    # 开始重绘
        self.paintBackgroundImage()
        penColor = QColor(30, 144, 245)     # 画笔颜色
        self.painter.setPen(QPen(penColor, 1, Qt.SolidLine, Qt.RoundCap))    # 设置画笔,蓝色,1px大小,实线,圆形笔帽
        if self.isMousePressLeft is True:
            pickRect = self.getRectangle(self.beginPosition, self.endPosition)   # 获得要截图的矩形框
            pickRect_p=self.getRectangle_p(self.beginPosition, self.endPosition)
            self.captureImage = self.fullScreenImage.copy(pickRect_p)         # 捕获截图矩形框内的图片
            self.painter.drawPixmap(pickRect.topLeft(), self.captureImage)  # 填充截图的图片
            self.painter.drawRect(pickRect)     # 画矩形边框
        self.painter.end()  # 结束重绘

    def getRectangle(self, beginPoint, endPoint):
        pickRectWidth = int(qAbs(beginPoint.x() - endPoint.x()))
        pickRectHeight = int(qAbs(beginPoint.y() - endPoint.y()))
        pickRectTop = beginPoint.x() if beginPoint.x() < endPoint.x() else endPoint.x()
        pickRectLeft = beginPoint.y() if beginPoint.y() < endPoint.y() else endPoint.y()
        pickRect = QRect(pickRectTop, pickRectLeft, pickRectWidth, pickRectHeight)
        # 避免高度宽度为0时候报错
        if pickRectWidth == 0:
            pickRect.setWidth(2)
        if pickRectHeight == 0:
            pickRect.setHeight(2)
        return pickRect
    def getRectangle_p(self, beginPoint, endPoint):
        pickRectWidth = int(qAbs(beginPoint.x() - endPoint.x()))
        pickRectHeight = int(qAbs(beginPoint.y() - endPoint.y()))
        pickRectTop = beginPoint.x() if beginPoint.x() < endPoint.x() else endPoint.x()
        pickRectLeft = beginPoint.y() if beginPoint.y() < endPoint.y() else endPoint.y()
        pickRect = QRect(pickRectTop*self.scale, pickRectLeft*self.scale, pickRectWidth*self.scale, pickRectHeight*self.scale)
        # 避免高度宽度为0时候报错
        if pickRectWidth == 0:
            pickRect.setWidth(2)
        if pickRectHeight == 0:
            pickRect.setHeight(2)
        return pickRect
    def saveImage(self):
        pix=self.captureImage.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        width = pix.width()
        height = pix.height()
        ptr = pix.bits()
        ptr.setsize(height * width * 4)
        img = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype("uint8")
        # down_points = (int(img.shape[1]/img.shape[0]*500), 500)
        # img = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)
        img=img_threshold(img)
        # img=color_split(img,6)
        cv2.imwrite('pic/'+file_date_name, img)
        cv2.imwrite('result.jpg', img)
        subprocess.run(["osascript", "-e", 'set the clipboard to (read (POSIX file "result.jpg") as JPEG picture)'])
if __name__ == "__main__":
    app = QApplication(sys.argv)
    windows = CaptureScreen()
    windows.show()
    sys.exit(app.exec_())

