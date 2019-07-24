# -*- coding: utf-8 -*-
"""
Created on Tue May 14 11:26:14 2019

@author: bittu
"""
import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx

#get mouse variable and screen size
mouse = Controller()
app = wx.App(False)
(sx,sy) = wx.GetDisplaySize()
(camx,camy) = (320,240) #setting the image resolution of captured image 

cam = cv2.VideoCapture(0)
cam.set(3,camx)
cam.set(4,camy)

#boundary conditions for green color H,S,V
lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])

#for removing noises window size
kernelOpen = np.ones((5,5))
kernelClose = np.ones((20,20))

mLocOld = np.array([0,0])
mouseLoc = np.array([0,0])
DampingFactor = 2 #factor must be >1
#mouseLoc = mLocOld+(targetLoc-mLocOld)/dampingfactor
#damping factor decides how smooth mouse movement is same as alpha in gradient descend.

pinchFlag = 0
openx,openy,openw,openh=(0,0,0,0)

while(True):
    ret, img = cam.read()
    #img = cv2.resize(img,(340,220))
    
    #convert BGR to HSV  HSV -> Hue, Saturation, Value.
    #The hue of a pixel is an angle from 0 to 359 the value of each angle decides the color of the pixel 
    #The Saturation is basically how saturated the color is, and the Value is how bright or dark the color is
    #So the range of these are as follows
    #Hue is mapped – >0º-359º as [0-179]
    #Saturation is mapped ->  0%-100% as [0-255]
    #Value is 0-255 (there is no mapping)
    
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    #create a filter or mask to filer out a specific color here we filter green color
    mask = cv2.inRange(imgHSV,lowerBound,upperBound)
    
    # removing the noises in the image. we use a window and slide it over the image whenever window find a white spot smaller than its size it covers it coz its due to noise
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    #opposite of maskopen it fill white if it finds black smaller than window size
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    
    maskFinal = maskClose
    #in the maskFinal it detect where the white space is showing and return its contour we can use this contour and draw it around object in actual image
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #if there are 2 contours then open gesture if 1 than close gesture
    
    if(len(conts)==2):
        if(pinchFlag==1):
            pinchFlag=0
            mouse.release(Button.left)
        #get bounding coordinates of both contours and draw rectangle
        x1,y1,w1,h1 = cv2.boundingRect(conts[0])
        x2,y2,w2,h2 = cv2.boundingRect(conts[1])
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
        
        #drawing line between both contour
        cx1 = x1+(int)(w1/2)
        cx2 = x2+(int)(w2/2)
        cy1 = y1+(int)(h1/2)
        cy2 = y2+(int)(h2/2)
        cv2.line(img,(cx1,cy1),(cx2,cy2),(255,0,0),2)
        
        #get center of above line to act as refrence to mouse position
        cx = (int)((cx1+cx2)/2)
        cy = (int)((cy1+cy2)/2)
        cv2.circle(img,(cx,cy),2,(0,0,255),2)
        
        mouseLoc = mLocOld+((cx,cy)-mLocOld)/DampingFactor
        
        #moving mouse code
        # we use sx-() to invert the movement
        mouse.position = (sx-(mouseLoc[0]*sx/camx),mouseLoc[1]*sy/camy)#camera screen is small while actuall screen is large so we need to convert coordinates based on resolution
        
        mLocOld = mouseLoc
        
        openx,openy,openw,openh = cv2.boundingRect(np.array([[[x1,y1],[x1+w1,y1+h1],[x2,y2],[x2+w2,y2+h2]]]))
        
        
    elif(len(conts) == 1):
        x,y,w,h = cv2.boundingRect(conts[0])
        if(pinchFlag==0):
            if(abs((w*h-openw*openh)/(w*h))<0.3):                
                pinchFlag=1
                mouse.press(Button.left)
                openx,openy,openw,openh=(0,0,0,0)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cx = x+(int)(w/2)
            cy = y+(int)(h/2)
            cv2.circle(img,(cx,cy),(int)(w/2),(0,0,255),2)
            #smooth movement
            mouseLoc = mLocOld+((cx,cy)-mLocOld)/DampingFactor
        
            #moving mouse code
            #we use sx-(cx*sx/camx) to invert the coordinates since webcam capture mirror image
            mouse.position = (sx-(mouseLoc[0]*sx/camx),mouseLoc[1]*sy/camy)#camera screen is small while actuall screen is large so we need to convert coordinates based on resolution
        
            mLocOld = mouseLoc
        
 
   
    cv2.imshow('image',img)
    #cv2.waitKey(10)
    
    if(cv2.waitKey(1) == ord('q')):
        break;#ord return unicode of q we exit on pressing q
del app
cam.release()
cv2.destroyAllWindows()