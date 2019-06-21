import cv2
import numpy as np
import Tkinter as tk       
import time

camera = cv2.VideoCapture(0)
camera.set(3,320)
camera.set(4,240)
    
# Threshold the HSV image
lower_black = np.array([0,0,38])
upper_black = np.array([255,255,255])

frame = 0

while True:
    _,image = camera.read()
    im = cv2.resize(image,(128, 128))

    blur = cv2.GaussianBlur(im, (5,5),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # save & viwe images
    cv2.imwrite("data_images/"+str(frame)+".jpg", im)
    cv2.imwrite("data_mask/"+str(frame)+".jpg", mask)
    #cv2.imshow('full window',im)
    #cv2.imshow('roi_window',mask)
		
           
    frame += 1
    time.sleep(0.2)
    
    if frame == 600:
        break
    
cv2.destroyAllWindows()
print("finish")    
                





