import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while(1):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([60,55,50])
    upper_blue = np.array([130,255,255])

    mask = cv2.inRange(hsv,lower_blue, upper_blue)
    res = cv2.bitwise_and(frame,frame,mask=mask)

    laplacian = cv2.Laplacian(frame,cv2.CV_64F)
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
    # ksize is the kernel size and 1,0 and 0,1 for x and y axis respectively

##    addition = cv2.add(sobelx,sobely)
##    cv2.imshow('addition',addition)

    cv2.imshow('Original',frame)
    cv2.imshow('mask',mask)
##    cv2.imshow('laplacian',laplacian)
##    cv2.imshow('sobelx',sobelx)
##    cv2.imshow('sobely',sobely)

    edges = cv2.Canny(frame,110,200)
    cv2.imshow('Edges', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

