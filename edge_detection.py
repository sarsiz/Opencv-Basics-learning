import numpy as np
import cv2

cap = cv2.VideoCapture(0)
z=1
while(z):
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # doing with blue  for red.. use  30,150,50   255 255 180
    lower_red = np.array([50,60,70])
    upper_red = np.array([130,255,255])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame,mask=mask)

##    cv2.imshow('frame', frame)
##    cv2.imshow('mask', mask)
##    cv2.imshow('res', res)


    kernel = np.ones((15,15),np.float32)/225
    smoothed = cv2.filter2D(res,-1,kernel)
    cv2.imshow('Original',frame)
    cv2.imshow('Averaging',smoothed)

    # Gaussian Blurring
    blur = cv2.GaussianBlur(res, (15,15), 0)
#    cv2.imshow('Gaussian Blurring', blur)

    # Median Blurring
    median = cv2.medianBlur(res,15)
#    cv2.imshow('Median Blurring', median)

    # Bilateral Blurring
    bilateral = cv2.bilateralFilter(res,15,75,75)
#    cv2.imshow('bilateral Blur',bilateral)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask,kernel,iterations = 1)
    dilation = cv2.dilate(mask,kernel,iterations = 1)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow('Original',frame)
    #cv2.imshow('Mask',mask)

    # For False Positive and Negatives Respectively
    cv2.imshow('Opening',opening)
    cv2.imshow('Closing',closing)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        z = 0

cv2.destroyAllWindows()
cap.release()

##import cv2
##import numpy as np
##import matplotlib.pyplot as plt
##
##bgr_img = cv2.imread('C:/Users/sazas/Desktop/face.jpg')
##b,g,r = cv2.split(bgr_img)
### rgb_img = cv2.merge([r,g,b]) # switch it to go rgb
##
##rgb_img = bgr_img[:,:,::-1]
##
###gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
###cv2.imwrite('face_grayscale.jpg', gray_img)
##
##plt.imshow(rgb_img)
##plt.xticks([]), plt.yticks([]) # to hide tick values on x and y axis
##plt.show()
##
##while True:
##    k=cv2.waitKey(0) & 0xFF
##    if k == 27:
##        break
##        
##cv2.destroyAllWindows()
##
