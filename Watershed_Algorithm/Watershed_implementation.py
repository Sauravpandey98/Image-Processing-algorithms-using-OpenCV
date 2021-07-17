import cv2 as cv
import numpy as np
import random as rng

img = cv.imread('/home/saurav/Desktop/oranges.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray_blurred=cv.GaussianBlur(gray,(5,5),0)
kernel=cv.getStructuringElement(cv.MORPH_RECT,(180,180))

#as the image have different illumination so we have to keep its illumination same
ill_img=cv.morphologyEx(gray_blurred,cv.MORPH_BLACKHAT,kernel,iterations=10) 
thresh=cv.inRange(ill_img,30,140,255)
cv.imshow("uniform_illumination",ill_img)
cv.imshow("thresholded",thresh)
#Morphological operations
kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
closing=cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel,iterations=3)
kernel2=cv.getStructuringElement(cv.MORPH_ELLIPSE,(11,11))
opening=cv.morphologyEx(closing,cv.MORPH_OPEN,kernel2,iterations=3)

#using distance transform to find peaks
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 3)
ret, sure_fg = cv.threshold(dist_transform,0.5*dist_transform.max(),255,0)
cv.imshow("closing",closing)
cv.imshow("OPENING",opening)
cv.normalize(dist_transform, dist_transform, 0, 1.0, cv.NORM_MINMAX)
cv.imshow("Distance transformed",dist_transform)
cv.imshow("sure_fg",sure_fg)


#creating mask of markers with giving every marker value as 1,2,3 while giving other values as 0
dist_8u = sure_fg.astype('uint8')
# Find total markers
contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# Create the marker image for the watershed algorithm
markers = np.zeros(sure_fg.shape, dtype=np.int32)
# Draw the foreground markers
for i in range(len(contours)):
    cv.drawContours(markers, contours,i,i+1,-1)
# Draw the background marker
#cv.circle(markers, (5,5), 3, (255,255,255), 1)
markers_8u = (markers*100).astype('uint8')
cv.imshow('Markers', markers_8u)
markers=cv.watershed(img, markers) #applying waterhsed algorithm
img[markers==-1]=(255,0,0)
cv.imshow("img",img) #segmented image
cv.imwrite("")
cv.waitKey(0)
cv.destroyAllWindows()
