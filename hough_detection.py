import cv2 as cv
import numpy as np
import time
#from matplotlib import pyplot as plt

# The image we are trying to read
start = time.time();
img = cv.imread('mult_colored_balls.jpg', -1)

# Convert RGB to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Range of yellow in HSV
lower_yellow = np.array([19, 150, 189]) # [50, 205, 205] RGB
upper_yellow = np.array([30, 255, 255]) # [0, 255, 255] RGB

# Threshold the HSV image to get only yellow colors
mask = cv.inRange(hsv, lower_yellow, upper_yellow)
smooth = cv.blur(mask, (5, 5))
edges = cv.Canny(smooth, 100, 200)

cimg = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

#1, 150, param1 = 200, param2 = 75
circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 150,
                            param1 = 200, param2 = 75,
                            minRadius = 0, maxRadius = 0)
if(circles is not None):
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        #draw the outer circle
        cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        #draw the center of the circle
        cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
print(time.time() - start)
#show the image
#cv.imshow('img', img)
#cv.imshow('mask', mask)
#cv.imshow('smooth', smooth)
#cv.imshow('edges', edges)
cv.imshow('detected circles', cimg)
#cv.imshow('mask', mask)
k = cv.waitKey(0)
if k == 27:
    cv.destoryAllWindows()
