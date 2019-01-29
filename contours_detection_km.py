import numpy as np
import cv2 as cv
#from matplotlib import pyplot as plt

#read image
raw_img = cv.imread('mult_colored_balls4.jpg', -1)
hsv_img = cv.cvtColor(raw_img, cv.COLOR_BGR2HSV) #convert to hsv

#range of yellow in HSV
lower_yellow = np.array([19, 150, 189])
upper_yellow = np.array([30, 255, 255])

mask_img = cv.inRange(hsv_img, lower_yellow, upper_yellow) #black and white

#Blur image - Bilateral preserves edges but is slower than Gaussian
#Parameters are just guesses - idk what parameters to use
bilat_filt_img = cv.bilateralFilter(mask_img, 5, 275, 275)
edges_img = cv.Canny(bilat_filt_img, 100, 200)

#gives a list of contours as outputs, fitler the list with particular shape
_, contours, hierarchy = cv.findContours(edges_img, cv.RETR_TREE,
                                         cv.CHAIN_APPROX_NONE)
contours_list = []
#just initialize closest_circle to something
closest_circle = contours[0]

#we will be searching for a polyDp as close to 8 as possible (closest to circle)
#40 is arbitary, can probably make this value smaller but this is good for now
polyDp = 40

for cont in contours:
    approx = cv.approxPolyDP(cont, 0.03*cv.arcLength(cont, True), True)
    
    #i don't use area rn, maybe i should set a minimum tho
    area = cv.contourArea(cont)

    if ((len(approx) >8) and ((len(approx) < polyDp))):
      	polyDp = len(approx)

contours_list.append(closest_circle)
cimg = cv.cvtColor(edges_img, cv.COLOR_GRAY2BGR)

cv.drawContours(cimg, contours_list,  -1, (158,255,0), 2)

cv.imshow('raw_img', raw_img)
cv.imshow('mask_img', mask_img)
cv.imshow('bilat_filt_img', bilat_filt_img)
cv.imshow('edges_img', edges_img)
cv.imshow('detected circles', cimg)

k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
