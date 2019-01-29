import numpy as np
import cv2 as cv
#from matplotlib import pyplot as plt
#TODO run against test images to make sure yellow color bounds and area min are good
#read image
raw_img = cv.imread('mult_colored_balls3.jpg', -1)
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
most_circular_contour = None

#we will be searching for a polyDp as close to 8 as possible (closest to circle)
#40 is arbitary, can probably make this value smaller but this is good for now
lowest_error = 0.3
#minimum area a circul must have to be the ball so we dont pick up random stuff
#this is measuered as pixels, so we may need to change this depending on the 
#image quality the rover captures
area_minimum = 100
#one issue: doesn't seem to fill sometimes, maybe contour of circle isn't always connected
for cont in contours:
    #is it faster to check if area is above a threshold first? or just fill all?
    area = cv.contourArea(cont)
    #if(area > area_minimum):
    cv.fillPoly(edges_img, pts =[cont], color=(255,255,255))

#now I should have far fewer contours to work with
_, contours, hierarchy = cv.findContours(edges_img, cv.RETR_TREE, 
    cv.CHAIN_APPROX_NONE)

#at this point, may not be many left in contours list because they got filled
#so, this loop will not loop very many times
for cont in contours:
    vertices = cv.approxPolyDP(cont, 0.01*cv.arcLength(cont, True), True)
    #check if this is the most circular contour so far 
    center, radius = cv.minEnclosingCircle(cont);
    area = cv.contourArea(cont)
    if(area > area_minimum):
        area_theoretical = 3.14 * radius * radius
        percent_error = abs((area-area_theoretical)/area_theoretical)
        if (percent_error < lowest_error):
            #check to make sure this is a circle big enough to not be just a random speck
            lowest_error = percent_error
            most_circular_contour = cont

if(most_circular_contour is not None):
	contours_list.append(most_circular_contour)




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
