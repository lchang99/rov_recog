import numpy as np
import cv2 as cv
import time
#TODO run against test images to make sure yellow color bounds and area min are good
#read image
start = time.time();
raw_img = cv.imread('yarn.jpg', -1)
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
#a sorting algorithm on the above will improve the speed, merge sort
contours_list = []

most_circular_contour = None

#we will be searching for the thing closest to a circle, this is the max allowed percent error to be considered a circle
lowest_error = 0.3
#minimum area we will consider to be the ball so we ignore specks
#this is measuered as pixels, so we may need to change this depending on the image quality the rover captures
area_minimum = 100

for cont in contours:    
    area = cv.contourArea(cont)
    if(area > area_minimum):
    	#check first if area is above the minimum so we only do this to possible balls

        cv.fillPoly(edges_img, pts =[cont], color=(255,255,255))
        #so for some reason the above doesn't tottally fill in lines, maybe unclosed contours idk, the below may be faster anyways

        #This thickens the lines, and I think this closes the contours for when we call findContours again
        cv.polylines(edges_img, cont, True, 255, 5, 8, 0)
#now I should have far fewer contours to work with
_, contours, hierarchy = cv.findContours(edges_img, cv.RETR_TREE, 
    cv.CHAIN_APPROX_NONE)

#at this point, may not be many left in contours list because they got filled
#so, this loop will not loop very many times
for cont in contours:
    area = cv.contourArea(cont)
    
    if(area > area_minimum):
        #draw a circle that best fits the points
        center, radius = cv.minEnclosingCircle(cont);

        area_theoretical = 3.14 * radius * radius
        percent_error = abs((area-area_theoretical)/area_theoretical)
        
        #check if this is the most circular contour so far
        if (percent_error < lowest_error):      
            lowest_error = percent_error
            most_circular_contour = cont

if(most_circular_contour is not None):
	contours_list.append(most_circular_contour)




cimg = cv.cvtColor(edges_img, cv.COLOR_GRAY2BGR)

cv.drawContours(cimg, contours_list,  -1, (158,255,0), 2)
print(time.time() - start)
cv.imshow('raw_img', raw_img)
cv.imshow('mask_img', mask_img)
cv.imshow('bilat_filt_img', bilat_filt_img)
cv.imshow('edges_img', edges_img)
cv.imshow('detected circles', cimg)

k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
