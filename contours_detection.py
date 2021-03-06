import numpy as np
import cv2 as cv
import time
#TODO run against test images to make sure yellow color bounds and area min are good
#TODO some how sort the contours from findContours from big to small so we can focus on the bigger ones?
#read image
start = time.time();
raw_img = cv.imread('real_ball_light_1.jpg', -1)
hsv_img = cv.cvtColor(raw_img, cv.COLOR_BGR2HSV) #convert to hsv

#range of yellow in HSV. h: 0-180, s and v: 0-255
#the range effects the performance

#wide range
#lower_yellow = np.array([19, 100, 70])
#upper_yellow = np.array([77, 255, 255])

#original range
#lower_yellow = np.array([19, 150, 189])
#upper_yellow = np.array([30, 255, 255])

lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([45, 255, 255])

#maybe we could use a binary image instead, do a different type of blur
mask_img = cv.inRange(hsv_img, lower_yellow, upper_yellow) #black and white

#Blur image - Bilateral preserves edges but is slower than Gaussian
#Parameters are just guesses - idk what parameters to use
#bilat_filt_img = cv.bilateralFilter(mask_img, 5, 275, 275)

#b/c before we were using the edges_img, we should just use the var mask_img when we're sure this is the way to go
edges_img = mask_img
#gives a list of contours as outputs, fitler the list with particular shape

#a sorting algorithm on the above will improve the speed, merge sort
contours_list = []

most_circular_contour = None

#we will be searching for the thing closest to a circle, this is the max allowed percent error to be considered a circle
lowest_error = 0.3
#minimum area we will consider to be the ball so we ignore specks
#this is measuered as pixels, so we may need to change this depending on the image quality the rover captures
area_minimum = 100


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

<<<<<<< HEAD
        area_theoretical = 3 * int(radius) * int(radius)
        #area_theoretical = 3.14 * radius * radius
        percent_error = abs((int(area)-area_theoretical)/area_theoretical)
        
=======
        area_theoretical = 3.14 * radius * radius
        percent_error = abs((area-area_theoretical)/area_theoretical)

>>>>>>> 5252456c20876792e89ac9a22db5eca4419bdf9f
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
#cv.imshow('mask_img', mask_img)
#cv.imshow('bilat_filt_img', bilat_filt_img)
#cv.imshow('edges_img', edges_img)
cv.imshow('detected circles', cimg)

k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
