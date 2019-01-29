#For mult_colored_balls.jpg:
#lower_yellow = np.array([19, 150, 189])
#upper_yellow = np.array([30, 255, 255])
#For solid_yellow_circle:
#lower_yellow = np.array([20, 250, 255])
#upper_yellow = np.array([40, 255, 255])

import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt

def main():
    #read image
    raw_img = cv.imread('/mnt/c/python_code/rov_recog/solid_yellow_circle.png', -1)
    #Convert all yellow to white, everything black
    #I have no idea what "_," I saw it online and it fixed my tuple error
    #Still can't recognize circles
    thresh = convert_image(raw_img, [20, 250, 255], [40, 255, 255]) # _, thresh =
    #gives a list of contours as outputs, filter the list with particular shape
    _, contour, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                            cv.CHAIN_APPROX_SIMPLE)

    contours_list = find_contours(contour)

    cv.drawContours(raw_img, contours_list,  -1, (158,255,0), 2)
    cv.drawContours(thresh, contours_list, -1, (158,255,0), 2)

    print("contour length:", len(contour))
    print(contours_list)
    #print(contour)
    cv.imshow('raw_img', raw_img)
    cv.imshow('thres', thresh)
    k = cv.waitKey(0)
    if k == 27:
        cv.destroyAllWindows()

def convert_image(img, lower, upper):
    #hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    #range of colors to make white, rest is black
    #lower_color = np.array(lower)
    #upper_color = np.array(upper)

    #mask_img = cv.inRange(hsv_img, lower_color, upper_color) # B&W
    blur_img = cv.GaussianBlur(img, (5, 5), 0)
    return cv.Canny(blur_img, 75,200)
    #return cv.threshold(blur_img, 60, 255, cv.THRESH_BINARY)

def find_contours(contour):
    contours_list = []
    for i in range (0, len(contour)):
        #perim, approx, and area will be used to see if it's a circle and how big of a circle to expect
        perim = cv.arcLength(contour[i], True)
        approx = cv.approxPolyDP(contour[i], 0.04*perim, True)
        area = cv.contourArea(contour[i])
        print("looped")
        # approximate the pixel area
        if (len(approx) > 8) and (len(approx) < 23) and (area > 40):# and (area > math.pi*150^2) and (area < math.pi*250^2):
            contours_list.append(contour[i])
    return contours_list

if __name__ == "__main__":
    main()
