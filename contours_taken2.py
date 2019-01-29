import numpy as np

import cv2

from matplotlib import pyplot as plt

plt.ion()

filteredContour = []

img = cv2.imread('/mnt/c/python_code/rov_recog/yellow_circle_white_background.jpg')

grayImage = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

binaryImage = np.uint8((grayImage < 100) *1)

binaryForContour = binaryImage*1

_, contour,hierarchy=cv2.findContours(binaryForContour,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for iteration in range (0,len(contour)):

    areaOfContour = cv2.contourArea(contour[iteration])


    if areaOfContour >= 30:

        filteredContour.append(contour[iteration])

        cv2.drawContours(img,filteredContour, -1, (255,0,0), 2)

        plt.imshow(img)

cv2.imshow("hi", img)
k = cv2.waitKey(0)
if k == 27:
        cv2.destoryAllWindows()
