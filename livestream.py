import cv2 as cv

windowName = "Live Video Feed"
cv.namedWindow(windowName)
cap = cv.VideoCapture(-1)

if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False

while ret:
    ret, fram = cap.read()
    #output = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow(windowName, frame)
    if cv.waitKey(1) == 27:
        break
cap.release()
cv.destroyWindow(windowName)
