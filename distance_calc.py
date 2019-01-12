# Use cv.VideoCapture to to capture a frame by frame of the of the video file
# Extract the frame size (pixels) so you can calculate distance from it
# Access pixels and change colors looking for yellow - HSL, HSV, Gray Scale
# HSV of Yellow = [90, 255, 255]
# Use object tracking/changing color space to convert anything not yellow to black and yellow to white
# Hough transform over white circle
# Do a math operation from each camera to find distance
# Overlay images and take the average - see how far it is from the center to determine orientation
