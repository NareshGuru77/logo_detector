import cv2
import numpy as np


img = cv2.imread('./images/bosch-logo.jpg')
mask = cv2.inRange(img, (182,182,182), (202,202,202))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.dilate(img_gray, kernel, iterations = 1)
edge_image = cv2.Canny(img, 100, 200)
# contours = cv2.findContours(img_gray, )

cv2.imshow('image', mask)
cv2.waitKey(0)