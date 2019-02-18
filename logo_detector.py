import cv2
import numpy as np


img = cv2.imread('./images/logo.jpeg')
img_blur = cv2.medianBlur(img, 5)
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

edge_image = cv2.Canny(img_gray, 100, 200)
range = cv2.inRange(img_blur, (182, 182, 182), (202, 202, 202))
cimg = img_blur
circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=40,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

# contours = cv2.findContours(range, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# for c in contours:
#     cnt = np.array(c[0])
#     check = cnt.copy()
#     check = check.reshape(-1)
#     check = [ch == -1 for ch in check]
#     if any(check):
#         continue
#     rect = cv2.minAreaRect(cnt)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
#
cv2.imshow('image', cimg)
cv2.waitKey(0)