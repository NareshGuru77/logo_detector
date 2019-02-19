import cv2
import numpy as np
from skimage import measure, draw


img = cv2.imread('./images/bosch-logo.jpg')
org_img = img.copy()
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 2)
img_blur = cv2.medianBlur(erosion, 5)
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

cv2.imshow('erode', erosion)
th3 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
edge_image = cv2.Canny(th3, 175, 200)
cv2.imshow('edge_image', edge_image)

coords = np.column_stack(np.nonzero(edge_image))
model, inliers = measure.ransac(coords, measure.CircleModel,
                                min_samples=3, residual_threshold=1,
                                max_trials=1000)

rr, cc = draw.circle_perimeter(int(model.params[0]),
                               int(model.params[1]),
                               int(model.params[2]),
                               shape=img.shape)

zero_img = np.zeros(img.shape[:2], dtype=np.uint8)
zero_img[rr, cc] = 255

contours,hierarchy = cv2.findContours(zero_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(org_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('edge', zero_img)
cv2.imshow('result', org_img)
cv2.waitKey(0)
