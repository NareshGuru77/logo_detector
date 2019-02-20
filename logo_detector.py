import cv2
import numpy as np


def get_circle_regions(image):
    image = image.copy()
    img_blur = cv2.medianBlur(image, 3)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)

    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(thresholded, el, iterations=1)
    circles = cv2.HoughCircles(eroded, cv2.HOUGH_GRADIENT,
                               1, img.shape[0] / 8, param1=300,
                               param2=30, minRadius=1,
                               maxRadius=100)

    roi = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(image, center, radius,
                       (255, 0, 255), 3)
            x, y = i[0] - i[2], i[1] - i[2]
            side = 2 * i[2]
            roi.append([x, y, side])

    return roi, image


def get_h_regions(image):
    image = image.copy()
    img_blur = cv2.medianBlur(image, 3)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    # eroded = cv2.erode(thresholded, kernel, iterations=2)
    dilated = cv2.dilate(thresholded, kernel, iterations=2)
    edged = cv2.Canny(thresholded, 30, 200)
    contours = cv2.findContours(thresholded.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)[0]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.1 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            cv2.drawContours(image, c, -1, (0, 255, 0), 3)
    cv2.imshow('thresholded', thresholded)
    cv2.imshow('eroded', dilated)
    cv2.imshow('edges', edged)
    cv2.imshow('image', image)
    cv2.waitKey(0)

    roi = []

    return roi, image

img = cv2.imread('./images/bosch-logo.jpg')
circle_roi, _ = get_circle_regions(img)
h_roi, _ = get_h_regions(img)
