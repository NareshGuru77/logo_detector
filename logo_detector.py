import cv2
import numpy as np
from parameters import Parameters
from visualizer import Visualizer
import argparse


def get_circle_regions(image, params):
    image = image.copy()
    img_blur = cv2.medianBlur(image, params.ksize)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, params.blockSize, params.C)

    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(params.anchor))
    eroded = cv2.erode(thresholded, el, iterations=params.iterations)
    circles = cv2.HoughCircles(eroded, cv2.HOUGH_GRADIENT,
                               1, img.shape[0] / 8, param1=params.param1,
                               param2=params.param2, minRadius=params.minRadius,
                               maxRadius=params.maxRadius)

    roi = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv2.circle(image, center, radius,
                       (255, 0, 255), 3)
            x = i[0] - i[2] if i[0] > i[2] else 0
            y = i[1] - i[2] if i[1] > i[2] else 0
            side = 2 * i[2]
            roi.append([x, y, side])

    results = {'result': image, 'threshold': thresholded, 'erode': eroded}
    return roi, results


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
    # cv2.imshow('thresholded', thresholded)
    # cv2.imshow('eroded', dilated)
    # cv2.imshow('edges', edged)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    roi = []

    return roi, image


def get_lines(image):
    image = image.copy()
    image_blur = cv2.GaussianBlur(image, (3,3), 0, 0, cv2.BORDER_DEFAULT)
    img_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)

    # el_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    # el_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    # dilated_x = cv2.erode(sobelx, el_v, iterations=1)
    # dilated_y = cv2.erode(sobely, el_h, iterations=1)

    gradient = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    thresholded = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)[1]
    edge = cv2.Canny(thresholded, 50, 150)

    lines = cv2.HoughLinesP(edge, 1, np.pi/90, 50, 10, 10)
    lines = np.array(lines)
    lines = lines.reshape((-1, 4))
    for x1, y1, x2, y2 in lines:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # lines = np.array(cv2.HoughLines(edge, 1, np.pi/180, 100))
    # lines = lines.reshape((-1, 2))
    # for rho, theta in lines:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cv2.imshow('thresholded', thresholded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Get yml path.')
    parser.add_argument('--yaml_path', default=None, type=str,
                        required=False,
                        help='Path to yaml.')
    args = parser.parse_args()
    parameters = Parameters(args.yaml_path)
    params = parameters.read_params()
    img = cv2.imread(params.image_path)
    circle_roi, circle_results = get_circle_regions(img, params.circle_region)
    Visualizer(params, circle_results, img)
    h_roi, _ = get_h_regions(img)
    get_lines(img)
