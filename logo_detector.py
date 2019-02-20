import cv2
import numpy as np
from parameters import Parameters
from visualizer import Visualizer
import argparse
from skimage.measure import compare_ssim


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


def match_template(roi, image, template):
    scores = []
    for x, y, side in roi:
        region_img = image[y: y+side, x: x+side, :]
        template = cv2.resize(template, tuple(reversed(
            region_img.shape[:2])))
        scores.append(compare_ssim(region_img, template,
                                   multichannel=True))

    return scores


def draw_detection(image, roi, scores, threshold):
    image = image.copy()
    for index, [x, y, side] in enumerate(roi):
        if scores[index] >= threshold:
            cv2.rectangle(image, (x, y), (x + side, y + side),
                          (0, 255, 0), 4)

    return image


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
    template = cv2.imread(params.template_path)
    roi_scores = match_template(circle_roi, img, template)
    image_det = draw_detection(img, circle_roi, roi_scores, params.score_thres)
    Visualizer(params, circle_results, image_det)
