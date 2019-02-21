import argparse

import cv2
import numpy as np
from skimage.measure import compare_ssim

from parameters import Parameters
from visualizer import Visualizer


def get_circle_regions(image, params):
    """
    :param image: The image on which circles are to be detected.
    :param params: Named tuple with all parameters for circle detection.
    :return:A dictionary with final result and intermediate results.
    """
    image = image.copy()
    img_blur = cv2.medianBlur(image, params.ksize)
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    thresholded = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, params.blockSize, params.C)

    # The threshold image is eroded with an eliptical kernel.
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, tuple(params.anchor))
    eroded = cv2.erode(thresholded, el, iterations=params.iterations)
    circles = cv2.HoughCircles(eroded, cv2.HOUGH_GRADIENT,
                               1, img.shape[0] / 8, param1=params.param1,
                               param2=params.param2, minRadius=params.minRadius,
                               maxRadius=params.maxRadius)

    # collecting roi regions and drawing proposed circles.
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
    """
    :param roi: List of [x, y, side], where x, y location of top corner,
                side is the side length of the square region.
    :param image: The image on which template should be matched.
    :param template: Template image.
    :return: scores list with a score ranging from 0 to 1 for each region.
            A score of 1 indicates region and template are identical.
    """
    scores = []
    for x, y, side in roi:
        # copy to get actual region and template in every iteration.
        image_cpy = image.copy()
        template_cpy = template.copy()
        region_img = image_cpy[y: y+side, x: x+side, :]
        region_img = cv2.medianBlur(region_img, 5)
        region_img = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        template_cpy = cv2.resize(template_cpy, tuple(reversed(
            region_img.shape[:2])))
        template_cpy = cv2.medianBlur(template_cpy, 5)
        template_cpy = cv2.cvtColor(template_cpy, cv2.COLOR_BGR2GRAY)
        # ssim scores of each proposed region is collected.
        scores.append(compare_ssim(region_img, template_cpy))

    return scores


def draw_detection(image, roi, scores, threshold):
    """
    :param image: Original image on which detections are to be drawn.
    :param roi: List of [x,y,side] indicating the region proposals.
    :param scores: List of score value of each region proposal.
    :param threshold: regions with score above this threshold are
            considered as detections.
    :return: Returns the image with drawn detected rectangles and
            written detection scores.
    """
    image = image.copy()
    for index, [x, y, side] in enumerate(roi):
        score = round(scores[index], 2)
        if score >= threshold:
            cv2.rectangle(image, (x, y), (x + side, y + side),
                          (0, 255, 0), 4)
            cv2.putText(image, str(score), (x, y + 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.5, (0, 0, 128), 1, cv2.LINE_AA)

    return image


if __name__ == '__main__':
    # parse yaml path..
    parser = argparse.ArgumentParser(
        description='Get yml path.')
    parser.add_argument('--yaml_path', default='./parameters.yml', type=str,
                        required=False,
                        help='Path to yaml with parameters.')
    args = parser.parse_args()
    # read all parameters..
    parameters = Parameters(args.yaml_path)
    params = parameters.read_params()
    # read the image..
    img = cv2.imread(params.image_path)
    # obtain region proposals..
    circle_roi, circle_results = get_circle_regions(img, params.circle_region)
    # read template..
    template_image = cv2.imread(params.template_path)
    # calculate region scores..
    roi_scores = match_template(circle_roi, img, template_image)
    # draw detections..
    image_det = draw_detection(img, circle_roi, roi_scores, params.score_thres)
    # visualize results..
    Visualizer(params, circle_results, image_det)
