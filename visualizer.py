import numpy as np
import cv2
import matplotlib.pyplot as plt


def write_text(image, text, pos):
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                1, (128, 128, 128), 2, cv2.LINE_AA)


def draw_separator(to_vis, loc1, loc2):
    cv2.line(to_vis, loc1, loc2, (128, 128, 128), 2)


class Visualizer:

    def __init__(self, params, circle_results, image):
        self.vis_c_result = params.vis_circle_det.result
        self.vis_c_inter = params.vis_circle_det.inter
        self.circle_result = circle_results
        self.image = image.copy()
        self.create_visuals()

    def create_visuals(self):
        if self.vis_c_result or self.vis_c_inter:
            result = self.circle_result['result']
            shape = result.shape
            text_position = ((result.shape[0] // 2) - 40, 40)
            write_text(result, 'Result', text_position)
            write_text(self.image, 'Image', text_position)
            if self.vis_c_result:
                to_vis = np.hstack((self.image, result))
                draw_separator(to_vis, shape[:2], (shape[0], 0))
                plt.imshow(cv2.cvtColor(to_vis, cv2.COLOR_BGR2RGB))
                plt.show()
            if self.vis_c_inter:
                thresholded = self.circle_result['threshold']
                eroded = self.circle_result['erode']
                write_text(thresholded, 'Threshold', text_position)
                write_text(eroded, 'Erosion', text_position)
                to_vis = np.hstack((thresholded, eroded))
                draw_separator(to_vis, shape[:2], (shape[0], 0))
                plt.imshow(cv2.cvtColor(to_vis, cv2.COLOR_BGR2RGB))
                plt.show()
