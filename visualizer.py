import numpy as np
import cv2
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, params, circle_proposals, detections):
        """
        :param params: All parameters from yaml.
        :param circle_proposals: results from circle region proposals
        :param detections: image with detections drawn on it.
        """
        self.vis_c_result = params.vis_circle_det.result
        self.vis_c_inter = params.vis_circle_det.inter
        self.circle_result = circle_proposals
        self.vis_det = params.vis_det
        self.detections = detections
        self.create_visuals()

    def create_visuals(self):
        """
        Create visuals required..
        :return:
        """
        if self.vis_c_result or self.vis_c_inter:
            result = self.circle_result['result']
            shape = result.shape
            if self.vis_c_result:
                plt.title('Circle proposals')
                plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                plt.show()
            if self.vis_c_inter:
                thresholded = self.circle_result['threshold']
                eroded = self.circle_result['erode']
                plt.title('Left: Threshold, Right: Eroded')
                to_vis = np.hstack((thresholded, eroded))
                cv2.line(to_vis, tuple(reversed(shape[:2])),
                         (shape[1], 0), (128, 128, 128), 2)
                plt.imshow(cv2.cvtColor(to_vis, cv2.COLOR_BGR2RGB))
                plt.show()

        if self.vis_det:
            plt.title('Detections')
            plt.imshow(cv2.cvtColor(self.detections, cv2.COLOR_BGR2RGB))
            plt.show()
