import numpy as np
import cv2
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, params, circle_results):
        self.vis_c_result = params.vis_circle_det.result
        self.vis_c_all = params.vis_circle_det.all
        self.circle_result = circle_results
        self.create_visuals()

    def create_visuals(self):
        if self.vis_c_result:
            plt.imshow(cv2.cvtColor(self.circle_result['result'],
                                    cv2.COLOR_BGR2RGB))

        if self.vis_c_all:
            pass

        plt.show()
