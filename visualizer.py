import numpy as np
import cv2
import matplotlib.pyplot as plt


class Visualizer:

    def __init__(self, params, circle_results):
        self.circle_result = True
        self.circle_all = True
        self.circle_result = circle_results
        self.create_visuals()

    def create_visuals(self):
        if self.circle_result:
            plt.imshow(cv2.cvtColor(self.circle_result['result'],
                                    cv2.COLOR_BGR2RGB))

        plt.show()
