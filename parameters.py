import yaml
import os
from collections import namedtuple


def read_circle_region_params(parameters):
    circle_tuple = namedtuple('circle_region',
                              ['ksize', 'blockSize', 'C', 'anchor',
                               'iterations', 'param1', 'param2',
                               'minRadius', 'maxRadius'])
    circle_defaults = {'ksize': 3, 'blockSize': 11, 'C': 2,
                       'anchor': [5, 5], 'iterations': 1, 'param1': 300,
                       'param2': 2, 'minRadius': 1, 'maxRadius': 100}
    circle_region = circle_tuple(**circle_defaults)

    if 'circle_region' in parameters.keys():
        circle_region = circle_region._replace(
            **parameters['circle_region'])

    return circle_region


def read_vis_circle_det(parameters):
    vis_tuple = namedtuple('vis_circle_det', ['result', 'all'])
    vis_defaults = {'result': False, 'all': False}
    vis_circle_det = vis_tuple(**vis_defaults)
    if 'vis_circle_det' in parameters.keys():
        vis_circle_det = vis_circle_det._replace(
            **parameters['vis_circle_det'])

    return vis_circle_det


class Parameters:

    def __init__(self, yaml_path):
        assert os.path.isfile(yaml_path)
        self.yaml_path = yaml_path

    def read_params(self):
        detector_params = namedtuple('params', ['image_path', 'circle_region',
                                                'vis_circle_det'])
        with open(self.yaml_path, 'r') as stream:
            try:
                parameters = yaml.load(stream)
            except:
                raise FileNotFoundError

        detector_params.image_path = parameters.get('image_path', None)
        if detector_params.image_path is None:
            raise ValueError('Please provide path to image.')

        detector_params.circle_region = read_circle_region_params(parameters)
        detector_params.vis_circle_det = read_vis_circle_det(parameters)

        return detector_params