import yaml
import os
from collections import namedtuple


def read_circle_region_params(parameters):
    """
    :param parameters: parameters obtained from yaml.
    :return: named tuple of circle region detection parameters.
    """
    circle_tuple = namedtuple('circle_region',
                              ['ksize', 'blockSize', 'C', 'anchor',
                               'iterations', 'param1', 'param2',
                               'minRadius', 'maxRadius'])
    # assign default values..
    circle_defaults = {'ksize': 3, 'blockSize': 11, 'C': 2,
                       'anchor': [5, 5], 'iterations': 1, 'param1': 300,
                       'param2': 2, 'minRadius': 1, 'maxRadius': 100}
    circle_region = circle_tuple(**circle_defaults)

    if 'circle_region' in parameters.keys():
        # replace values read from yaml..
        circle_region = circle_region._replace(
            **parameters['circle_region'])

    return circle_region


def read_vis_circle_det(parameters):
    """
    :param parameters: parameters obtained from yaml.
    :return: named tuple of visualize circle detection parameters.
    """
    vis_tuple = namedtuple('vis_circle_det', ['result', 'inter'])
    # assign default values..
    vis_defaults = {'result': False, 'inter': False}
    vis_circle_det = vis_tuple(**vis_defaults)
    if 'vis_circle_det' in parameters.keys():
        # replace values read from yaml..
        vis_circle_det = vis_circle_det._replace(
            **parameters['vis_circle_det'])

    return vis_circle_det


class Parameters:
    """
    Class to handle all parameters in one place.
    """

    def __init__(self, yaml_path):
        """
        :param yaml_path: path to yaml file with parameters..
        """
        assert os.path.isfile(yaml_path)
        self.yaml_path = yaml_path

    def read_params(self):
        """
        Assign default values to all parameters and replace parameter values
        found in yaml.
        :return: named tuple of all parameters.
        """
        detector_params = namedtuple('params', ['image_path', 'circle_region',
                                                'vis_circle_det',
                                                'template_path', 'score_thres',
                                                'vis_det'])
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

        detector_params.template_path = parameters.get('template_path', None)
        if detector_params.template_path is None:
            raise ValueError('Please provide path to template image.')

        detector_params.vis_det = parameters.get('vis_det', True)
        detector_params.score_thres = parameters.get('score_thres', 0.3)

        return detector_params
