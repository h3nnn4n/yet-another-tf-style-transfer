import numpy as np
import datetime
import string
import random
import time


class config:
    def __init__(self):
        self.content_weight = 1e2
        self.style_weight = 1e4
        self.tv_weight = 5e1

        self.style_layers = [
                'conv1_1',
                'conv2_1',
                'conv3_1',
                'conv4_1',
                'conv5_1']
        self.style_layer_weights = self.normalize([1, 1, 1, 1, 1])

        self.content_layers = ['conv3_2', 'conv4_2', 'conv5_2']
        self.content_layer_weights = self.normalize([0, 1, 0])

        self.optimizer_to_use = 'adam'

        self.max_iterations = 10

        self.original_colors = False
        self.color_convert_type = 'luv'

        self.content_image_name = 'images/lion.jpg'
        self.style_image_name = 'images/eternity.jpg'

        self.time_string = datetime.datetime.fromtimestamp(time.time()).strftime('%m_%d_%H_%M_%S')
        self.random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        self.output_name = 'out_%s_%s.png' % (self.time_string, self.random_string)

        self.device = '/gpu:0'

    def normalize(self, target):
        m = np.mean(target)
        return list(map(lambda x: x / m, target))
