"""
A conv net model using tflearn wrapper for tensorflow
"""

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.normalization import local_response_normalization


class CNNModel(object):

    def __init__(self,network=None):
        self.network = network
        self.model = None

    def preprocessing(self):
        img_prep = ImagePreprocessing()
        img_prep.add_featurewise_zero_center()
        img_prep.add_featurewise_stdnorm()
        return img_prep

    def augmentation(self):
        img_aug = ImageAugmentation()
        img_aug.add_random_flip_leftright()
        img_aug.add_random_rotation(max_angle=90.)
        img_aug.add_random_blur(sigma_max=3.)
        return img_aug

    def input_layer(self, X_images, name):
        img_prep = self.preprocessing()
        img_aug = self.augmentation()
        self.network = input_data(shape=[None, X_images.shape[1], X_images.shape[2], X_images.shape[3]],
                                  data_augmentation=img_aug, data_preprocessing=img_prep, name=name)
        return self.network

    def convolution_layer(self, num_filters, filter_size, name, activation_type='relu', regularizer=None):
        self.network = conv_2d(self.network, num_filters, filter_size, activation=activation_type, regularizer=regularizer, name=name)
        return self.network

    def max_pooling_layer(self, kernel_size, name):
        self.network = max_pool_2d(self.network, kernel_size, name=name)
        return self.network

    def fully_connected_layer(self, num_units, activation_type, name):
        self.network = fully_connected(self.network, num_units, activation=activation_type, name=name)
        return self.network

    def dropout_layer(self, name, prob=0.5):
        if(prob > 1) or (prob < 0):
            raise ValueError('Probability must be the range from 0 to 1')
        self.network = dropout(self.network, prob, name=name)
        return self.network

    def define_network(self, x_images, mode='testtrain'):

        inp_layer = self.input_layer(x_images, name='input_1')
        conv_layer_1 = self.convolution_layer(32, 5, 'conv1', 'relu', 'L2')
        mp_layer_1 = self.max_pooling_layer(2, 'max_pool_1')
        conv_layer_2 = self.convolution_layer(64, 5, 'conv2', 'relu', 'L2')
        mp_layer_2 = self.max_pooling_layer(2, 'max_pool_2')
        conv_layer_3 = self.convolution_layer(64, 3, 'conv3', 'relu', 'L2')
        mp_layer_3 = self.max_pooling_layer(2, 'max_pool_3')

        fully_connected_layer_1 = self.fully_connected_layer(1024, 'relu', 'fully_connected_1')
        fully_connected_layer_2 = self.fully_connected_layer(512, 'relu', 'fully_connected_2')
        dropout_layer_1 = self.dropout_layer('dropout_l1', 0.5)
        softmax_layer = self.fully_connected_layer(2, 'softmax', 'soft_max_l')

        self.network = regression(self.network,optimizer='adam', loss = 'categorical_crossentropy', learning_rate= 0.001)

        if mode == 'testtrain':
            return self.network
        if mode == 'visual':
            return conv_layer_1, conv_layer_2, conv_layer_3, self.network