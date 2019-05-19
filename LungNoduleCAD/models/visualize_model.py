"""
A script to visualize layers and filters in the conv net model 
"""

import tflearn
from models.cnn_model import CNNModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread, zoom
from skimage import data


def get_layer_output(layer, model, inp):
    """
    Returns the model layer output.
    :param layer: cnn layer
    :param model: cnn model
    :param inp: input image
    """
    m2 = tflearn.DNN(layer, session=model.session)
    yhat = m2.predict(inp.reshape(-1, inp.shape[0], inp.shape[1], 1))
    yhat_1 = np.array(yhat[0])
    return m2, yhat_1


def create_mosaic(image, nrows, ncols):
    """
    Tiles all the layers in nrow * ncol
    :param image:
    :param nrows:
    :param ncols:
    :return: formatted image
    """
    m = image.shape[0]
    n = image.shape[1]
    npad = ((1, 1), (1, 1), (0, 0))
    image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
    m += 2
    n += 2
    image = image.reshape(m, n, nrows, ncols)
    image = np.transpose(image, (2, 0, 3, 1))
    image = image.reshape(m*nrows, n*ncols)
    return image

def plot_layers(image, idx, pltfilename, size=12, campx='gray'):
    """
    Plots filters in the layers
    :param image: layer output of form M x N x nfilt
    :param idx: layer number
    :param pltfilename: output saving filename
    """
    nfilt = image.shape[-1]

    mosaic = create_mosaic(image, int(nfilt / 4), 4)
    plt.figure(figsize=(size, size))
    plt.imshow(mosaic, cmap=campx)
    plt.axis('off')
    plt.savefig(pltfilename+str(idx)+'.png', bbox_inches='tight')
    plt.show()


def get_weights(m2, layer):
    """
    Gets a layer's weights
    :param m2:
    :param layer:
    :return:
    """
    weights = m2.get_weights(layer.W)
    print(weights.shape)
    weights = weights.reshape(weights.shape[0], weights.shape[1], weights.shape[-1])
    return weights


def plot_single_output(image, size=6):

    nfilt = image.shape[-1]

    mosaic = create_mosaic(image, int(nfilt / 4), 4)
    plt.figure(figsize=(size, size))
    plt.imshow(mosaic, cmap='gray')
    plt.axis('off')
    plt.savefig('filterout'+'.png', bbox_inches='tight')
    plt.show()

def main():

    filename = '../data/train/image_2473853.jpg'
    inp = imread(filename).astype('float32')
    plt.imshow(inp, cmap='gray')
    plt.show()
    print(inp)
    print(len(inp.reshape(-1, inp.shape[0], inp.shape[1], 1)))
    cnnnet = CNNModel()
    conv_layer1, conv_layer2, conv_layer3, network = cnnnet.define_network(inp.reshape(-1, inp.shape[0], inp.shape[1], 1), 'visual')
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='nodule-classifier.tfl.ckpt')
    model.load('nodule-classifier.tfl')

    labels = model.predict(inp.reshape(-1, 50, 50, 1))
    print("This is the labels :", labels)
    if labels[0][1] == 1:
        print('The input image is a nodule.')
    else:
        print('The input image is not a nodule.')

    layer_to_be_plotted = [conv_layer1, conv_layer2, conv_layer3]
    for idx, layer in enumerate(layer_to_be_plotted):
        m2, yhat = get_layer_output(layer, model, inp)
        plot_layers(yhat, idx, '../visual_images/conv_layer_')

    weights = get_weights(m2, conv_layer1)
    plot_layers(weights, 0, '../visual_images/weight_conv_layer_', 6)
    plt.show()


if __name__ == "__main__":
    main()
