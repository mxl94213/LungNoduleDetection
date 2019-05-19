"""
A script to predict nodules using conv net model and for analysis of results
"""

import tflearn
from models.cnn_model import CNNModel

import tensorflow as tf

import pickle
import pandas as pd 
import numpy as np 
import h5py
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools

import matplotlib.pyplot as plt

test_hdf5 = '../data/test.h5'


def load_images(filename):
    """
    Load test h5df files which contains the testing images.
    :param filename: test hdf5 files
    :return: test images and response to labels
    """
    test_h5 = h5py.File(filename, 'r')
    x_test_images = test_h5['X']
    y_test_labels = test_h5['Y']
    print(x_test_images)
    print(y_test_labels)
    return x_test_images, y_test_labels


def create_mosaic(image, nrows, ncols):
    """
    Tiles all the images in nrow * ncols.
    :param image:
    :param nrows: number of images in a row
    :param ncols: number of images in a column
    :return: formatted images
    """
    m = image.shape[1]
    n = image.shape[2]

    npad = ((0, 0), (1, 1), (1, 1))
    image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
    m += 2
    n += 2
    image = image.reshape(nrows, ncols, m, n)
    image = np.transpose(image, (0, 2, 1, 3))
    image = image.reshape(m * nrows, n * ncols)
    return image


def get_predictions(x_test_images, y_test_labels):
    """
    Given the test images, cnn network predicted the labels for images.
    :param x_test_images: test images
    :param y_test_labels: response to labels for the images
    :return:
            predictions: the predicted probability values of test labels for images
            label_predictions:  the specific class for each image
    """
    cnn_net = CNNModel()
    network = cnn_net.define_network(x_test_images)
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='nodule-classifier.tfl.ckpt')
    model.load('nodule-classifier.tfl')

    predictions = np.vstack(model.predict(x_test_images[:, :, :, :]))
    score = model.evaluate(x_test_images, y_test_labels)
    print('The total classification accuracy is : ', score)
    label_predictions = np.zeros_like(predictions)
    label_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1
    return predictions, label_predictions


def get_roc_curve(y_test_labels, predictions):
    """
    Generates the roc curve for the predicted labels and actual labels.
    :param y_test_labels: actual labels
    :param predictions: predicted labels
    :return:
            fpr: false positive rate
            tpr: true positive rate
            roc_auc: area under the curve
    """
    fpr, tpr, thresholds = roc_curve(y_test_labels[:, 1], predictions[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plot the roc curve.
    :param fpr:
    :param tpr:
    :param roc_auc:
    """
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.axis('equal')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('../visual_images/roc1.png', bbox_inches='tight')
    plt.show()


def get_matrics(y_test_labels, label_predictions):
    """
    Plot the confusion matrix.
    :param y_test_labels:
    :param label_predictions:
    :return:
        precision, recall(sensitivity), specificity, confusion matrix
    """
    confs_matrix = confusion_matrix(y_test_labels[:, 1], label_predictions[:, 1])

    TN = confs_matrix[0][0]
    FP = confs_matrix[0][1]
    FN = confs_matrix[1][0]
    TP = confs_matrix[1][1]

    precision = TP * 1.0 / (TP + FP)
    recall = TP * 1.0 / (TP + FN)
    specificity = TN * 1.0 / (TN + FP)
    return precision, recall, specificity, confs_matrix


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Purples):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


def format_image(images, num_images):
    """
    Formats the images
    :param images: 
    :param num_images: the number of images should be format
    :return: Formatted images
    """
    idxs = np.random.choice(images.shape[0], num_images)
    m = images.shape[1]
    n = images.shape[2]
    imagex = np.squeeze(images[idxs, :, :, :])
    return imagex
    

def plot_predictions(images, filename):
    imagex = format_image(images, 16)
    mosaic = create_mosaic(imagex, 2, 8)
    plt.figure(figsize=(8, 8))
    plt.imshow(mosaic, cmap='gray')
    plt.title(filename[17:])
    plt.axis('off')
    plt.savefig(filename + '.png', bbox_inches='tight')
    plt.show()
    

def main():
    x_test_images, y_test_labels = load_images(test_hdf5)
    predictions, label_predictions = get_predictions(x_test_images, y_test_labels)
    fpr, tpr, roc_auc = get_roc_curve(y_test_labels, predictions)
    plot_roc_curve(fpr, tpr, roc_auc)

    precision, recall, specificity, confs_matrix = get_matrics(y_test_labels, label_predictions)

    print(precision, recall, specificity)

    plt.figure()
    plot_confusion_matrix(confs_matrix, classes=['No-nodule', 'Nodule'], title='Confusion Matrix')
    plt.savefig('../visual_images/confusion_matrix.png', bbox_inches='tight')
    plt.show()

    # Plot all images representing TP, FP, TN, FN
    TP_images = x_test_images[(y_test_labels[:, 1] == 1) & (label_predictions[:, 1] == 1), :, :, :]
    FP_images = x_test_images[(y_test_labels[:, 1] == 0) & (label_predictions[:, 1] == 1), :, :, :]
    TN_images = x_test_images[(y_test_labels[:, 1] == 0) & (label_predictions[:, 1] == 0), :, :, :]
    FN_images = x_test_images[(y_test_labels[:, 1] == 1) & (label_predictions[:, 1] == 0), :, :, :]
    
    plot_predictions(TP_images, '../visual_images/True_Positives')
    plot_predictions(TN_images, '../visual_images/True_Negatives')
    plot_predictions(FN_images, '../visual_images/False_Negatives')
    plot_predictions(FP_images, '../visual_images/False_Positives')
  
    
if __name__ == '__main__':
    main()
