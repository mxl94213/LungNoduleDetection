"""
A script to visualize results and writing the results into submission files
"""

import tflearn
from models.cnn_model import CNNModel
from models.performance_evaluate import Performance_evaluate
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import imread, zoom
from glob import glob
from sklearn.metrics import roc_curve, auc, confusion_matrix
import csv

def main():

    height = 50
    width = 50

    # To get the images from the folder and to sort them as the sequence as csv files
    images_path = '../images/'
    filelist = glob(images_path+'*.png')
    print(filelist)
    filelist.sort(key=lambda x:int(x[29:-4]))

    # Read images and corresponding labels
    csvfile = open('../submission_files/regions_labels.csv', 'r')
    csvReader = csv.reader(csvfile)

    # images_labels contains the name of images and the labels generated from selective search
    images_labels = list(csvReader)
    del images_labels[0]
    for i_l in images_labels:
        i_l[1] = eval(i_l[1])
        i_l[2] = eval(i_l[2])
    print(len(images_labels))

    # Pre-processing the data to generate all dataset.
    image_num = len(filelist)
    data = np.zeros((image_num, height, width, 1))
    for idx in range(image_num):
        img = imread(filelist[idx]).astype('float32')
        img = img.reshape(-1, img.shape[0], img.shape[1], 1)
        data[idx, :, :, :] = img
    print(data.shape)

    # Feed the data into trained model.
    cnnnet = CNNModel()
    network = cnnnet.define_network(data, 'testtrain')
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='nodule-classifier.tfl.ckpt')
    model.load('nodule-classifier.tfl')
    predictions = np.vstack(model.predict(data[:, :, :, :]))





    label_predictions = np.zeros_like(predictions)
    label_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1
    print(len(label_predictions))
    index_list = []
    for ind, val in enumerate(label_predictions):
        if val[1] == 1:
            index_list.append(ind)
    images_name = []
    for index in index_list:
        #print(images_labels[index][0][:18])
        images_name.append(images_labels[index][0][:18]+'.png')
    images_name2 = list(set(images_name))
    images_name2.sort(key=images_name.index)
    print(images_name2)
    print(len(images_name2))
    img_list = glob('../data/test_images/*.png')
    img_list.sort()
    allimages = []
    for l in img_list:
        allimages.append(l[20:])
    print(allimages)
    # Writing the results into the files to show whether the patient contains nodules
    f_submission1 = open('../submission_files/predicted_patient_labels.csv','w')
    writer_1 = csv.writer(f_submission1)
    writer_1.writerow(('Patient','Predict_Labels'))
    for a_img in allimages:
        if a_img in images_name2:
            writer_1.writerow((a_img, 1))
        else:
            writer_1.writerow((a_img, 0))

    # Writing the results into the submission csv file
    f_submission = open('../submission_files/predicted_regions_labels.csv','w')
    writer = csv.writer(f_submission)
    writer.writerow(('File_name', 'Labels', 'Rect', 'Actual_Nodule', 'Predict_Nodule'))
    for i in range(len(images_labels)):
        #index = index_list[i]
        #print(images_labels[index][0][:])
        writer.writerow((images_labels[i][0][:], images_labels[i][1], images_labels[i][2], images_labels[i][3],
                         label_predictions[i][1]))


    actual = []
    pre = []
    for i in range(len(images_labels)):
            actual.append(eval(images_labels[i][3]))
            pre.append(predictions[i][1])
    print(actual)
    print(pre)
    fpr, tpr, thresholds = roc_curve(actual, pre, pos_label=1)
    roc_auc = auc(fpr, tpr)

    print("The false positive rate is: ", fpr)
    print("The true positive rate is:", tpr)
    print(thresholds)
    print("The auc is:", roc_auc)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.axis('equal')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()



    csvfile.close()
    f_submission.close()
    f_submission1.close()

    # Performance_evaluate()


if __name__ == "__main__":
    main()
