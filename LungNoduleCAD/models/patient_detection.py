import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
from skimage import io
from glob import glob
import cv2
import csv
import tflearn
import numpy as np
from scipy.ndimage import imread
from models.cnn_model import CNNModel
import shutil
import os


def SeletiveSearch(image):
    # loading  image


    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=15)
    print(len(regions))

    print(regions)
    candidates = []
    label = []
    for r in regions:
        # abandon duplicated regions
        if r['rect'] in candidates:
            continue
        if r['size'] < 40 or r['size'] > 2000:
            continue
        #if len(r['labels']) > 3:
        #    continue
        x, y, w, h = r['rect']
        if len(r['labels']) > 2:
            continue
        if w == 0 or (w != 0 and h / w > 1.8) or (h != 0 and w / h > 1.5) or h == 0:
           continue
        label.append(r['labels'])
        candidates.append((x, y, w, h))
    print(len(label))
    print(len(candidates))
    print(label)


    length = len(candidates)
    width = 50
    height = 50
    img_sample = np.zeros((length, width * height))
    text = '../testing_patient/predicted_nodules.csv'
    f = open(text, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(('File_name', 'Labels', 'Rect'))
    
    i = 0
    for rect in candidates:
        x, y, w, h = rect
        img_cut = image[y:y + h, x:x + w, :]
        if w > h:
            real_size = w
        else:
            real_size = h
        top_padding = int((real_size - h) / 2)
        left_padding = int((real_size - w) / 2)
        img_cut = cv2.copyMakeBorder(img_cut, top_padding, top_padding, left_padding, left_padding,
                                         borderType=cv2.BORDER_REPLICATE)
        img_resize = cv2.resize(img_cut, (width, height), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('../temptation/img_' + str(i) + '.png', gray)
        writer.writerow(('img_'+ str(i) + '.png', label[i], rect))
        i += 1

    f.close()


def main():
    height = 50
    width = 50
    workpath = '../testing_patient/'
    file_name = '../testing_patient/lungmask_0003_0125'
    image = cv2.imread(file_name + '.png')
    lung_mask = np.load(workpath+file_name+'.npy')
    image_files = np.load(workpath+'images_'+file_name[28:]+'.npy')



    SeletiveSearch(image)
    images_path = '../temptation/'
    filelist = glob(images_path + '*.png')

    filelist.sort(key=lambda x:int(x[18:-4]))
    print(filelist)

    image_num = len(filelist)
    data = np.zeros((image_num, height, width, 1))
    for idx in range(image_num):
        img = imread(filelist[idx]).astype('float32')
        img = img.reshape(-1, img.shape[0], img.shape[1], 1)
        data[idx, :, :, :] = img
    print(data.shape)

    # Read images and corresponding labels
    csvfile = open('../testing_patient/predicted_nodules.csv', 'r')
    csvReader = csv.reader(csvfile)
    images_labels = list(csvReader)
    csvfile_1 = open('../testing_patient/file_classes.csv', 'r')
    csvReader_1 = csv.reader(csvfile_1)
    images_nodules = list(csvReader_1)
    real_candidates = []
    candidates = []
    del images_labels[0]
    del images_nodules[0]
    for i_l in images_labels:
        i_l[1] = eval(i_l[1])
        i_l[2] = eval(i_l[2])
        candidates.append(i_l[2])
    print(images_labels)
    print(candidates)

    # Get the lung nodule coordinates
    for j_l in images_nodules:
        if j_l[0] == file_name[19:]:
            j_l[1] = eval(j_l[1])
            j_l[2] = eval(j_l[2])
            real_candidates.append((j_l[1], j_l[2]))

    # Mapping the real regions that contain lung nodules
    real_nodules = []
    for candidate in candidates:
        if (candidate[0], candidate[1]) in real_candidates:
            real_nodules.append(candidate)
    print(real_nodules)

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
    print(len(index_list))
    print(index_list)

    nodule_candidate = []
    for i in index_list:
        nodule_candidate.append(candidates[i])
    print(nodule_candidate)
    fig, ax = plt.subplots(2, 2, figsize=[8, 8])
    ax[0, 0].imshow(image_files[0], cmap='gray')
    ax[0, 1].imshow(image_files[0] * lung_mask[0], cmap='gray')
    ax[1, 0].imshow(image)
    ax[1, 1].imshow(image)
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax[1, 0].add_patch(rect)
    for x, y, w, h in real_nodules:
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='yellow', linewidth=1)
        ax[1, 1].add_patch(rect)
    for x, y, w, h in nodule_candidate:
        rect = mpatches.Rectangle((x-3, y-3), w+5, h+5, fill=False, edgecolor='red', linewidth=1)
        ax[1, 1].add_patch(rect)
    plt.show()

    shutil.rmtree('../temptation')
    os.mkdir('../temptation')


if __name__ == '__main__':
    main()