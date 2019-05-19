from __future__ import print_function, division
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc as misc
import SimpleITK as sitk
import numpy as np
import selectivesearch
import cv2
import os
import csv
import pandas as pd
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('TQDM does make much nicer wait bars...')
    tqdm = lambda x: x

def make_mask(center,diam,z,width,height,spacing,origin):
    '''
Center : centers of circles px -- list of coordinates x,y,z
diam : diameters of circles px -- diameter
widthXheight : pixel dim of image
spacing = mm/px conversion rate np array x,y,z
origin = x,y,z mm np.array
z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height,width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center-origin)/spacing
    v_diam = int(diam/spacing[0]+5)
    v_xmin = np.max([0,int(v_center[0]-v_diam)-5])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+5])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-5])
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+5])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[int((p_y-origin[1])/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def matrix2int16(matrix):
    '''
matrix must be a numpy array NXN
Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0),dtype=np.uint16))

#####################
#
# Helper function to get rows in data frame associated
# with each file
def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)
#
# The locations of the nodes
def mask_extra():
    luna_path = "../data/"
    luna_subset_path = luna_path + "subset1/"
    output_path = "../data/masks/"
    file_list = glob(luna_subset_path + "*.mhd")
    df_node = pd.read_csv(luna_path + "annotations.csv")
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()
    print(df_node.shape)
    print(len(file_list))
    for fcount, img_file in enumerate(tqdm(file_list)):
        mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
        # if mini_df.shape[0] > 0: # some files may not have a nodule--skipping those
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())  # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())  # spacing of voxels in world coor. (mm)
        # go through all nodes (why just the biggest?)
        for node_idx, cur_row in mini_df.iterrows():
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            # just keep 3 slices
            imgs = np.ndarray([3, height, width], dtype=np.float32)
            masks = np.ndarray([3, height, width], dtype=np.uint8)
            center = np.array([node_x, node_y, node_z])  # nodule center
            v_center = np.rint((center - origin) / spacing)  # nodule center in voxel space (still x,y,z ordering)
            for i, i_z in enumerate(np.arange(int(v_center[2]) - 1,
                                              int(v_center[2]) + 2).clip(0,
                                                                         num_z - 1)):  # clip prevents going out of bounds in Z
                mask = make_mask(center, diam, i_z * spacing[2] + origin[2],
                                 width, height, spacing, origin)
                masks[i] = mask
                imgs[i] = img_array[i_z]
            np.save(os.path.join(output_path,"images_%04d_%04d.npy" % (fcount, node_idx)),imgs)
            np.save(os.path.join(output_path,"masks_%04d_%04d.npy" % (fcount, node_idx)),masks)


def lungs_ROI():
    working_path = "../data/masks/"
    file_list = glob(working_path + "images_*.npy")

    for img_file in file_list:
        # I ran into an error when using Kmean on np.float16, so I'm using np.float64 here
        imgs_to_process = np.load(img_file).astype(np.float64)
        print("on image", img_file)
        for i in range(len(imgs_to_process)):
            img = imgs_to_process[i]
            # Standardize the pixel values
            mean = np.mean(img)
            std = np.std(img)
            img = img - mean
            img = img / std
            # Find the average pixel value near the lungs
            # to renormalize washed out images
            middle = img[100:400, 100:400]
            mean = np.mean(middle)
            max = np.max(img)
            min = np.min(img)
            # To improve threshold finding, I'm moving the
            # underflow and overflow on the pixel spectrum
            img[img == max] = mean
            img[img == min] = mean
            #
            # Using Kmeans to separate foreground (radio-opaque tissue)
            # and background (radio transparent tissue ie lungs)
            # Doing this only on the center of the image to avoid
            # the non-tissue parts of the image as much as possible
            #
            kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
            centers = sorted(kmeans.cluster_centers_.flatten())
            threshold = np.mean(centers)
            thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
            #
            # I found an initial erosion helful for removing graininess from some of the regions
            # and then large dialation is used to make the lung region
            # engulf the vessels and incursions into the lung cavity by
            # radio opaque tissue
            #
            eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
            dilation = morphology.dilation(eroded, np.ones([10, 10]))
            #
            #  Label each region and obtain the region properties
            #  The background region is removed by removing regions
            #  with a bbox that is to large in either dimnsion
            #  Also, the lungs are generally far away from the top
            #  and bottom of the image, so any regions that are too
            #  close to the top and bottom are removed
            #  This does not produce a perfect segmentation of the lungs
            #  from the image, but it is surprisingly good considering its
            #  simplicity.
            #
            labels = measure.label(dilation)
            label_vals = np.unique(labels)
            regions = measure.regionprops(labels)
            good_labels = []
            for prop in regions:
                B = prop.bbox
                if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
                    good_labels.append(prop.label)
            mask = np.ndarray([512, 512], dtype=np.int8)
            mask[:] = 0
            #
            #  The mask here is the mask for the lungs--not the nodes
            #  After just the lungs are left, we do another large dilation
            #  in order to fill in and out the lung mask
            #
            for N in good_labels:
                mask = mask + np.where(labels == N, 1, 0)
            mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation
            imgs_to_process[i] = mask
        np.save(img_file.replace("images", "lungmask"), imgs_to_process)

#
#    Here we're applying the masks and cropping and resizing the image
#


    file_list = glob(working_path + "lungmask_*.npy")
    out_images = []  # final set of images
    out_nodemasks = []  # final set of nodemasks
    for fname in file_list:
        print("working on file ", fname)
        imgs_to_process = np.load(fname.replace("lungmask", "images"))
        masks = np.load(fname)
        node_masks = np.load(fname.replace("lungmask", "masks"))
        for i in range(len(imgs_to_process)):
            mask = masks[i]
            node_mask = node_masks[i]
            img = imgs_to_process[i]
            new_size = [512, 512]  # we're scaling back up to the original size of the image
            img = mask * img  # apply lung mask
            #
            # renormalizing the masked image (in the mask region)
            #
            new_mean = np.mean(img[mask > 0])
            new_std = np.std(img[mask > 0])
            #
            #  Pulling the background color up to the lower end
            #  of the pixel range for the lungs
            #
            old_min = np.min(img)  # background color
            img[img == old_min] = new_mean - 1.2 * new_std  # resetting backgound color
            img = img - new_mean
            img = img / new_std
            # make image bounding box  (min row, min col, max row, max col)
            labels = measure.label(mask)
            regions = measure.regionprops(labels)
            #
            # Finding the global min and max row over all regions
            #
            min_row = 512
            max_row = 0
            min_col = 512
            max_col = 0
            for prop in regions:
                B = prop.bbox
                if min_row > B[0]:
                    min_row = B[0]
                if min_col > B[1]:
                    min_col = B[1]
                if max_row < B[2]:
                    max_row = B[2]
                if max_col < B[3]:
                    max_col = B[3]
            width = max_col - min_col
            height = max_row - min_row
            if width > height:
                max_row = min_row + width
            else:
                max_col = min_col + height
            #
            # cropping the image down to the bounding box for all regions
            # (there's probably an skimage command that can do this in one line)
            #
            img = img[min_row:max_row, min_col:max_col]
            mask = mask[min_row:max_row, min_col:max_col]
            if max_row - min_row < 5 or max_col - min_col < 5:  # skipping all images with no god regions
                pass
            else:
                # moving range to -1 to 1 to accomodate the resize function
                mean = np.mean(img)
                img = img - mean
                min = np.min(img)
                max = np.max(img)
                img = img / (max - min)
                new_img = resize(img, [512, 512])
                new_node_mask = resize(node_mask[min_row:max_row, min_col:max_col], [512, 512])
                out_images.append(new_img)
                out_nodemasks.append(new_node_mask)


def generate_images():
    workpath = '../data/masks/'
    workpath_1 = '../data/test_images/'
    image_files = glob(workpath+'images_*.npy')
    lung_mask_files = glob(workpath+'lungmask_*.npy')
    for i_f in image_files:
        flag = False
        imgs = np.load(i_f)
        for l_m in lung_mask_files:
            if l_m[22:] == i_f[20:]:
                lungmask = np.load(l_m)
                print("image %d" % 0)
                image = misc.imsave(workpath_1+l_m[14:-4]+'.png',imgs[0] * lungmask[0])
                img = Image.open(workpath_1+l_m[14:-4]+'.png')
                print(workpath_1+l_m[14:-4]+'.png')
                fig, ax = plt.subplots(2, 2, figsize=[8, 8])
                ax[0, 0].imshow(imgs[0], cmap='gray')
                ax[0, 1].imshow(lungmask[0], cmap='gray')
                ax[1, 0].imshow(imgs[0] * lungmask[0], cmap='gray')
                ax[1, 1].imshow(np.array(img), cmap='gray')


def Selective_search(filename,file_classes):
    img = cv2.imread(filename)
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=15)
    candis = []
    classes = []
    labls = []
    coordinates = []
    for i_l in file_classes:
        sub_cor = []
        if filename[20:-4] == i_l[0]:
            sub_cor.append(eval(i_l[1]))
            sub_cor.append(eval(i_l[2]))
            coordinates.append(sub_cor)
    print(coordinates)
    for r in regions:
        sel_cor = []
        # Delete duplicated regions
        if r['rect'] in candis:
            continue
        if r['size'] < 60 or r['size'] > 1100:
            continue
        if len(r['labels']) > 1:
            continue
        x, y, w, h = r['rect']
        if w == 0 or (w != 0 and h / w > 1.8) or (h != 0 and w / h > 1.5) or h == 0:
            continue
        sel_cor.append(x)
        sel_cor.append(y)
        if sel_cor in coordinates:
            classes.append(1.0)
        else:
            classes.append(0.0)
        labls.append(r['labels'])
        candis.append((x, y, w, h))
    print(len(classes))
    return img, labls, candis, classes


def Extracte_candidate_regions():
    # loading  image

    width = 50
    height = 50

    # Read the annotation csv that contains the coordinates of the lung images for selective search
    file_class = open('../submission_files/file_classes.csv', 'r')
    csvReader = csv.reader(file_class)
    images_labels = list(csvReader)
    del images_labels[0]

    # Write the regions associated to the files and records the information
    f_label = open('../submission_files/regions_labels.csv', 'w', newline='')
    writer_label = csv.writer(f_label)
    writer_label.writerow(('File_name', 'Labels', 'Rect', 'Nodule'))



    image_path = '../data/test_images/lungmask'
    filelist = glob(image_path+'_*.png')
    filelist.sort()
    i = 0
    for filename in filelist:
        count = 0
        print("Processing the image: "+filename[20:-4])
        img, label, candidates, classes = Selective_search(filename,images_labels)
        print("The regions extracted: ",len(label))
        for rect in candidates:
            x, y, w, h = rect
            img_cut = img[y:y + h, x:x + w, :]
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

            cv2.imwrite('../images/' + filename[20:-4] + '_' + str(i) + '.png', gray)
            writer_label.writerow((filename[20:-4] + '_' + str(i) + '.png', label[count], rect, classes[count]))
            i += 1
            count += 1
    f_label.close()
    file_class.close()

def main():
    mask_extra()
    lungs_ROI()
    generate_images()
    Extracte_candidate_regions()


if __name__ == '__main__':
    main()