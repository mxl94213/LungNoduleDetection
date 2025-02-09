"""
Trains a CNN model using tflearn wrapper for tensorflow
"""


import tflearn
import h5py
from models.cnn_model import CNNModel


# Load HDF5 dataset
h5f = h5py.File('../data/train.h5', 'r')
X_train_images = h5f['X']
Y_train_labels = h5f['Y']


h5f2 = h5py.File('../data/val.h5', 'r')
X_val_images = h5f2['X']
Y_val_labels = h5f2['Y']


## Model definition
convnet  = CNNModel()
network = convnet.define_network(X_train_images)
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='nodule-classifier.tfl.ckpt')
model.fit(X_train_images, Y_train_labels, n_epoch=50, shuffle=True, validation_set=(X_val_images, Y_val_labels),
show_metric=True, batch_size=128, snapshot_epoch=False, run_id='nodule-classifier')
model.save("nodule-classifier.tfl")
print("Network trained and saved as nodule-classifier.tfl!")

h5f.close()
h5f2.close()