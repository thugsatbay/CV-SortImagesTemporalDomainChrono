
# coding: utf-8

# In[60]:

import matplotlib.pyplot as plt
import os
import glob
import h5py
import numpy as np

import scipy.io as sio 
import cv2
import pickle
import time
import random
import sys
from collections import namedtuple as nt


def sample_result_image_name(count):
    return str(("%04d") % count) + '.jpg'
# In[61]:


os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[62]:


from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
from keras.utils import data_utils, plot_model
from keras import backend as K
from keras.models import model_from_json
from keras.callbacks import TensorBoard
K.clear_session()


# In[63]:


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))



TYNAMO_HOME_DIR = '/nfs/tynamo/home/data/vision7/gdhody/chrono/'
BLITZLE_HOME_DIR = '/nfs/blitzle/home/data/vision5/gdhody/chrono/'
HOME_DIR = TYNAMO_HOME_DIR


# In[65]:



EPOCH = str(80)
BATCH_SIZE = 128
LR, LR_DECAY = 0.01, 0.0000625
LOSS = 'mean_squared_error'
MODEL_NAME = 'URLSS'
MODEL_PATH = os.path.join(HOME_DIR, 'models', MODEL_NAME, 'model.json')
MODEL_WEIGHTS = os.path.join(HOME_DIR, 'models', MODEL_NAME, 'weights', 'EPOCH_' + EPOCH + '.h5')
HDF5_PATH = os.path.join(HOME_DIR, 'patch.hdf5')
PICKLE_PATH = os.path.join(HOME_DIR, 'patch.pkl')


# In[66]:


# %run model_and_callback.ipynb
# %run patch_DataLoad_Binary.ipynb
from compare_model_and_callback import lstm_model, EpochCallback
from patch_DataLoad_Binary_test import ChronoDataSet


# In[67]:


model_input, model_output = lstm_model(
                                        input_shape=(4, 80, 80, 3),
                                        resnet=False,
                                        dense_feature_to_compare=0,
                                        drop_feature=0,
                                        compare_layer=True,
                                        compare_layer_dense=512,
                                        drop_compare_dense=0,
                                        lstm_cells=[(0, True, 'sum')],
                                        many_to_many=False,
                                        dense_layer_after_lstm=0,
                                        multi_output=False,
                                        number_of_classes=1,
                                     )
model = Model(inputs=[model_input], outputs=[model_output])

# loaded_model_json = None
# with open(MODEL_PATH, 'r') as f:
#     loaded_model_json = f.read()
# model = model_from_json(loaded_model_json)
# # load weights into new model
output_layer_name = model.layers[-1].name
print model.summary()



sgd = SGD(lr=LR, decay=LR_DECAY, momentum=0.9, nesterov=True)
model.compile(
                loss={output_layer_name : LOSS},
                optimizer=sgd,
                metrics=['accuracy']
             )


# In[70]:


model.load_weights(MODEL_WEIGHTS, by_name=True)
print "Weights from", str(EPOCH), "epoch Loaded", MODEL_WEIGHTS


# In[ ]:


print "TESTING"
print model.metrics_names
test_generator = ChronoDataSet(
                                        PICKLE_PATH,
                                        HDF5_PATH,
                                        batch_size=BATCH_SIZE,
                                        start_epoch=0,
                                        mode='TEST',
                                        dataset='UCF'
                                )
print "Test Data Generator Ready"
print model.evaluate_generator(
                            generator=test_generator,
                            steps=test_generator.getBatchRunsPerEpoch(),
                            max_queue_size=9,
                            workers=3,
                            use_multiprocessing=True
)


# In[74]:

# '''
# Output File Code
# '''
# TRAIN_UCF, TRAIN_HMDB = None, None
# with open(PICKLE_PATH, 'rb') as f:
#     TRAIN_UCF, TRAIN_HMDB = pickle.load(f)
# print "Dataset Opened"
# train_samples_ucf, validation_samples_ucf = (70 * len(TRAIN_UCF)) / 100, len(TRAIN_UCF) / 10
# test_samples_ucf = len(TRAIN_UCF) - train_samples_ucf - validation_samples_ucf
# TEST_DATA = TRAIN_UCF[train_samples_ucf + validation_samples_ucf:]
# test_generator = ChronoDataSet(
#                                         PICKLE_PATH,
#                                         HDF5_PATH,
#                                         batch_size=BATCH_SIZE,
#                                         start_epoch=0,
#                                         mode='TEST',
#                                         dataset='UCF'
#                                 )
# ITERAATIONS = test_generator.getBatchRunsPerEpoch()
# TRUE, FALSE = [], []
# for idx in xrange(ITERAATIONS):
#     image_batch = test_generator.__getitem__(idx)
#     true_false = np.squeeze(model.predict_on_batch(image_batch[0])) >= 0.5
#     assert len(true_false) == len(TEST_DATA[idx * BATCH_SIZE:(idx+1) * BATCH_SIZE])
#     for value, data in zip(true_false, TEST_DATA[idx * BATCH_SIZE:(idx+1) * BATCH_SIZE]):
#         if value:
#             TRUE.append(data)
#             continue
#         FALSE.append(data)
# #     if len(TRUE) == 10:
# #         break

# POS_IMG_SAMPLES = os.path.join(HOME_DIR, 'image_results', 'pos')
# NEG_IMG_SAMPLES = os.path.join(HOME_DIR, 'image_results', 'neg')
# if not os.path.exists(POS_IMG_SAMPLES):
#     os.makedirs(POS_IMG_SAMPLES)
# if not os.path.exists(NEG_IMG_SAMPLES):
#     os.makedirs(NEG_IMG_SAMPLES)
    
# print "Finished finding results", len(TRUE), len(FALSE)

# file_no_per_class = {}
# correct_per_class = {}
# for index, pos_sample in enumerate(TRUE):
#     if index and not index % 500:
#         print "Correct processed", index, '|', len(TRUE)
#     im_file_name, crop, frames = pos_sample
#     image_class = im_file_name.split('/')[-1].split('_')[1].upper()
#     if np.random.randint(5) - 4 == 0:
#         im_file_paths = [os.path.join(HOME_DIR, im_file_name, 'Image' + str(frame) + '.jpg') for frame in frames]      
#         ims = [cv2.imread(im_path) for im_path in im_file_paths] 
#         save_image = np.zeros((ims[0].shape[0], ims[0].shape[1] * 4, 3))
#         save_image[:,:ims[0].shape[1],:] = ims[0]
#         save_image[:,ims[0].shape[1]:ims[0].shape[1]*2,:] = ims[1]
#         save_image[:,ims[0].shape[1]*2:ims[0].shape[1]*3,:] = ims[2]
#         save_image[:,ims[0].shape[1]*3:ims[0].shape[1]*4,:] = ims[3]
#         if image_class not in file_no_per_class:
#             file_no_per_class[image_class] = 1
#         save_image_path = os.path.join(POS_IMG_SAMPLES, image_class)
#         if not os.path.exists(save_image_path):
#             os.makedirs(save_image_path)
#         save_image_path = os.path.join(save_image_path, sample_result_image_name(file_no_per_class[image_class]))
#         cv2.imwrite(save_image_path, save_image)
#         file_no_per_class[image_class] += 1
#     if image_class not in correct_per_class:
#         correct_per_class[image_class] = 0
#     correct_per_class[image_class] += 1

# print "Correct Samples stored"

# file_no_per_class = {}
# wrong_per_class = {}
# for index, pos_sample in enumerate(FALSE):
#     if index and not index % 500:
#         print "Wrong processed", index, '|', len(FALSE)
#     im_file_name, crop, frames = pos_sample
#     image_class = im_file_name.split('/')[-1].split('_')[1].upper()
#     if np.random.randint(5) - 4 == 0:
#         im_file_paths = [os.path.join(HOME_DIR, im_file_name, 'Image' + str(frame) + '.jpg') for frame in frames]      
#         ims = [cv2.imread(im_path) for im_path in im_file_paths] 
#         save_image = np.zeros((ims[0].shape[0], ims[0].shape[1] * 4, 3))
#         save_image[:,:ims[0].shape[1],:] = ims[0]
#         save_image[:,ims[0].shape[1]:ims[0].shape[1]*2,:] = ims[1]
#         save_image[:,ims[0].shape[1]*2:ims[0].shape[1]*3,:] = ims[2]
#         save_image[:,ims[0].shape[1]*3:ims[0].shape[1]*4,:] = ims[3]
#         if image_class not in file_no_per_class:
#             file_no_per_class[image_class] = 1
#         save_image_path = os.path.join(NEG_IMG_SAMPLES, image_class)
#         if not os.path.exists(save_image_path):
#             os.makedirs(save_image_path)
#         save_image_path = os.path.join(save_image_path, sample_result_image_name(file_no_per_class[image_class]))
#         cv2.imwrite(save_image_path, save_image)
#         file_no_per_class[image_class] += 1
#     if image_class not in wrong_per_class:
#         wrong_per_class[image_class] = 0
#     wrong_per_class[image_class] += 1
    
# print "Wrong Samples stored"


# plot = {}
# for k,v in correct_per_class.items():
#     if k in wrong_per_class:
#         plot[k] = [v, wrong_per_class[k]]
#     else:
#         plot[k] = [v, 0]

# with open('store_test.pkl', 'wb') as f:
#     pickle.dump([plot], f, pickle.HIGHEST_PROTOCOL)

# # data to plot
# n_groups = len(plot)
# correct_bar = map(lambda x : x[1][0], plot.items())
# wrong_bar = map(lambda x : x[1][1], plot.items())
 
# # create plot
# fig = plt.figure()
# ax = plt.subplot(111)
# index = np.arange(n_groups)
# bar_width = 0.35
# opacity = 0.8
 
# rects1 = ax.bar(index, correct_bar, bar_width,
#                  alpha=opacity,
#                  color='r',
#                  label='Wrong')
 
# rects2 = ax.bar(index + bar_width, wrong_bar, bar_width,
#                  alpha=opacity,
#                  color='g',
#                  label='Correct')
 
# print "Plotted graphs for both correct and wrong"
# ax.xlabel('Class')
# ax.ylabel('Samples')
# ax.title('Correct Wrong')
# ax.xticks(index + bar_width, tuple(plot.keys()))
# ax.legend()
# fig.savefig('abc.jpg')


'''
Output File Code Finish
'''




