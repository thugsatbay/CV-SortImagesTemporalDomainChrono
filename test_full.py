
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
from keras.layers import Input
from keras.utils import data_utils, plot_model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers.core import Activation, Dense
from keras.layers.merge import concatenate
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



EPOCH = str(36)
BATCH_SIZE = 36
LR, LR_DECAY = 0.01, 0.0000625
LOSS = 'mean_squared_error'
MODEL_NAME = 'RES_IMG_OPT_CONCAT_1/'
MODEL_PATH = os.path.join(HOME_DIR, 'models', MODEL_NAME, 'model.json')
MODEL_WEIGHTS = os.path.join(HOME_DIR, 'models', MODEL_NAME, 'weights', 'EPOCH_' + EPOCH + '.h5')
HDF5_PATH = os.path.join(HOME_DIR, 'storage.hdf5')
PICKLE_PATH = os.path.join(HOME_DIR, 'storage.pkl')


# In[66]:


# %run model_and_callback.ipynb
# %run patch_DataLoad_Binary.ipynb
from model_and_callback_full import lstm_model, EpochCallback
from patch_DataLoad_Binary_full_test import ChronoDataSet


# In[67]:


dic_model_params = {
'input_shape_img':(4, 112, 112, 3),
'input_shape_opt':(4, 112, 112, 3),
'resnet':True,
'dense_feature_to_compare':0,
'drop_feature':0,
'compare_layer':True,
'compare_layer_dense':512,
'drop_compare_dense':0,
'lstm_cells':[(128, True, 'sum')],
'many_to_many':False,
'dense_layer_after_lstm':0,
'multi_output':False,
'number_of_classes':1,
}
print "IMG Model Build started..."
model_input_img, model_output_img = lstm_model(
                                        input_shape=dic_model_params['input_shape_img'],
                                        resnet=dic_model_params['resnet'],
                                        dense_feature_to_compare=dic_model_params['dense_feature_to_compare'],
                                        drop_feature=dic_model_params['drop_feature'],
                                        compare_layer=dic_model_params['compare_layer'],
                                        compare_layer_dense=dic_model_params['compare_layer_dense'],
                                        drop_compare_dense=dic_model_params['drop_compare_dense'],
                                        lstm_cells=dic_model_params['lstm_cells'],
                                        many_to_many=dic_model_params['many_to_many'],
                                        dense_layer_after_lstm=dic_model_params['dense_layer_after_lstm'],
                                        multi_output=dic_model_params['multi_output'],
                                        number_of_classes=dic_model_params['number_of_classes'],
                                        model_name='IMG_'
                                     )
print "IMG Model Build completed."
print "OPT Model Build started..."
model_input_opt, model_output_opt = lstm_model(
                                        input_shape=dic_model_params['input_shape_opt'],
                                        resnet=dic_model_params['resnet'],
                                        dense_feature_to_compare=dic_model_params['dense_feature_to_compare'],
                                        drop_feature=dic_model_params['drop_feature'],
                                        compare_layer=dic_model_params['compare_layer'],
                                        compare_layer_dense=dic_model_params['compare_layer_dense'],
                                        drop_compare_dense=dic_model_params['drop_compare_dense'],
                                        lstm_cells=dic_model_params['lstm_cells'],
                                        many_to_many=dic_model_params['many_to_many'],
                                        dense_layer_after_lstm=dic_model_params['dense_layer_after_lstm'],
                                        multi_output=dic_model_params['multi_output'],
                                        number_of_classes=dic_model_params['number_of_classes'],
                                        model_name='OPT_'
                                     )
print "OPT Model Build completed."

concat_decision = concatenate([model_output_img, model_output_opt], name='Decision_IMG_OPT', axis=-1)
concat_decision = Dense(dic_model_params['number_of_classes'], name='class_dense')(concat_decision)
concat_decision = Activation('sigmoid', name='output')(concat_decision)
model = Model(inputs=[model_input_img, model_input_opt], outputs=[concat_decision])

# loaded_model_json = None
# with open(MODEL_PATH, 'r') as f:
#     loaded_model_json = f.read()
# model = model_from_json(loaded_model_json)
# # load weights into new model
print model.summary()



sgd = SGD(lr=LR, decay=LR_DECAY, momentum=0.9, nesterov=True)
model.compile(
                loss={'output': LOSS},
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


# # In[74]:

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

# # # data to plot
# # n_groups = len(plot)
# # correct_bar = map(lambda x : x[1][0], plot.items())
# # wrong_bar = map(lambda x : x[1][1], plot.items())
 
# # # create plot
# # fig = plt.figure()
# # ax = plt.subplot(111)
# # index = np.arange(n_groups)
# # bar_width = 0.35
# # opacity = 0.8
 
# # rects1 = ax.bar(index, correct_bar, bar_width,
# #                  alpha=opacity,
# #                  color='r',
# #                  label='Wrong')
 
# # rects2 = ax.bar(index + bar_width, wrong_bar, bar_width,
# #                  alpha=opacity,
# #                  color='g',
# #                  label='Correct')
 
# # print "Plotted graphs for both correct and wrong"
# # ax.xlabel('Class')
# # ax.ylabel('Samples')
# # ax.title('Correct Wrong')
# # ax.xticks(index + bar_width, tuple(plot.keys()))
# # ax.legend()
# # fig.savefig('abc.jpg')


# '''
# Output File Code Finish
# '''




