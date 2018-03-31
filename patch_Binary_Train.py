
# coding: utf-8

# In[ ]:


import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio 
import cv2
import pickle
import time
import random
import sys
from PIL import Image
from collections import namedtuple as nt


# In[ ]:


'''
User params to run the network
'''
all_params = sys.argv[1:][:]
if "help" in all_params:
    print "GPU_AVAILABLE -", "gpu"
    print "MODEL_NAME -", "name"
    print "BATCH_SIZE -", "batch"
    print "EPOCHS -", "epoch"
    print "INITIAL_EPOCH -", "start"
    print "LR -", "lr"
    print "LR_DECAY -", "decay"
    print "LOSS - ", "loss"
    sys.exit()
params = {}
if '-f' not in all_params:
    params = dict([tuple(each_param.split('=')) for each_param in all_params])
GPU_AVAILABLE = params.get("gpu", "1")
MODEL_NAME = params.get("name", 'EXPERIMENT_111' + str(time.time()))
BATCH_SIZE = int(float(params.get("batch", 128)))
EPOCHS = int(float(params.get("epoch", 64)))
INITIAL_EPOCH = int(float(params.get("start", 0)))
LR = float(params.get("lr", 0.05))
LR_DECAY = float(params.get("decay", 0.0000125))
LOSS = params.get("loss", 'mean_squared_error')


# In[ ]:


os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_AVAILABLE


# In[ ]:


from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input
from keras.utils import data_utils, plot_model
from keras import backend as K
from keras.callbacks import TensorBoard
K.clear_session()


# In[ ]:


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


# ### Run the notebooks to get access to the following
# 
# 1. lstm_model - Function
# 2. EpochCallback - Class
# 3. ChronoDataSet - Class

# In[ ]:


# %run compare_model_and_callback.ipynb
# %run patch_DataLoad_Binary.ipynb
from model_and_callback import lstm_model, EpochCallback
from patch_DataLoad_Binary import ChronoDataSet


# In[ ]:


TYNAMO_HOME_DIR = '/l/vision/v7/gdhody/chrono/'
BLITZLE_HOME_DIR = '/nfs/blitzle/home/data/vision5/gdhody/chrono/'
HOME_DIR = TYNAMO_HOME_DIR


# In[ ]:


model_input, model_output = lstm_model(
                                        input_shape=(4, 80, 80, 3),
                                        resnet=True,
                                        dense_feature_to_compare=0,
                                        drop_feature=0,
                                        compare_layer=True,
                                        compare_layer_dense=512,
                                        drop_compare_dense=0,
                                        lstm_cells=[(128, True, 'sum')],
                                        many_to_many=False,
                                        dense_layer_after_lstm=0,
                                        multi_output=False,
                                        number_of_classes=1,
                                     )
model = Model(inputs=[model_input], outputs=[model_output])
output_layer_name = model.layers[-1].name
print model.summary()


# In[ ]:


HDF5_PATH = os.path.join(HOME_DIR, 'patch.hdf5')
PICKLE_PATH = os.path.join(HOME_DIR, 'patch.pkl')
MODEL_PLOT_FILE_PATH = os.path.join(HOME_DIR, 'models', MODEL_NAME)
TENSORBOARD_PATH = os.path.join(MODEL_PLOT_FILE_PATH, "tensorboard")
if not os.path.exists(MODEL_PLOT_FILE_PATH):
    os.makedirs(MODEL_PLOT_FILE_PATH)
if not os.path.exists(TENSORBOARD_PATH):
    os.makedirs(TENSORBOARD_PATH)


# In[ ]:


train_generator = ChronoDataSet(
                                    PICKLE_PATH,
                                    HDF5_PATH,
                                    batch_size=BATCH_SIZE,
                                    start_epoch=INITIAL_EPOCH,
                                    mode='TRAIN',
                                    dataset='UCF'
                               )
print "Train Data Generator Ready"
STEPS_PER_EPOCH = train_generator.getBatchRunsPerEpoch()


# In[ ]:


validation_generator = ChronoDataSet(
                                        PICKLE_PATH,
                                        HDF5_PATH,
                                        batch_size=BATCH_SIZE,
                                        start_epoch=INITIAL_EPOCH,
                                        mode='VALIDATION',
                                        dataset='UCF'
                                    )
print "Validation Data Generator Ready"


# In[ ]:


#LR = (1. / ( 1. + (float(INITIAL_EPOCH) * LR_DECAY * float(train_generator.getBatchRunsPerEpoch())))) * LR
sgd = SGD(lr=LR, decay=LR_DECAY, momentum=0.9, nesterov=True)
model.compile(
                loss={output_layer_name : LOSS},
                optimizer=sgd,
                metrics=['accuracy']
             )

tensorboard_callback = TensorBoard(
                                    log_dir=TENSORBOARD_PATH, 
                                    histogram_freq=0, 
                                    batch_size=BATCH_SIZE,  
                                    write_graph=True, 
                                    write_images=True
                                  )


# In[ ]:


plot_model(model, to_file=os.path.join(MODEL_PLOT_FILE_PATH, 'model_plot' + '.png'))
model_json = model.to_json()
with open(os.path.join(MODEL_PLOT_FILE_PATH, 'model' + '.json'), "w") as json_file:
    json_file.write(model_json)
print "Model plotting done and saved in json file"


# In[ ]:


print "-------------------------------------------------------------------"
print "RUN PARAMS"
print "MODEL_NAME =", MODEL_NAME
print "GPU =", GPU_AVAILABLE
print "BATCH_SIZE =", BATCH_SIZE
print "STARTING LR", (1. / ( 1. + (float(INITIAL_EPOCH) * LR_DECAY * float(STEPS_PER_EPOCH)))) * LR
print "LR_DECAY", LR_DECAY
print "-------------------------------------------------------------------"


# In[ ]:


if INITIAL_EPOCH:
    LOAD_WEIGHT = os.path.join(MODEL_PLOT_FILE_PATH, 'weights', 'EPOCH_' + str(INITIAL_EPOCH) + '.h5')
    model.load_weights(LOAD_WEIGHT, by_name=True)
    print "Weights from", str(INITIAL_EPOCH), "epoch Loaded"
START_ITERATION = INITIAL_EPOCH * STEPS_PER_EPOCH
model.fit_generator(
                        generator=train_generator,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        verbose=1, 
                        callbacks=[EpochCallback(MODEL_NAME, MODEL_PLOT_FILE_PATH, STEPS_PER_EPOCH, START_ITERATION), 
                                   tensorboard_callback], 
                        validation_data=validation_generator,
                        validation_steps=validation_generator.getBatchRunsPerEpoch(),
                        max_queue_size=9,
                        workers=3,
                        use_multiprocessing=True,
                        initial_epoch=INITIAL_EPOCH
                   )


# In[ ]:


# for layyer in model.layers:
#     print layyer.name

