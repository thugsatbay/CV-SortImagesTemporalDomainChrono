
# coding: utf-8

# In[30]:


get_ipython().magic(u'matplotlib inline')
import glob
import os
import numpy as np
import scipy.io as sio
import time
import pickle
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import random
from PIL import Image


# In[15]:


TYNAMO_HOME_DIR = '/nfs/tynamo/home/data/vision7/gdhody/chrono/'
BLITZLE_HOME_DIR = '/nfs/blitzle/home/data/vision5/gdhody/chrono/'
HOME_DIR = TYNAMO_HOME_DIR
HDF5_PATH = os.path.join(HOME_DIR, 'storage.hdf5')
PICKLE_PATH = os.path.join(HOME_DIR, 'storage.pkl')


# In[26]:


LABEL_FILE = os.path.join(HOME_DIR, 'UCF_HMDB_ACT.mat')
PICKLE_PATH = os.path.join(HOME_DIR, 'patch.pkl')
HDF5_PATH = os.path.join(HOME_DIR, 'patch.hdf5')


# In[17]:


TRAIN_DATA = sio.loadmat(LABEL_FILE)


# In[18]:


REMOVE_CLASSES = [
    'MILITARYPARADE',
    'HAIRCUT',
    'TAICHI',
    'YOYO',
    'APPLYEYEMAKEUP',
    'BABYCRAWLING',
    'PLAYINGTABLA',
    'WRITINGONBOARD',
    'ROCKCLIMBINGINDOOR',
    'BANDMARCHING',
    'DRUMMING',
    'PLAYINGDAF',
    'BLOWINGCANDLES',
    'PLAYINGDHOL',
    'MOPPINGFLOOR',
    'PLAYINGPIANO',
    'TYPING',
    'SKIJET',
    'HEADMASSAGE',
    'PLAYINGSITAR',
    'HORSERACE',
    'SKYDIVING',
    'PLAYINGFLUTE',
    'APPLYLIPSTICK',
    'BRUSHINGTEETH',
    'SURFING',
    'JUGGLINGBALLS',
    'PLAYINGGUITAR',
    'SHAVINGBEARD',
    'BILLIARDS',
    'KNITTING',
    'FENCING',
    'BOXINGSPEEDBAG',
    'MIXING',
    'BLOWDRYER',
    'SALSASPIN',
    'CUTTINGINKITCHEN',
    'RAFTING',
    'HORSERACE',
]


# In[19]:


TRAIN_UCF = []
TRAIN_HMDB = []
for index, sample in enumerate(TRAIN_DATA['filename']):
    image_file_name = sample[0][0]
    if 'UCF101' in image_file_name:
        image_class = image_file_name.split('/')[-1].split('_')[1].upper()
        if image_class not in REMOVE_CLASSES:
            image_path = os.path.join(HOME_DIR, image_file_name)
            if len(glob.glob(image_path)):
                frames, crop = TRAIN_DATA['frame'][index], TRAIN_DATA['crop'][index]
                TRAIN_UCF.append((image_file_name, np.array(list(crop)), np.array(list(frames))))
    if 'HMDB51' in image_file_name:
        image_path = os.path.join(HOME_DIR, image_file_name)
        if len(glob.glob(image_path)):
            frames, crop = TRAIN_DATA['frame'][index], TRAIN_DATA['crop'][index]
            TRAIN_HMDB.append((image_file_name, np.array(list(crop)), np.array(list(frames))))


# In[20]:


print len(TRAIN_UCF), len(TRAIN_HMDB)
random.shuffle(TRAIN_UCF)
random.shuffle(TRAIN_HMDB)
random.shuffle(TRAIN_UCF)
random.shuffle(TRAIN_HMDB)


# In[21]:


with open(PICKLE_PATH, 'wb') as f:
    pickle.dump([TRAIN_UCF, TRAIN_HMDB], f, pickle.HIGHEST_PROTOCOL)


# In[22]:


with open(PICKLE_PATH, 'rb') as f:
    TRAIN_UCF, TRAIN_HMDB = pickle.load(f)


# In[24]:


train_samples_ucf, validation_samples_ucf = (70 * len(TRAIN_UCF)) / 100, len(TRAIN_UCF) / 10
test_samples_ucf = len(TRAIN_UCF) - train_samples_ucf - validation_samples_ucf

train_samples_hmdb, validation_samples_hmdb = (70 * len(TRAIN_HMDB)) / 100, len(TRAIN_HMDB) / 10
test_samples_hmdb = len(TRAIN_HMDB) - train_samples_hmdb - validation_samples_hmdb


# In[35]:


train_shape_ucf = (train_samples_ucf, 4, 100, 100, 3)
val_shape_ucf = (validation_samples_ucf, 4, 100, 100, 3)
test_shape_ucf = (test_samples_ucf, 4, 100, 100, 3)


# In[36]:


hdf5_file = h5py.File(HDF5_PATH, mode='w')
hdf5_file.create_dataset("train_img", train_shape_ucf, np.float32)
hdf5_file.create_dataset("val_img", val_shape_ucf, np.float32)
hdf5_file.create_dataset("test_img", test_shape_ucf, np.float32)


# In[37]:


image_size, image_padding = 80, 20
for index, sample in enumerate(TRAIN_UCF[:train_samples_ucf]):
    if index % 10000 == 0 and index > 1:
        print 'Train data: {}/{}'.format(index, test_samples_ucf)
    im_file_name, crop, frames = sample
    im_file_paths = [os.path.join(HOME_DIR, im_file_name, 'Image' + str(frame) + '.jpg') for frame in frames]
    images = [Image.open(im_file_path) for im_file_path in im_file_paths]
    
    top_point = crop[0]
    left_point = crop[1]

    images = [image.crop((left_point, top_point, 
                          left_point + image_size + image_padding, 
                          top_point + image_size + image_padding)) 
                          for image in images]
    
    images = [np.array(image, dtype=np.float32) for image in images]
    img = np.stack((images[0], images[1], images[2], images[3]), axis=0)
    hdf5_file["train_img"][index, ...] = img[None]


# In[38]:


image_size, image_padding = 80, 20
for index, sample in enumerate(TRAIN_UCF[train_samples_ucf:train_samples_ucf + validation_samples_ucf]):
    if index % 10000 == 0 and index > 1:
        print 'Train data: {}/{}'.format(index, validation_samples_ucf)
    im_file_name, crop, frames = sample
    im_file_paths = [os.path.join(HOME_DIR, im_file_name, 'Image' + str(frame) + '.jpg') for frame in frames]
    images = [Image.open(im_file_path) for im_file_path in im_file_paths]
    
    top_point = crop[0]
    left_point = crop[1]

    images = [image.crop((left_point, top_point, 
                          left_point + image_size + image_padding, 
                          top_point + image_size + image_padding)) 
                          for image in images]
    
    images = [np.array(image, dtype=np.float32) for image in images]
    img = np.stack((images[0], images[1], images[2], images[3]), axis=0)
    hdf5_file["val_img"][index, ...] = img[None]


# In[40]:


image_size, image_padding = 80, 20
for index, sample in enumerate(TRAIN_UCF[train_samples_ucf + validation_samples_ucf:]):
    if index % 5000 == 0 and index > 1:
        print 'Train data: {}/{}'.format(index, test_samples_ucf)
    im_file_name, crop, frames = sample
    im_file_paths = [os.path.join(HOME_DIR, im_file_name, 'Image' + str(frame) + '.jpg') for frame in frames]
    images = [Image.open(im_file_path) for im_file_path in im_file_paths]
    
    top_point = crop[0]
    left_point = crop[1]

    images = [image.crop((left_point, top_point, 
                          left_point + image_size + image_padding, 
                          top_point + image_size + image_padding)) 
                          for image in images]
    
    images = [np.array(image, dtype=np.float32) for image in images]
    img = np.stack((images[0], images[1], images[2], images[3]), axis=0)
    hdf5_file["test_img"][index, ...] = img[None]

