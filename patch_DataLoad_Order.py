
# coding: utf-8

# In[1]:


import itertools
import scipy.io as sio 
import numpy as np
import os
import cv2
import random
import h5py
from PIL import Image
import time
import pickle


# In[2]:


os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


# In[3]:


from keras.utils.np_utils import to_categorical
from keras.utils.data_utils import Sequence


# In[29]:


class ChronoDataSet(Sequence):
    def __init__(self,
                label_file_path,
                data_file_path,
                batch_size=128,
                start_epoch=0,
                mode='TRAIN',
                dataset='UCF'):

        '''
        Constant parameters
        '''
        self.n_frames = 4
        self.image_padding = 20
        self.image_size = 80
        self.image_jitter = 5
        self.channels = 3
        self.epoch = start_epoch + 1
        
        self.batch_size = batch_size

        self.order_type = [
                            list(ele) for ele in 
                            itertools.permutations(range(self.n_frames)) 
                            if list(ele)[0] < list(ele)[-1]
                          ]

#         TRAIN_UCF, TRAIN_HMDB = [], []
#         with open(label_file_path, 'rb') as f:
#             TRAIN_UCF, TRAIN_HMDB = pickle.load(f)
        
#         self.label = TRAIN_UCF
#         if dataset == 'HMDB':
#             self.label = TRAIN_HMDB
            
        self.data_file_path = data_file_path
        
        data = h5py.File(data_file_path,'r')
        
        self.samples = data['train_img'].shape[0]
        
        self.mode = "train_img"
        if mode == "VALIDATION":
            self.mode = "val_img"
            self.samples = data['val_img'].shape[0]
        elif mode == "TEST":
            print "TEST MODE INITIALIZED"
            self.mode = "test_img"
            self.samples = data['test_img'].shape[0]
            
        data.close()
        
        self.blocks = [item for item in xrange(0, self.samples, self.batch_size)][:-1]
        self.current_img_size = self.image_size + self.image_padding

    def getBlocks(self):
        return self.blocks;
    
    def __len__(self):
        # return batch size?
        return int(self.samples / self.batch_size)

    def getBatchRunsPerEpoch(self):
        return int(self.samples / self.batch_size)
    
    def opticalFlow(frame1, frame2):
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    def jitter_image(self, image):
        startxy = (np.random.randint(0, self.image_padding), np.random.randint(0, self.image_padding))
        
        # Start points by default
        newx, newy = startxy[0], startxy[1]
        
        # Jitter points
        sx, sy = random.randint(-self.image_jitter, self.image_jitter), random.randint(-self.image_jitter, self.image_jitter)
        
        # Jitter should not move the crop window outside the image
        if startxy[0] + sx > 0 and startxy[0] + self.image_size + sx < self.current_img_size: newx += sx 
        if startxy[1] + sy > 0 and startxy[1] + self.image_size + sy < self.current_img_size: newy += sy
        return image[newx : newx + self.image_size, newy : newy + self.image_size, ...]
    
    def on_epoch_end(self):
        random.shuffle(self.blocks)
        print "\nEPOCH", self.epoch, "Ends. Data blocks shuffled!"
        self.epoch += 1
    
    def __getitem__(self, idx):
        
        start_time = time.time()
            
        idxcounter = idx % self.getBatchRunsPerEpoch()
        
        with h5py.File(self.data_file_path,'r') as f:
            batch_data = f[self.mode][self.blocks[idxcounter]:self.blocks[idxcounter] + self.batch_size, ...]
        np.random.shuffle(batch_data)
        
        # Spatial Jittering
        images = [[self.jitter_image(each_image) for each_image in sample] 
                  for sample in batch_data]
        

        # Horizontal Flip
        images = [[cv2.flip(images[index][image_index], 1) for image_index in xrange(self.n_frames)] 
                  if np.random.randint(0,2) else images[index] for index in xrange(self.batch_size)]
        

        # Channel Splitting
        images = [[np.stack((images[index][image_index][:,:,np.random.randint(0, 3)],)*3, axis=2) 
                   for image_index in xrange(self.n_frames)] for index in xrange(self.batch_size)]
        
#         for image in images:
#             print len(image), image[0].shape, image[1].shape, image[2].shape, image[3].shape

        randomness = [np.random.randint(0, len(self.order_type)) for batch_size in xrange(self.batch_size)]
        inverse = [np.random.randint(0, 2) for batch_size in xrange(self.batch_size)]
        
        # Randomly +ve and -ve test case
        images = [(np.stack(tuple([images[index][random_index] 
                  for random_index in self.order_type[randomness[index]] ]), axis=0)[::1 * ((-1)**(inverse[index] + 1))], randomness[index]) 
                  for index in xrange(self.batch_size)]
        
#         print "After labelling"
#         for image in images:
#             print len(image[1]), image[1][0].shape, image[1][1].shape, image[1][2].shape, image[1][3].shape

        
    
        labels = np.array([ np.array([ np.array([int(idx == ele) for idx in xrange(4)])
                                      for ele in self.order_type[randomness[index]][::1 * ((-1)**(inverse[index] + 1))] ])
                           for index in xrange(self.batch_size) ])

        images = np.array([image for image, _ in images])
        
        images = np.array(images, dtype=np.float32)
        
        images -= 97.3
        
#         if not (idxcounter + 1)%500:
#             print "\nLOAD TIME | Epoch", str(self.epoch), "| Batch", str(self.blocks[idxcounter]), "| processed with time : ", str(time.time() - start_time), "\n"
        
#         for index in xrange(self.batch_size):
#             print self.order_type[randomness[index]], inverse[index]-1
        
        return (images, labels)


# *Testing of the module*
# ```
# TYNAMO_HOME_DIR = '/nfs/tynamo/home/data/vision7/gdhody/chrono/'
# BLITZLE_HOME_DIR = '/nfs/blitzle/home/data/vision5/gdhody/chrono/'
# HOME_DIR = TYNAMO_HOME_DIR
# HDF5_PATH = os.path.join(HOME_DIR, 'patch.hdf5')
# PICKLE_PATH = os.path.join(HOME_DIR, 'patch.pkl')
# cds = ChronoDataSet(PICKLE_PATH, HDF5_PATH)
# cds.__getitem__(99)
# ```

# In[126]:


# import matplotlib.pyplot as plt
# %matplotlib inline
# TYNAMO_HOME_DIR = '/nfs/tynamo/home/data/vision7/gdhody/chrono/'
# BLITZLE_HOME_DIR = '/nfs/blitzle/home/data/vision5/gdhody/chrono/'
# HOME_DIR = TYNAMO_HOME_DIR
# HDF5_PATH = os.path.join(HOME_DIR, 'patch.hdf5')
# PICKLE_PATH = os.path.join(HOME_DIR, 'patch.pkl')
# cds = ChronoDataSet(PICKLE_PATH, HDF5_PATH)
# print cds.__getitem__(50)[0][0].shape
# np.amax(cds.__getitem__(50)[0][0])
# for inn, frames in enumerate(im):
#     index = 1
#     print inn, frames.shape
#     plt.figure()
#     plt.subplots_adjust(wspace=1, hspace=1)
#     plt.subplots(figsize=(15, 20))
#     for frame in frames:
#         suub_plt = plt.subplot(1, 4, index)
#         plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), interpolation='nearest', aspect='equal')
#         index += 1
# print l


# In[29]:


# TYNAMO_HOME_DIR = '/nfs/tynamo/home/data/vision7/gdhody/chrono/'
# BLITZLE_HOME_DIR = '/nfs/blitzle/home/data/vision5/gdhody/chrono/'
# HOME_DIR = TYNAMO_HOME_DIR
# HDF5_PATH = os.path.join(HOME_DIR, 'patch.hdf5')
# with h5py.File(HDF5_PATH,'r') as f:
#     print np.mean(f['train_img'])


# In[3]:


# import matplotlib.pyplot as plt
# %matplotlib inline
# TYNAMO_HOME_DIR = '/nfs/tynamo/home/data/vision7/gdhody/chrono/'
# BLITZLE_HOME_DIR = '/nfs/blitzle/home/data/vision5/gdhody/chrono/'
# HOME_DIR = TYNAMO_HOME_DIR
# HDF5_PATH = os.path.join(HOME_DIR, 'patch.hdf5')
# PICKLE_PATH = os.path.join(HOME_DIR, 'patch.pkl')
# cds = ChronoDataSet(PICKLE_PATH, HDF5_PATH)
# cds.__getitem__(50)[1]


# In[10]:


np.array([
    np.array([1 if idx == ((ele+1) - 1) else 0 for idx in xrange(4)])
    for ele in [1,0,2,3]]
)

