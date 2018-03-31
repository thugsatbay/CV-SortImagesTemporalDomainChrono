
# coding: utf-8

# In[ ]:


import numpy as np
import os
import time
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = "0"
# set_session(tf.Session(config=config))


# In[ ]:


from keras import backend as K
from keras.callbacks import Callback
from keras.engine import Layer
from keras import layers
from keras.layers import Input, LSTM, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model
from keras.utils import plot_model


# In[ ]:


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(input_shape=(224,224,3), bottom_identity_layer=True):
    
    img_input = Input(shape=input_shape)

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    if bottom_identity_layer:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((x._keras_shape[-2], x._keras_shape[-2]), name='avg_pool')(x)

    inputs = img_input
    model = Model(inputs, x, name='resnet50')
    
    return model


# In[ ]:


class LRN2D(Layer):
    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        #super(LRN2D, self).all(*args, **kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n

    def get_output(self, train):
        X = self.get_input(train)
        b, ch, r, c = K.shape(X)
        half_n = self.n // 2
        input_sqr = K.square(X)

        extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
        input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
                                   input_sqr,
                                   extra_channels[:, half_n + ch:, :, :]],
                                  axis=1)
        scale = self.k

        for i in range(self.n):
            scale += self.alpha * input_sqr[:, i:i + ch, :, :]
        scale = scale ** self.beta

        return X / scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[ ]:


def custom_lstm_layers(model,
                       lstm_layers,
                       many_to_many=False,
                       lstm_name='lstm_'):
    '''
    Defines custom lstm layers

    #Arguments

    model:
        Previous model to which lstm layers are to be attached

    lstm_layers:
        A list of tuples: (x, y, z)
            x - Units in lstm layers
            y - Bidirectional status
            z - Merge mode, TODO: None mode not supported
        Example [512] (Only 1 LSTM layer with 512 units)
        TODO: Add GRU as well

    many_to_many:
        True - many_to_may
        False - many_to_one
        Last LSTM layer output sequence
    '''

    # Add LSTM layers stacked on top of each other
    for index, cell in enumerate(lstm_layers):
        units, bi, bi_mode = cell
        lstm_layer_name = lstm_name + str(index + 1) + '_' + str(units)
        # None bi_mode not supported
        if not bi_mode: bi_mode = 'concat'
        
        if bi:
            model = Bidirectional(LSTM(units,
                                       return_sequences=many_to_many or (len(lstm_layers) != (index + 1)),
                                       name=lstm_layer_name),
                                  merge_mode=bi_mode,
                                  name='bi_' + lstm_layer_name)(model)
        else:
            model = LSTM(units,
                             return_sequences=many_to_many or (len(lstm_layers) != index + 1),
                             name=lstm_layer_name)(model)
    return model


# In[ ]:


def lstm_model(input_shape,
               resnet = False,
               dense_feature_to_compare=1024,
               drop_feature=0.5,
               compare_layer=False,
               compare_layer_dense=1024,
               drop_compare_dense=0.5,
               lstm_cells=[(512, False, 'concat')],
               many_to_many=False,
               dense_layer_after_lstm=128,
               multi_output=False,
               number_of_classes=12,
               model_name='lstm_model',
               test_mode=False):
    '''
    #Arguments

    input_shape:
        The input for the CNN
        
    resnet:
        Use resnet or caffe model for image feature extraction

    dense_feature_to_compare:
        Add a dense layer after image features, specifying a 0 won't add one

    drop_feature:
        drop layer after feature map, 0 or 0.0 will cancel the layer

    compare_layer:
        True - In a tensor v(:, dim, :) compares layer (dim, dim+1)
        TODO: Only works for dim=4

    compare_layer_dense:
        Adds a dense layer after concat step of compare_layer, specifying a 0
        won't add one

    drop_compare_dense:
        Add a dense layer for compare dense layer, specifying a 0 won't add one

    lstm_cells[
    
    ]
        A list of tuples: (x, y, z)
            x - Units in lstm layers
            y - Bidirectional status
            z - Merge mode, TODO: None mode not supported
        Example [512] (Only 1 LSTM layer with 512 units)
        TODO: Add GRU as well
    
    many_to_many:
        True - many_to_may
        False - many_to_one
        Last LSTM layer output sequence

    dense_layer_after_lstm:
        If Integer given adds a Dense layer on top of the output of LSTM
        If 0 is given no layer is added

    multi_output:
        True - Have multiple outputs

    number_of_classes:
        Number of output classes

    #NamingFormat - Caffe Net TimeDistributed
    <layer><depth>_<filter>_<kernel>_<stride>
        
    -------------------- *** Caffe Net - TimeDistributed *** ------------------
    
    Original CaffeNet model, modified as below
    Changes Made:   (Original Copy)
    conv1_96_11_4   kernel_size=(11, 11), strides=(4, 4)
    flatten6        after this layer customized for problem

    INPUT (None, TD, W, H, C)
    =========================
    conv1_96_7_3
    mpool1_3_2
    lrnorm1
    conv2_256_3_1
    mpool2_3_2
    lrnorm2
    conv3_384_3_1
    bn3
    relu3
    conv4_384_3_1
    bn4
    relu4
    conv5_256_3_1
    bn5
    relu5
    mpool5_3_2
    flatten6
    =========================

    -------------------- *** Caffe Net - TimeDistributed *** ------------------

    #NamingFormat - LSTM Configurable Model
    [] Means a stack of layers, one after another (depth/height)
    ? Means the parameter is configurable by the value specified in if condition
    {TD} Means the output of that layer may or may not be in time dimension

    ---------------------- *** LSTM Configurable Model *** --------------------
    
    INPUT (None, TD, C)
    ===================
    if dense_feature_to_compare:
        dense6_?
        bn6
        relu6
    if drop_feature:
        drop6_?
    if compare_layer:
        concat_01
        concat_12
        concat_23
        concat_03
        if compare_layer_dense:
            dense_01_?
            bn_01
            relu_01
            dense_12_?
            bn_12
            relu_12
            dense_23_?
            bn_23
            relu_23
            if drop_compare_dense:
                dropout_01_?
                dropout_12_?
                dropout_23_?
    [lstm_sort_?]
    if dense_layer_after_lstm:
        dense7_lstm
    if many_to_many and not multi_output:
        flatten7
    lstm_catg{TD}
    lstm_output{TD}
    ===================

    ---------------------- *** LSTM Configurable Model *** --------------------
    '''
    print "Model Creation Started"
    
    # Some sanity checks
    if multi_output: many_to_many=True
    
    # Input variables used in model
    compare_layer_image_dim = None
    
    input_layer = Input((input_shape), name=model_name+'input')
    
    # Resnet or Caffe Model for feature extraction
    if resnet:
        resNet = ResNet50(input_shape=input_shape[1:], bottom_identity_layer=False)
        model = TimeDistributed(resNet, name=model_name+'TD_resnet')(input_layer)
    
    else:
        # conv1_96_7_3
        model = TimeDistributed(Conv2D(96, 
                                       kernel_size=(11, 11),
                                       strides=(3, 3),
                                       activation='relu',
                                       padding='valid',
                                       name=model_name+'conv1_96_7_3'),
                                name=model_name+'TD_conv1_96_7_3')(input_layer)

        # mpool1_3_2
        model = TimeDistributed(MaxPooling2D(pool_size=(3, 3), 
                                             strides=(2, 2),
                                             padding='valid',
                                             name=model_name+'mpool1_3_2'),
                                name=model_name+'TD_mpool1_3_2')(model)

        # lrnorm1
        model = TimeDistributed(LRN2D(name=model_name+'lrnorm1'), name=model_name+'TD_lrnorm1')(model)

        # conv2_256_3_1
        model = TimeDistributed(Conv2D(256,
                                       kernel_size=(5, 5),
                                       strides=(1, 1),
                                       padding='same',
                                       activation='relu',
                                       name=model_name+'conv2_256_3_1'),
                                name=model_name+'TD_conv2_256_3_1')(model)

        # mpool2_3_2
        model = TimeDistributed(MaxPooling2D(pool_size=(3, 3),
                                             strides=(2, 2),
                                             padding='valid',
                                             name=model_name+'mpool2_3_2'),
                                name=model_name+'TD_mpool2_3_2')(model)

        # lrnorm2
        model = TimeDistributed(LRN2D(name=model_name+'lrnorm2'), name=model_name+'TD_lrnorm2')(model)

        # conv3_, bn, relu <3,4,5>
        filters = [384, 384, 256]
        for layer in xrange(3, 6, 1):
            conv_step_name = model_name+'conv' + str(layer) + '_' + str(filters[layer - 3]) + '_3_1'

            model = TimeDistributed(Conv2D(filters[layer - 3],
                                           kernel_size=(3, 3),
                                           strides=(1, 1),
                                           padding='same',
                                           name=model_name+conv_step_name),
                                    name=model_name+'TD_' + conv_step_name)(model)

            model = TimeDistributed(BatchNormalization(name=model_name+'bn_' + str(layer)), 
                                    name=model_name+'TD_bn_' + str(layer))(model)

            model = TimeDistributed(Activation('relu', name=model_name+'relu' + str(layer)),
                                    name=model_name+'TD_relu_' + str(layer))(model)


        # mpool5_3_2
        model = TimeDistributed(MaxPooling2D(pool_size=(3, 3),
                                             strides=(2, 2),
                                             padding='valid',
                                             name=model_name+'mpool5_3_2'),
                                name=model_name+'TD_mpool5_3_2')(model)

    # flatten6, feature map from CNN
    model = TimeDistributed(Flatten(name=model_name+'flatten6'), name=model_name+'TD_flatten6')(model)

    # If no dense layer specified below on feature map get this layer dimension
    if not dense_feature_to_compare: compare_layer_image_dim = model._keras_shape[-1]

    print "Model Base Created"
    
    # Add a dense layer to feature of images
    if dense_feature_to_compare:
        # Update compare_layer_image_dim, if using compare_layer in model
        compare_layer_image_dim = dense_feature_to_compare
        model = TimeDistributed(Dense(dense_feature_to_compare,
                                      name=model_name+'dense6_' + str(dense_feature_to_compare)),
                                name=model_name+'TD_dense6_' + str(dense_feature_to_compare))(model)
        
        # bn6
        model = TimeDistributed(BatchNormalization(name=model_name+'bn6'),
                                name=model_name+'TD_bn6')(model)

        # relu6
        model = TimeDistributed(Activation('relu', name=model_name+'relu6'),
                                name=model_name+'TD_relu6')(model)

    if drop_feature:
        # drop6
        model = TimeDistributed(Dropout(drop_feature, name=model_name+'drop6'),
                                name=model_name+'TD_drop6' + str(drop_feature))(model)

    # Introducing LSTM compare layer
    if compare_layer:
        # image_<0,1,2,3>
        image_f = [Lambda(lambda model : model[:, frame, :], 
                          output_shape=(compare_layer_image_dim,),
                          name=model_name+'image_' + str(frame))(model) 
                   for frame in xrange(4)]

        # compare <01,12,23> 
        concat_01 = concatenate([
                                    image_f[0],
                                    image_f[1]
                                ],
                                name=model_name+'compare_01',
                                axis=-1)
        
        concat_12 = concatenate([
                                    image_f[1],
                                    image_f[2]
                                ],
                                name=model_name+'compare_12',
                                axis=-1)

        concat_23 = concatenate([
                                    image_f[2],
                                    image_f[3]
                                ], 
                                name=model_name+'compare_23',
                                axis=-1)

        concat_03 = concatenate([
                                    image_f[0],
                                    image_f[3]
                                ], 
                                name=model_name+'compare_03',
                                axis=-1)

        if compare_layer_dense:
            # dense<01,12,23>_7
            concat_01 = Dense(compare_layer_dense,
                              name=model_name+"dense_01_7")(concat_01)
            concat_12 = Dense(compare_layer_dense,
                              name=model_name+"dense_12_7")(concat_12)
            concat_23 = Dense(compare_layer_dense,
                              name=model_name+"dense_23_7")(concat_23)
            concat_03 = Dense(compare_layer_dense,
                              name=model_name+"dense_03_7")(concat_03)
            
            concat_01 = BatchNormalization(name=model_name+'bn_01_7')(concat_01)
            concat_12 = BatchNormalization(name=model_name+'bn_12_7')(concat_12)
            concat_23 = BatchNormalization(name=model_name+'bn_23_7')(concat_23)
            concat_03 = BatchNormalization(name=model_name+'bn_03_7')(concat_03)

            concat_01 = Activation('relu',
                                   name=model_name+'relu_01_7')(concat_01)
            concat_12 = Activation('relu',
                                  name=model_name+'relu_12_7')(concat_12)
            concat_23 = Activation('relu',
                                  name=model_name+'relu_23_7')(concat_23)
            concat_03 = Activation('relu',
                                  name=model_name+'relu_03_7')(concat_03)
            
            if drop_compare_dense:
                concat_01 = Dropout(drop_compare_dense, 
                                    name=model_name+"dropout_01_7_" + str(drop_compare_dense))(concat_01)
                concat_12 = Dropout(drop_compare_dense,
                                    name=model_name+"dropout_12_7_" + str(drop_compare_dense))(concat_12)
                concat_23 = Dropout(drop_compare_dense,
                                    name=model_name+"dropout_23_7_" + str(drop_compare_dense))(concat_23)
                concat_03 = Dropout(drop_compare_dense,
                                    name=model_name+"dropout_03_7_" + str(drop_compare_dense))(concat_23)

        # Concat the layers into 3 transitive units
        concat_0123 = concatenate([
                                      concat_01,
                                      concat_12,
                                      concat_23,
                                      concat_03,
                                  ],
                                  name=model_name+'concat_0123',
                                  axis=-1)
        
        # If LSTM layer exits reshape it for lstm input
        if lstm_cells[0][0]:
            if compare_layer_dense:
                model = Reshape(target_shape=(4, compare_layer_dense),
                                name=model_name+'reshape_' + str(compare_layer_dense))(concat_0123)
            else:
                model = Reshape(target_shape=(4, concat_01._keras_shape),
                                name=model_name+'reshape_' + str(concat_01._keras_shape))(concat_0123)
        else: model = concat_0123

    print "Model Compare Unit Finished"
    
    if lstm_cells[0][0]:
        # Add LSTM layers stacked on top of each other
        model = custom_lstm_layers(model, lstm_cells, many_to_many, lstm_name=model_name+'lstm_sort_')

    print "Model LSTM Unit Finished"
    
    return input_layer, model
    
#     # Add dense layer after lstm
#     if dense_layer_after_lstm:
#         if many_to_many:
#             # Add dense on many to many
#             model = TimeDistributed(Dense(dense_layer_after_lstm, name=model_name+'dense7_lstm'), name=model_name+'TD_dense7_lstm')(model)
#         else:
#             # Add dense on many to many
#             model = Dense(dense_layer_after_lstm, name=model_name+'dense7_lstm')(model)

#     # Many output of LSTM enabled and no multi output then flatten layer
#     if many_to_many and not multi_output: model = Flatten(name=model_name+'flatten7')(model)
    
#     print "Model Output Layers Finished"
    
#     # Class Output with respect to multiple outputs
#     if not multi_output:    
#         model = Dense(number_of_classes, name=model_name+'lstm_catg')(model)
#         model = Activation('sigmoid', name=model_name+'output')(model)
#     else:
#         model = TimeDistributed(Dense(number_of_classes, name=model_name+'lstm_catg'), name=model_name+'TD_lstm_catg')(model)
#         model = TimeDistributed(Activation('softmax', name=model_name+'lstm_output'), name=model_name+'TD_output')(model)
    
#     if test_mode:
#         model = Model(inputs=[input_layer], outputs=[model])
#         #plot_model(model, to_file=model_name + '.png')
#         print model.summary()
        
    print "Model Creation Finished"

    return input_layer, model


# *For Testing the model*
# ```
# model = lstm_model(
#                 input_shape=(4, 80, 80, 3),
#                 resnet=False,
#                 dense_feature_to_compare=0,
#                 drop_feature=0,
#                 compare_layer=True,
#                 compare_layer_dense=512,
#                 drop_compare_dense=0.25,
#                 lstm_cells=[(256, True, 'mul')],
#                 many_to_many=False,
#                 dense_layer_after_lstm=0,
#                 multi_output=False,
#                 number_of_classes=2,
#                 model_name='chrono_lstm',
#                 test_mode=True
#           )
# ```

# In[ ]:


import tensorflow as tf


# In[ ]:


class EpochCallback(Callback):
    def __init__(self, model_name, save_dir_path, epoch_size, initial_iteration=0):
        self.name = model_name
        self.epoch_run_time = time.time()
        self.total_batch_run_time = 0.0
        self.batch_run_time = 0
        self.dir = os.path.join(save_dir_path, 'weights')
        self.initial_iteration = initial_iteration
        self.epoch_size = epoch_size
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            
    def on_train_begin(self, logs=None):
        print "\n->TRAINING STARTING"
        if self.initial_iteration:
            K.set_value(self.model.optimizer.iterations, self.initial_iteration)
            print "Initial Iteration Set To", self.initial_iteration, "\n"
            
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_run_time = time.time()
        
    def on_batch_begin(self, batch, logs={}):
        self.batch_run_time = time.time()
        
    def on_batch_end(self, batch, logs={}):
        self.total_batch_run_time += time.time() - self.batch_run_time

    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.dir, "EPOCH_" + str(epoch + 1) + ".h5"))
        optimizer = self.model.optimizer
        lr = K.eval(tf.to_float(optimizer.lr) * (1. / (1. + tf.to_float(optimizer.decay) * tf.to_float(optimizer.iterations))))
        print "\n", self.name, "| EPOCH", str(epoch + 1), "ENDS"
        print "Epoch weights saved", os.path.join(self.dir, "EPOCH_" + str(epoch + 1) + ".h5")
        print "AVG BATCH GPU TIME | Batches", str(self.epoch_size), "| processed with time : ", str(self.total_batch_run_time / float(self.epoch_size))
        print "Epoch", str(epoch + 1), "Completed | Time taken", str(time.time() - self.epoch_run_time)
        print 'LEARNING RATE: {:.6f}'.format(lr), "| ITERATIONS:", K.eval(optimizer.iterations), "| DECAY:", K.eval(optimizer.decay), "\n\n"
        self.total_batch_run_time = 0.0


# In[ ]:


# from keras.applications.resnet50 import ResNet50

# resNet = ResNet50(weights=None, include_top=False)

# input_layer = Input(shape=(4, 80, 80, 3))
# curr_layer = TimeDistributed(resNet)(input_layer)
# # curr_layer = Reshape(target_shape=(4, 2048))(curr_layer)
# # curr_layer = LSTM(384, return_sequences=False)(curr_layer)
# # curr_layer = Dense(1)(curr_layer)
# model = Model(inputs=[input_layer], outputs=[curr_layer])
# print model.summary()
# a=ResNet50(input_shape=(112,112,3), bottom_identity_layer=True)
# print a.summary()
# a=lstm_model(input_shape,
#                resnet = False,
#                dense_feature_to_compare=1024,
#                drop_feature=0.5,
#                compare_layer=False,
#                compare_layer_dense=1024,
#                drop_compare_dense=0.5,
#                lstm_cells=[(512, False, 'concat')],
#                many_to_many=False,
#                dense_layer_after_lstm=128,
#                multi_output=False,
#                number_of_classes=12,
#                model_name='lstm_model',
#                test_mode=True)

