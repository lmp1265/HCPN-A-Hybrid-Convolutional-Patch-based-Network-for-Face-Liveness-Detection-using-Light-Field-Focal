# -*- coding: utf-8 -*-

"""
build 3D CNN networks

@author: Mupei Li
@mail : limupei2022(at)ia.ac.cn

"""

import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Convolution3D, MaxPooling3D, ZeroPadding3D, GlobalAveragePooling3D, \
Dense, Dropout, Flatten, concatenate, add

class C3DModel(Model):
    def __init__(self):
        super().__init__()

    def c3d(self, l_size = 32, l_len = 21, summary=False):
        model = Sequential()
        # 1st layer group

        model.add(Convolution3D(64, (3, 3, 3), activation='relu',
                                    padding='same', name='c3d_conv1',
                                    input_shape=(l_len, l_size, l_size, 3)))

        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),  # change
                                   padding='valid', name='c3d_pool1'))

        # 2nd layer group
        model.add(Convolution3D(128, (3, 3, 3), activation='relu',
                                    padding='same', name='c3d_conv2', ))

        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),  ##change
                                   padding='valid', name='c3d_pool2'))

        # 3rd layer group
        model.add(Convolution3D(256, (3, 3, 3), activation='relu',
                                    padding='same', name='c3d_conv3', ))

        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                                   padding='valid', name='c3d_pool3'))

        # 4th layer group
        model.add(Convolution3D(512, (3, 3, 3), activation='relu',
                                    padding='same', name='c3d_conv4', ))

        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                                   padding='valid', name='c3d_pool4'))

        model.add(GlobalAveragePooling3D())

        # FC layers group
        # c3d_model.add(Dense(512, activation='relu', name='c3d_fc5'))
        # c3d_model.add(Dropout(.5))
        # c3d_model.add(Dense(1, activation='sigmoid', name='c3d_out'))

        if summary:
            print(model.summary())

        return model

    def c3d_resnet(self, l_size = 32, l_len = 21, summary=False):
        # 1st 2nd block
        input = Input(shape=(l_len, l_size, l_size, 3))
        x1 = Convolution3D(64, (3, 3, 3), activation='relu', padding='same', name='conv1_1')(input)
        x1 = Convolution3D(128, (3, 3, 3), activation='relu', padding='same', name='conv1_2')(x1)
        x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),  # change
                         padding='valid', name='pooling1')(x1)
        shortcut = Convolution3D(128, (2, 2, 2), strides=(2, 2, 2),
                                 activation='relu', padding='valid', name='conv1_res')(input)
        x = add(inputs=[x1, shortcut])

        # 3rd 4th block
        x1 = Convolution3D(256, (3, 3, 3), activation='relu', padding='same', name='conv2_1')(x)
        x1 = Convolution3D(512, (3, 3, 3), activation='relu', padding='same', name='conv2_2')(x1)
        x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),  # change
                          padding='valid', name='pooling2')(x1)
        shortcut = Convolution3D(512, (2, 2, 2), strides=(2, 2, 2),
                                 activation='relu', padding='valid', name='conv2_res')(x)
        x = add(inputs=[x1, shortcut])
        #pooling
        x = GlobalAveragePooling3D()(x)

        model = Model(inputs=input, outputs=x)
        if summary:
            print(model.summary())
        return model

    def c2_1d(self, l_size = 64, l_len = 21, summary=False):
        model = Sequential()
        # 1st layer group

        model.add(Convolution3D(64, (1, 3, 3), activation='relu',
                                    padding='same', name='c3d_conv1_2d',
                                    input_shape=(l_len, l_size, l_size, 3)))
        model.add(Convolution3D(64, (3, 1, 1), activation='relu',
                                    padding='same', name='c3d_conv1_1d', ))

        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),  # change
                                   padding='valid', name='c3d_pool1'))

        # 2nd layer group
        model.add(Convolution3D(128, (1, 3, 3), activation='relu',
                                    padding='same', name='c3d_conv2_2d', ))
        model.add(Convolution3D(128, (3, 1, 1), activation='relu',
                                    padding='same', name='c3d_conv2_1d', ))

        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),  ##change
                                   padding='valid', name='c3d_pool2'))

        # 3rd layer group
        model.add(Convolution3D(256, (1, 3, 3), activation='relu',
                                    padding='same', name='c3d_conv3_2d', ))
        model.add(Convolution3D(256, (3, 1, 1), activation='relu',
                                    padding='same', name='c3d_conv3_1d', ))

        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                                   padding='valid', name='c3d_pool3'))

        # 4th layer group
        model.add(Convolution3D(512, (1, 3, 3), activation='relu',
                                    padding='same', name='c3d_conv4_2d', ))
        model.add(Convolution3D(512, (3, 1, 1), activation='relu',
                                    padding='same', name='c3d_conv4_1d', ))

        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                                   padding='valid', name='c3d_pool4'))

        model.add(GlobalAveragePooling3D())

        if summary:
            print(model.summary())

        return model

    def c2_1d_resnet(self, l_size = 32, l_len = 21, summary=False):
        # 1st block
        input = Input(shape=(l_len, l_size, l_size, 3))
        x1 = Convolution3D(64, (1, 3, 3), activation='relu', padding='same', name='conv1_2d')(input)
        x1 = Convolution3D(64, (3, 1, 1), activation='relu', padding='same', name='conv1_1d')(x1)
        x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pooling1')(x1)    #change
        shortcut = Convolution3D(64, (2, 2, 2), strides=(2, 2, 2),
                                 activation='relu', padding='valid', name='conv1_res')(input)
        x = add(inputs=[x1, shortcut])

        # 2nd block
        x1 = Convolution3D(128, (1, 3, 3), activation='relu', padding='same', name='conv2_2d')(x)
        x1 = Convolution3D(128, (3, 1, 1), activation='relu', padding='same', name='conv2_1d')(x1)
        x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pooling2')(x1)
        shortcut = Convolution3D(128, (2, 2, 2), strides=(2, 2, 2),
                                 activation='relu', padding='valid', name='conv2_res')(x)
        x = add(inputs=[x1, shortcut])

        # 3rd block
        x1 = Convolution3D(256, (1, 3, 3), activation='relu', padding='same', name='conv3_2d')(x)
        x1 = Convolution3D(256, (3, 1, 1), activation='relu', padding='same', name='conv3_1d')(x1)
        x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pooling3')(x1)
        shortcut = Convolution3D(256, (2, 2, 2), strides=(2, 2, 2),
                                 activation='relu', padding='valid', name='conv3_res')(x)
        x = add(inputs=[x1, shortcut])


        # 4th block
        x1 = Convolution3D(512, (1, 3, 3), activation='relu', padding='same', name='conv4_2d')(x)
        x1 = Convolution3D(512, (3, 1, 1), activation='relu', padding='same', name='conv4_1d')(x1)
        x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                         padding='valid', name='pooling4')(x1)
        shortcut = Convolution3D(512, (2, 2, 2), strides=(2, 2, 2),
                                 activation='relu', padding='valid', name='conv4_res')(x)
        x = add(inputs=[x1, shortcut])


        #pooling
        x = GlobalAveragePooling3D()(x)

        model = Model(inputs=input, outputs=x)
        if summary:
            print(model.summary())
        return model


    def hcpn(self, l_size_large=64, l_size_small=16, l_len=21, summary=True):
        large_model = self.c2_1d_resnet(l_size=l_size_large, l_len=l_len)
        small_model = self.c3d_resnet(l_size=l_size_small, l_len=l_len)

        large_input = Input(shape=(l_len, l_size_large, l_size_large, 3), name='large_input')
        small_input = Input(shape=(l_len, l_size_small, l_size_small, 3), name='small_input')
        large_output = large_model(large_input)
        small_output = small_model(small_input)
        x = concatenate([large_output, small_output])
        x = Dense(64, activation='relu')(x)
        x = Dropout(.5)(x)
        x = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[large_input, small_input], outputs=x)
        if summary:
            print(model.summary())
        return model
