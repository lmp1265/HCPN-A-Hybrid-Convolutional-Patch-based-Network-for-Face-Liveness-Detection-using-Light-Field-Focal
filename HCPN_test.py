# -*- coding: utf-8 -*-

"""
patch C3D double-channel CNN evaluation

@author: Mupei Li
@mail : limupei2022(at)ia.ac.cn

"""

from __future__ import absolute_import
import os
from utils.LLFFSD_datagen import DataGenerator
from utils.build_c3d import C3DModel
import numpy as np
import random
import time
from keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def CalculateAccuracy(y_true, y_pred):
    assert type(y_true) == np.ndarray
    assert type(y_pred) == np.ndarray
    return (y_true == y_pred).sum() / y_true.size

def CalculateBPCER(y_true, y_pred):
    assert type(y_true) == np.ndarray
    assert type(y_pred) == np.ndarray
    N = np.sum(y_true)
    if N == 0:
        out = 0
    # False Negatives
    FN = np.sum(y_true - y_pred * y_true)
    out = FN / N
    return out

def CalculateAPCER(y_true, y_pred):
    assert type(y_true) == np.ndarray
    assert type(y_pred) == np.ndarray
    N = np.sum(1 - y_true)
    if N == 0:
        out = 0
    # False Positives
    FP = np.sum(y_pred - y_pred * y_true)
    out = FP / N
    return out

def CalculateMetrics(y_true, y_pred):
    assert type(y_true) == np.ndarray
    assert type(y_pred) == np.ndarray
    accuracy = CalculateAccuracy(y_true=y_true, y_pred=y_pred)
    BPCER = CalculateBPCER(y_true=y_true, y_pred=y_pred)
    APCER = CalculateAPCER(y_true=y_true, y_pred=y_pred)
    ACER = (BPCER + APCER) / 2

    return accuracy, BPCER, APCER, ACER

def GetTrainTestData(patchnum):

    # train data
    print('=*' * 50)
    print('Train image are patched ...')

    i = 0
    names = train_names
    while i < len(names):
        name = names[i]
        print(i, name)

        j = 0
        while j < patchnum:
            fs_data, fs_data_small, label, reallabel = data_gen._GetDoublePatchSample(name,patchsize=64,smallpatchsize=16)
            train_img.append(fs_data)
            train_img_small.append(fs_data_small)
            train_label.append(label)
            j += 1

        i += 1

    # test data
    print('=*' * 50)
    print('Test image are loaded ...')

    i = 0
    names = test_names
    while i < len(names):
        name = names[i]
        print(i, name)

        fs_data, label, reallabel = data_gen._GetNoResizeSample(name)
        test_img.append(fs_data)
        test_label.append(label)
        test_reallabel.append(reallabel)
        i += 1

def Predict(img, c3d_model, patch_size=64, patch_small_size=16, patch_stride = 20):
    model = c3d_model
    shape = np.shape(img)
    x_range = shape[1]
    y_range = shape[2]
    x_points = [i * patch_stride for i in range(int((x_range - patch_size + 1) / patch_stride))]
    y_points = [i * patch_stride for i in range(int((y_range - patch_size + 1) / patch_stride))]

    patch_img_list = []
    small_patch_img_list = []
    for x in x_points:
        for y in y_points:
            patch_img = img[:, x:x+patch_size, y:y+patch_size, :]
            a = patch_size - patch_small_size
            b = patch_size + patch_small_size
            small_patch_img = patch_img[:, int(a/2):int(b/2), int(a/2):int(b/2), :]
            patch_img_list.append(patch_img)
            small_patch_img_list.append(small_patch_img)

    patch_img_list = np.array(patch_img_list)
    small_patch_img_list = np.array(small_patch_img_list)
    score_list = model.predict([patch_img_list, small_patch_img_list])
    score_mat = score_list.reshape((len(x_points),len(y_points)))

    zero = 0
    one = 0
    for score in score_list:
        if score > 0.5:
            one += 1
        else:
            zero += 1
    if one > zero:
        pred = 1
    else:
        pred = 0

    return pred,score_mat


data_gen = DataGenerator(root_dir='./data/LLFFSD_Refocus')
test_set_range = [i + 1 for i in range(25)]
selected_attributes = ['Bon', 'Tab', 'Lap', 'Pap', 'Mb1', 'Mb2', 'Wpa']
data_gen._GenerateSet(test_set_range=test_set_range, selected_attributes=selected_attributes, rand=True)
data_dict, train_names = data_gen._Getname(sample_set='train')
_, test_names = data_gen._Getname(sample_set='test')

train_img = []
train_img_small = []
train_label = []

test_img = []
test_label = []
test_reallabel = []
#
patchnum = 30
GetTrainTestData(patchnum=patchnum)

#model train
epochs = 5
lr = 0.01
my_c3d = C3DModel()
double_model = my_c3d.hcpn()
double_model.load_weights('./models/epoch20.h5')
double_model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr), metrics=['accuracy'])
double_model.fit(x=[train_img, train_img_small],
                 y=train_label, epochs=epochs, batch_size=8,
                 shuffle=True, validation_split=0.1)
double_model.save_weights('./models/epoch25.h5')

#model test
my_c3d = C3DModel()
double_model = my_c3d.hcpn()
double_model.load_weights('./models/epoch25.h5')

test_pred = []
for img in test_img:
    pred, score_mat = Predict(img=img, c3d_model=double_model, patch_stride=20)
    test_pred.append(pred)
accuracy, BPCER, APCER, ACER = CalculateMetrics(np.array(test_label), np.array(test_pred))
print('==================================================')
print(accuracy, BPCER, APCER, ACER)
