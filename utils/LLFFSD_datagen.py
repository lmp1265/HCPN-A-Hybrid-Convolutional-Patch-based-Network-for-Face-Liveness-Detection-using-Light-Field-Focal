# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

"""
LLFFSD Dataset Generator

@author: Mupei Li
@mail : limupei2022(at)ia.ac.cn

"""
import numpy as np
import cv2
import os
from os.path import join, isdir, isfile
from os import listdir
import random
import time
from skimage.io import ImageCollection, imread, concatenate_images
from skimage import transform
import torch
from torchvision.transforms import Normalize


def CropBorder(mb, border_ratio):
    if mb == 'LT':
        s_h = 0
        s_w = 0
    elif mb == 'RT':
        s_h = 0
        s_w = border_ratio
    elif mb == 'LB':
        s_h = border_ratio
        s_w = 0
    elif mb == 'RB':
        s_h = border_ratio
        s_w = border_ratio
    elif mb == 'C':
        s_h = border_ratio // 2
        s_w = border_ratio // 2
    else:
        raise IOError('No such options.')

    return [s_h, s_w]


def ImageCollectionResize(f_in):
    im = imread(f_in)
    f_out = transform.resize(im, (224, 224))
    return f_out


def ImageCollectionNoResize(f_in):
    im = imread(f_in)
    return im


class DataGenerator():
    """
    DataGenerator Class : To generate Train, Validation and Test sets for LFLD datasets
    Formalized DATA:
        Inputs (two modalities):
            1) Focal stacks:
            Inputs have a shape of (Batchsize) X (Number of slices: 145) X (Height: 88) X (Width: 120) X (Channels: 1)
            Notice original height 96, width 128, i.e., cropped border 8 pixels
            2) Crop ROI
            Inputs have a shape of (Batchsize) X (Height: 440) X (Width: 600) X (Channels: 1)
            Notice orignial height 480, width 640, i.e., cropped border 40 pixels
        Outputs:
            Labels have a shape (Batchsize) X (Classes: 2)
            Notice Fake 0 and Real 1
    """

    def __init__(self, root_dir, c2d_ext='.png', fs_ext='.jpg'):
        """ Initializer
        Args:
            root_dir            : Directory containing everything
            train_data_file        : Text file with training set data
            remove_joints        : Joints List to keep (See documentation)
        """
        self.root_dir_ = root_dir
        assert isdir(self.root_dir_)
        self.fs_dir_ = join(self.root_dir_, 'FocalStack')
        assert isdir(self.fs_dir_)
        self.c2d_dir_ = join(self.root_dir_, 'Cropped2D')
        assert isdir(self.c2d_dir_)

        self.c2d_ext_ = c2d_ext
        self.fs_ext_ = fs_ext

    # --------------------Generator Initialization Methods ---------------------

    def _CreateSamplesTable(self):
        """ Create Table of samples from TEXT file
        """
        self.samples_table_ = []
        self.data_dict_ = {}
        print('=' * 50)
        print('Creating Samples Table.')
        # print('>>>%d samples in total.' % len(listdir(self.c2d_dir_)))
        for sample_fs in listdir(self.fs_dir_):

            if 'Bon' in sample_fs:  # 'Bon' is short for 'BonaFide'
                label = 1     # label is 0 or 1
                reallabel = 6   # real label represent specific attack type
            else:
                label = 0
                if 'Tab' in sample_fs:
                    reallabel = 0
                elif 'Pap' in sample_fs:
                    reallabel = 1
                elif 'Mb1' in sample_fs:
                    reallabel = 2
                elif 'Mb2' in sample_fs:
                    reallabel = 3
                elif 'Lap' in sample_fs:
                    reallabel = 4
                elif 'Wpa' in sample_fs:
                    reallabel = 5


            _, attrbitue, subject_id, sample_number, _ = sample_fs.split('_')

            sample_dict_name = attrbitue + '_' + subject_id + '_' + sample_number

            fs_dir = join(self.fs_dir_, 'LF_' + sample_dict_name + '_refocusedImgs')
            assert isdir(fs_dir)
            fs_imnames = join(fs_dir, '*' + self.fs_ext_)

            self.data_dict_[sample_dict_name] = {
                'attribute': attrbitue,
                'subject_id': int(subject_id),
                'sample_number': int(sample_number),
                # 'c2d_imname': c2d_imname,
                'fs_imnames': fs_imnames,
                'label': label,
                'reallabel': reallabel}

            self.samples_table_.append(sample_dict_name)

        print('=*' * 50)
        print('%d samples in total.' % len(self.samples_table_))

    def _Randomize(self):
        """ Randomize the set
        """
        random.shuffle(self.samples_table_)

    def _SplitSets(self, test_set_range, selected_attributes):
        """ Select Elements to feed training and validation set
        Args:
            validation_rate        : Percentage of validation data (in ]0,1[, don't waste time use 0.1)
        """

        self.train_set_ = []
        self.test_set_ = []

        # nrof_samples = len(self.samples_table_)
        # cutoff_sample = int(nrof_samples * cutoff_rate)
        # Here are the rules for splitting the train and test sets
        # 1. cutoff_rate
        # self.train_set_ = self.samples_table_[:(nrof_samples - cutoff_sample)]
        # self.test_set_ = self.samples_table_[(nrof_samples - cutoff_sample):]

        # 2. 5-fold cross validation, (80% for training, 20% for test)
        # LLFFSD 50 subjects, each has 2 samples
        # test_set_range = [i+1 for i in range(10)]
        # selected_attributes = ['Bon','Lap']

        for sample_name in self.samples_table_:
            sample_info = self.data_dict_[sample_name]
            subject_id = sample_info['subject_id']
            sample_attribute = sample_info['attribute']

            if not sample_attribute in selected_attributes:
                continue

            if subject_id in test_set_range:
                self.test_set_.append(sample_name)
            else:
                self.train_set_.append(sample_name)

        print('SET CREATED')
        # np.save('dataset-train-set-names', self.train_set_)
        # np.save('dataset-test-set-names', self.test_set_)
        print('--Training set :', len(self.train_set_), ' samples.')
        print('--Test set :', len(self.test_set_), ' samples.')

    def _GenerateSet(self, test_set_range, selected_attributes, rand=False):
        """ Generate the training and validation set
        Args:
            rand : (bool) True to shuffle the set
        """
        self._CreateSamplesTable()
        if rand:
            self._Randomize()
        self._SplitSets(test_set_range=test_set_range, selected_attributes=selected_attributes)

    # ----------------------- Batch Generator ----------------------------------
    def _AuxGenerator(self, batch_size=16, normalize=True, sample_set='train', debug=False):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        while True:
            fs_set = []
            c2d_set = []
            label_set = []
            i = 0
            while i < batch_size:
                if sample_set == 'train':
                    name = random.choice(self.train_set_)
                elif sample_set == 'test':
                    name = random.choice(self.test_set_)

                fs_data, c2d_im, label = self._GetSample(name)

                # interval = 1
                time_steps_last_dimension = True

                if normalize:
                    # norm_fs = fs[::interval, fs_h:fs_h+fs_cropped_size, fs_w:fs_w+fs_cropped_size]
                    fs_data = (fs_data.astype(np.float32) - 127.5) / 128.0
                    # stacked_fs = np.stack([norm_fs, norm_fs, norm_fs], axis=0)
                    # Time steps in the last dimension
                    if time_steps_last_dimension:
                        fs_data = np.transpose(fs_data, (1, 2, 3, 0))

                    # norm_roi = roi_im[roi_h:roi_h+roi_cropped_size, roi_w:roi_w+roi_cropped_size]
                    c2d_im = (c2d_im.astype(np.float32) - 127.5) / 128.0
                    # stacked_roi = np.stack([norm_roi, norm_roi, norm_roi], axis=0)

                    # stacked_roi= np.transpose(stacked_roi,(1,2,0))

                # else:
                #     cropped_fs = fs[::interval, fs_h:fs_h+fs_cropped_size, fs_w:fs_w+fs_cropped_size]
                #     stacked_fs = np.stack([cropped_fs, cropped_fs, cropped_fs], axis=0)
                #     stacked_fs= np.transpose(stacked_fs,(1,2,3,0))
                #     train_fs.append(stacked_fs.astype(np.float32))
                #     cropped_roi = roi_im[roi_h:roi_h+roi_cropped_size, roi_w:roi_w+roi_cropped_size]
                #     stacked_roi = np.stack([cropped_roi, cropped_roi, cropped_roi], axis=0)
                #     stacked_roi=np.transpose(stacked_roi,(1,2,0))
                #     train_roi.append(stacked_roi.astype(np.float32))

                fs_set.append(fs_data)
                c2d_set.append(c2d_im)
                label_set.append(label)

                i = i + 1

            if debug:
                print('=' * 50)
                print('Mini-batch shape: ',
                      np.array(fs_set).shape, ' ',
                      np.array(c2d_set).shape, ' ',
                      np.array(label_set).shape)

            yield ([np.array(fs_set), np.array(c2d_set)], np.array(label_set))

    def _Generator(self, batchSize=16, norm=True, sample='train'):
        """ Create a Sample Generator
        Args:
            batchSize     : Number of image per batch
            norm               : (bool) True to normalize the batch
            sample          : 'train'/'valid' Default: 'train'
        """
        return self._AuxGenerator(batch_size=batchSize, normalize=norm, sample_set=sample)

    def _GetData(self, sample_set, normalize=True, debug=False):
        fs_set = []
        c2d_set = []
        label_set = []

        if sample_set == 'train':
            data_set = self.train_set_
        elif sample_set == 'test':
            data_set = self.test_set_
        else:
            raise IOError('{} not supported.'.format(sample_set))

        for name in data_set:
            print(name)
            fs_data, c2d_im, label = self._GetSample(name)

            # interval = 1
            time_steps_last_dimension = True

            if normalize:
                # norm_fs = fs[::interval, fs_h:fs_h+fs_cropped_size, fs_w:fs_w+fs_cropped_size]
                fs_data = (fs_data.astype(np.float32) - 127.5) / 128.0
                # stacked_fs = np.stack([norm_fs, norm_fs, norm_fs], axis=0)
                # Time steps in the last dimension
                if time_steps_last_dimension:
                    fs_data = np.transpose(fs_data, (1, 2, 3, 0))

                # norm_roi = roi_im[roi_h:roi_h+roi_cropped_size, roi_w:roi_w+roi_cropped_size]
                c2d_im = (c2d_im.astype(np.float32) - 127.5) / 128.0
                # stacked_roi = np.stack([norm_roi, norm_roi, norm_roi], axis=0)

                # stacked_roi= np.transpose(stacked_roi,(1,2,0))

                # else:
                #     cropped_fs = fs[::interval, fs_h:fs_h+fs_cropped_size, fs_w:fs_w+fs_cropped_size]
                #     stacked_fs = np.stack([cropped_fs, cropped_fs, cropped_fs], axis=0)
                #     stacked_fs= np.transpose(stacked_fs,(1,2,3,0))
                #     train_fs.append(stacked_fs.astype(np.float32))
                #     cropped_roi = roi_im[roi_h:roi_h+roi_cropped_size, roi_w:roi_w+roi_cropped_size]
                #     stacked_roi = np.stack([cropped_roi, cropped_roi, cropped_roi], axis=0)
                #     stacked_roi=np.transpose(stacked_roi,(1,2,0))
                #     train_roi.append(stacked_roi.astype(np.float32))

            fs_set.append(fs_data)
            c2d_set.append(c2d_im)
            label_set.append(label)

        return np.array(fs_set), np.array(c2d_set), np.array(label_set)

    def _GetNoResizeSample(self, sample_name=None):
        if sample_name is not None:
            fs_img_stacks = self.data_dict_[sample_name]['fs_imnames']
            fs_data = concatenate_images(ImageCollection(fs_img_stacks, load_func=ImageCollectionNoResize))
            fs_data = fs_data / 255
            label = self.data_dict_[sample_name]['label']
            reallabel = self.data_dict_[sample_name]['reallabel']
            return fs_data, label, reallabel
        else:
            print('Specify a sample name')

    def _GetSample(self, sample_name=None):
        """
        Returns information of a sample

        """
        if sample_name is not None:
            # try:
            # start_p = self.data_dict[name]['start_p']
            # assert start_p is not None
            fs_img_stacks = self.data_dict_[sample_name]['fs_imnames']
            fs_data = concatenate_images(ImageCollection(fs_img_stacks, load_func=ImageCollectionResize))
            assert fs_data is not None

            if len(fs_data.shape) > 3:
                nrof_fs, height_fs, width_fs, channel_fs = fs_data.shape
            else:
                nrof_fs, height_fs, width_fs = fs_data.shape
            # assert fs is not None
            c2d_in = self.data_dict_[sample_name]['c2d_imname']
            c2d_im = imread(c2d_in)
            c2d_im = transform.resize(c2d_im, (224, 224))
            assert c2d_im is not None
            # roi_im = imresize(roi_im, size=0.5)
            height_c2d, width_c2d = c2d_im.shape[:2]
            assert height_c2d == height_fs and width_c2d == width_fs

            # assert roi_im is not None
            label = self.data_dict_[sample_name]['label']
            reallabel = self.data_dict_[sample_name]['reallabel']
            return fs_data, c2d_im, label, reallabel
            # except:
            # return False
        else:
            print('Specify a sample name')

    def _GetPatchSample(self, sample_name=None, patchsize=0):
        if sample_name is not None:

            fs_img_dir = self.root_dir_ + '/FocalStack/LF_' + sample_name + '_refocusedImgs/'
            img_name_list = listdir(fs_img_dir)
            img_list = []

            tempimg = imread(fs_img_dir + img_name_list[0])
            shape = np.shape(tempimg)
            x_range = shape[0]
            y_range = shape[1]
            x_pos = np.random.randint(0, x_range - patchsize + 1)
            y_pos = np.random.randint(0, y_range - patchsize + 1)

            for name in img_name_list:
                wholeimg = imread(fs_img_dir + name)
                cropimg = wholeimg[x_pos:(x_pos + patchsize), y_pos:(y_pos + patchsize), :]
                img_list.append(cropimg)

            fs_data = concatenate_images(img_list)
            # normalization
            fs_data = fs_data / 255

            label = self.data_dict_[sample_name]['label']
            reallabel = self.data_dict_[sample_name]['reallabel']
            return fs_data, label, reallabel

    def _GetDoublePatchSample(self, sample_name=None, patchsize=64, smallpatchsize=16):
        if sample_name is not None:
            fs_img_dir = self.root_dir_ + '/FocalStack/LF_' + sample_name + '_refocusedImgs/'
            img_name_list = listdir(fs_img_dir)
            img_list = []
            small_img_list = []

            tempimg = imread(fs_img_dir + img_name_list[0])
            shape = np.shape(tempimg)
            x_range = shape[0]
            y_range = shape[1]
            x_pos = np.random.randint(0, x_range - patchsize + 1)
            y_pos = np.random.randint(0, y_range - patchsize + 1)
            x_small_pos = np.random.randint(0, patchsize - smallpatchsize + 1)
            y_small_pos = np.random.randint(0, patchsize - smallpatchsize + 1)

            for name in img_name_list:
                wholeimg = imread(fs_img_dir + name)
                cropimg = wholeimg[x_pos:(x_pos + patchsize), y_pos:(y_pos + patchsize), :]
                cropsmallimg = cropimg[x_small_pos:(x_small_pos + smallpatchsize),
                               y_small_pos:(y_small_pos + smallpatchsize), :]
                img_list.append(cropimg)
                small_img_list.append(cropsmallimg)

            fs_data = concatenate_images(img_list)
            fs_data_small = concatenate_images(small_img_list)
            # normalization
            fs_data = fs_data / 256
            fs_data_small = fs_data_small / 256

            label = self.data_dict_[sample_name]['label']
            reallabel = self.data_dict_[sample_name]['reallabel']
            return fs_data, fs_data_small, label, reallabel

    def _Getname(self, sample_set='train'):
        if sample_set == 'train':
            return self.data_dict_, self.train_set_
        elif sample_set == 'test':
            return self.data_dict_, self.test_set_
        else:
            raise IOError('{} NOT supported.'.format(sample_set))