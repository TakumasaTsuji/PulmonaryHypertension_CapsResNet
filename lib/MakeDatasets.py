#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:20:09 2020

@author: tsuji
"""
import os
import csv
import chainer
import math
import numpy as np
from skimage.io import imread
from skimage.transform import rotate, resize
from chainer.datasets import get_cross_validation_datasets_random


def load_csv(filename):
    data = []
    with open(filename, 'r', encoding="utf-8-sig") as filename:
        reader = csv.reader(filename)
        header = next(reader)
        for row in reader:
            data.append(row)
    return data, header

def save_dataset(dataset, savepath):
    with open(savepath, "w") as f:
        for i in dataset:
            writer = csv.writer(f)
            writer.writerow(i)
    
def gamma_correction(image):
    gammalist = [0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]
    np.random.shuffle(gammalist)
    gamma = gammalist[0]
    image = 255.0 * (image/255.0)**gamma
    image = image / image.max()
    return image.astype(image.dtype)
    
def pixel_shift(x, ratio):
    MAX_SHIFT = 2 * ratio
    h, w = x.shape
    h_shift, w_shift = np.random.randint(-MAX_SHIFT, MAX_SHIFT + 1, size=2)
    a_h_sl = slice(max(0, h_shift), h_shift + h)
    a_w_sl = slice(max(0, w_shift), w_shift + w)
    x_h_sl = slice(max(0, - h_shift), - h_shift + h)
    x_w_sl = slice(max(0, - w_shift), - w_shift + w)
    a = np.zeros(x.shape)
    a[a_h_sl, a_w_sl] = x[x_h_sl, x_w_sl]
    return a.astype(x.dtype)


def flip_image(image):
    random = np.random.randint(0, 2)
    
    if random == 0:
        image = image[:, ::-1]
    
    return image
    
def rotate_processnig(image):
    angle = np.random.randint(-5, 5) * 4 # -20~20 
    image = rotate(image, angle=angle, resize=False)
    return image



def load_tokushima_dataset(csv_path, dir_path, PAm_threshold, remove_list="sample"):
    abnormal, normal = [], []
    
    csv_list, header = load_csv(csv_path)
    
    patient_index = header.index("No.")
    PAm_index = header.index("PAm")
    
    
    
    for i, data in enumerate(csv_list):
        patient_id = data[patient_index]
        PAm        = int(data[PAm_index]) 
        
        path = os.path.join(dir_path, "Xp_" + patient_id + ".png")
        
        
        if PAm >= PAm_threshold:
            abnormal.append([path, 0])
        else:
            normal.append([path, 1])
        
        
        dataset = {"Abnormal" : abnormal, "Normal": normal}
    return dataset

            
    
def split_dataset(dataset, val_num, test_num, seed=0):   
    
    normal     = dataset["Normal"]
    abnormal   = dataset["Abnormal"]
    train, validation, test = [], [], []

    np.random.seed(seed)
    np.random.shuffle(normal)
    np.random.shuffle(abnormal)
    dataset_list = [abnormal, normal]

    for i, data in enumerate(dataset_list):
        test.extend(data[: test_num])
        del data[: test_num]
        
        np.random.shuffle(data)
        
        validation.extend(data[: val_num])
        train.extend(data[val_num : ])
        
    return train , validation, test

def split_cross_validation_dataset(dataset, k=1, k_folds=10, seed=0):
    
    
    normal   = dataset["Normal"]   #461
    abnormal = dataset["Abnormal"] #439
    train, validation, test = [], [], []
    
    np.random.seed(seed)
    np.random.shuffle(normal)
    np.random.shuffle(abnormal)

    test_normal_num   = math.ceil((len(normal)   / k_folds) * 1)
    test_abnormal_num = math.ceil((len(abnormal) / k_folds) * 1) 

    k = k - 1
    test_normal = normal[test_normal_num*k : test_normal_num*(k+1)]
    test_abnormal = abnormal[test_abnormal_num*k : test_abnormal_num*(k+1)]
    test.extend(test_normal)
    test.extend(test_abnormal)
    
    del normal[test_normal_num*k : test_normal_num*(k+1)]
    del abnormal[test_abnormal_num*k : test_abnormal_num*(k+1)]
    
    np.random.seed()
    np.random.shuffle(normal)
    np.random.shuffle(abnormal)

    if len(normal) <= len(abnormal):
       val_num = math.ceil((len(normal) / k_folds) * 2)
    else:
        val_num = math.ceil((len(abnormal) / k_folds) * 2)
    
    val_normal   = normal[: val_num]
    val_abnormal = abnormal[: val_num]
    
    validation.extend(val_normal)
    validation.extend(val_abnormal)
    
    train_normal   = normal[val_num : ]
    train_abnormal = abnormal[val_num : ] 
    train.extend(train_normal)
    train.extend(train_abnormal)
    
    print("{}-fold cross validation".format(k_folds))
    print("k is {} in this training".format(k))
    
    return train, validation, test

   

def split_nested_cross_validation_dataset(dataset, k=1, l=1, k_folds=10, seed=0):
    
    normal   = dataset["Normal"]   #461
    abnormal = dataset["Abnormal"] #439
    train, validation, test = [], [], []
    sub_train, sub_validation, sub_test = [], [], []
    
    np.random.seed(seed)
    np.random.shuffle(normal)
    np.random.shuffle(abnormal)

    test_normal_num   = math.ceil((len(normal)   / k_folds) * 1)
    test_abnormal_num = math.ceil((len(abnormal) / k_folds) * 1) 

    k = k - 1
    test_normal = normal[test_normal_num*k : test_normal_num*(k+1)]
    test_abnormal = abnormal[test_abnormal_num*k : test_abnormal_num*(k+1)]
    test.extend(test_normal)
    test.extend(test_abnormal)
    
    
    del normal[test_normal_num*k : test_normal_num*(k+1)]
    del abnormal[test_abnormal_num*k : test_abnormal_num*(k+1)]
    
    sub_test_normal_num = math.ceil((len(normal)     / k_folds) * 1)
    sub_test_abnormal_num = math.ceil((len(abnormal) / k_folds) * 1)
    
    l = l - 1
    sub_test_normal = normal[sub_test_normal_num*l : sub_test_normal_num*(l+1)]
    sub_test_abnormal = abnormal[sub_test_abnormal_num*l : sub_test_abnormal_num*(l+1)] 
    sub_test.extend(sub_test_normal)
    sub_test.extend(sub_test_abnormal)

    del normal[sub_test_normal_num*l : sub_test_normal_num*(l+1)]
    del abnormal[sub_test_abnormal_num*l : sub_test_abnormal_num*(l+1)] 

    np.random.seed()
    np.random.shuffle(normal)
    np.random.shuffle(abnormal)

    if len(normal) <= len(abnormal): 
        sub_val_num = math.ceil((len(normal) / k_folds) * 2)
    else:
        sub_val_num = math.ceil((len(abnormal) / k_folds) * 2)
    
    sub_val_normal   = normal[: sub_val_num]
    sub_val_abnormal = abnormal[: sub_val_num]
    
    sub_validation.extend(sub_val_normal)
    sub_validation.extend(sub_val_abnormal)
    
    sub_train_normal   = normal[sub_val_num : ]
    sub_train_abnormal = abnormal[sub_val_num : ] 
    sub_train.extend(sub_train_normal)
    sub_train.extend(sub_train_abnormal)
    
    print("{}-fold cross validation".format(k_folds))
    print("k is {} in this training".format(k+1))
    print("l is {} in this training".format(l+1))
    
    return sub_train, sub_validation, sub_test, test


def load_image_preprocessing(image_label_dataset, size):
    dataset = []
    
    for i, image_label in enumerate(image_label_dataset):
        
        path  = image_label[0]
        label = image_label[1]
        
        image = imread(path, as_gray=True)
        image = resize(image, (size, size), order=1, mode="reflect")
        image = image / image.max()
        
        dataset.append([image, label])
    return dataset




class MakeDatasets(chainer.dataset.DatasetMixin):
    
    def __init__(self,datasets,size, gamma, flip, shift, rotate):        
        self.datasets = datasets
        self.size     = size
        self.gamma    = gamma
        self.flip     = flip
        self.shift    = shift
        self.rotate   = rotate
        self.ratio    = round(size / 28) # the ratio of the input size to the size of mnist image
        
    def __len__(self):
        return len(self.datasets)
        
    def augmentation(self, image):
        if self.gamma == True:
            image = gamma_correction(image)
        
        if self.flip == True:
            image = flip_image(image)
        
        if self.shift == True:
            image = pixel_shift(image, self.ratio)
            
        if self.rotate == True:
            image = rotate_processnig(image)
            
        return image
            
    def get_example(self, i):     
        image = self.datasets[i][0]
        image = self.augmentation(image) #Augmentation
        image = image[np.newaxis, :, :]      
        image = image.astype(np.float32)
        
        label = np.array(self.datasets[i][1]).astype(np.int32)        
        
        return image, label


#----------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    import time
    from matplotlib import pyplot as plt
    start = time.time()  
    
    PAm_threshold = 20
    size = 512
    
    csv_path = "./Cateter_data.csv"
    dir_path = "./Chest_Xray_dataset"

    dataset = load_tokushima_dataset(csv_path, dir_path, PAm_threshold=PAm_threshold)
        
    #k-fold cross validation    
    train_dataset, validation_dataset, test_dataset = split_cross_validation_dataset(
        dataset, k=1, k_folds=10, seed=0)
    
    #nested k-fold cross validation
    sub_train_dataset, sub_validation_dataset, sub_test_dataset, test_dataset = split_nested_cross_validation_dataset(
        dataset, k=1, l=1, k_folds=10, seed=0)
    
    
    train_image_dataset = load_image_preprocessing(sub_train_dataset, size)
    validation_image_dataset = load_image_preprocessing(sub_validation_dataset, size)
    test_image_dataset = load_image_preprocessing(sub_test_dataset, size)
    
    
    train = MakeDatasets(train_image_dataset, size, gamma=True, flip=True, shift=True, rotate=True)
    validation = MakeDatasets(validation_image_dataset, size, gamma=False, flip=False, shift=False, rotate=False)
    test = MakeDatasets(test_image_dataset, size, gamma=False, flip=False, shift=False, rotate=False)
    
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("------------------------------------------------------------------")
    print("sub-training datasets   : " + str(len(sub_train_dataset)))
    print("sub-validation datasets : " + str(len(sub_validation_dataset)))
    print("sub-test datasets       : " + str(len(sub_test_dataset)))
    print("test datasets           : " + str(len(test_dataset)))
    print("-------------------------------------------------------------------")
    
   
            
