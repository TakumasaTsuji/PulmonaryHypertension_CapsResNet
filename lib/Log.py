#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 10:46:23 2020

@author: tsuji
"""


import os
import csv
import json
import numpy as np
import scipy as sp
import pandas as pd




def load_path(path):
    return [os.path.join(path, i) for i in os.listdir(path)]


def load_json(filename):
    with open(filename) as f:
        log = json.load(f)
    f.close()
    return log


def load_best_lr(logdirpath):
    
    log_path_list = load_path(logdirpath)
    
    best_mean_AUC = 0.0
    lr = 0.0
    
    for log_path in log_path_list:
        log_path = os.path.join(log_path, "test", "crossvalidation_result.json")
    
        log = load_json(log_path)
    
        lr = log["lr"]
        mean_acc = log["mean_Accuracy"]
        mean_AUC = log["mean_AUC"]
        
        if mean_AUC > best_mean_AUC:
            best_mean_AUC = mean_AUC
            best_lr = lr
    print("Best learning rate : ", lr)
    print("Best mean AUC :", best_mean_AUC)
    return best_lr


def load_best_model_path(logdirpath):
    log = load_json(os.path.join(logdirpath, "log"))

    epocs, TrainAccuracy, TrainLoss, TestAccuracy, TestLoss = [], [], [], [], []
    for i in range(len(log)):
        epoch = log[i]["epoch"]
        MainAccuracy = log[i]["main/accuracy"]
        MainLoss = log[i]["main/loss"]
        ValidationMainAccuracy = log[i]["validation/main/accuracy"]
        ValidationMainLoss = log[i]["validation/main/loss"]

        epocs.append(epoch)
        TrainAccuracy.append(MainAccuracy)
        TrainLoss.append(MainLoss)
        TestAccuracy.append(ValidationMainAccuracy)
        TestLoss.append(ValidationMainLoss)
    
    best_epoch = np.argmax(TestAccuracy) + 1
    best_train_accuracy = TrainAccuracy[best_epoch - 1]
    best_validation_accuracy = TestAccuracy[best_epoch - 1]
    print("--------------------------------------------------------------------")
    print("Log path: {}".format(logdirpath))
    print("Best epoch : " + str(best_epoch))
    print("Best train accurayc      : " + str(best_train_accuracy))
    print("Best validation accuracy : " + str(best_validation_accuracy))
    print("--------------------------------------------------------------------")
    best_model_path = os.path.join(logdirpath, "model_epoch_{}.npz".format(best_epoch))
    return best_validation_accuracy, best_model_path


def load_best_model_path_in_search(pathdir):
    validation_accuracy_list, model_path_list = [], []

    search_list = load_path(pathdir)
    
    for i, path in enumerate(search_list):
        best_validation_accuracy, best_model_path = load_best_model_path(path)
        validation_accuracy_list.append(best_validation_accuracy)
        model_path_list.append(best_model_path)
        
    best_model_path_in_search_list = model_path_list[np.argmax(validation_accuracy_list)]
    return best_model_path_in_search_list
    
if __name__ == "__main__":
    pass
