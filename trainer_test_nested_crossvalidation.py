#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:45:22 2020

@author: tsuji
"""

import os
import numpy as np
from skimage.io import imread
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


import argparse

import chainer
from chainer import training, Variable, serializers
from chainer.training import extensions
import chainer.functions as F

from lib.MakeDatasets import MakeDatasets, split_nested_cross_validation_dataset, save_dataset, load_tokushima_dataset, load_image_preprocessing
from lib.Log import load_best_model_path
from lib.backprop import GradCAM, superimpose_two_images
from lib.ROC_AUC import ROC, test_classification

from models import CapsResNet





        
def run(args):
    model = CapsResNet.CapsResNet(cls=args.cls)
    size  = 512 
  
    #Load Dataset
    dataset = load_tokushima_dataset(
        args.Csvpath, args.DataDirPath, PAm_threshold=args.PAm_threshold)
    
    #nested 10fold cross validation 
    sub_train_dataset, sub_validation_dataset, sub_test_dataset, test_dataset = split_nested_cross_validation_dataset(
        dataset, k=args.k, l=args.l, k_folds=10, seed=args.seed)
    
    
    sub_train_image_dataset = load_image_preprocessing(sub_train_dataset, size)
    sub_validation_image_dataset = load_image_preprocessing(sub_validation_dataset, size)
    sub_test_image_dataset = load_image_preprocessing(sub_test_dataset, size)
    
    train = MakeDatasets(sub_train_image_dataset, size, gamma=True, flip=True, shift=True, rotate=True)
    validation = MakeDatasets(sub_validation_image_dataset, size, gamma=False, flip=False, shift=False, rotate=False)
    test = MakeDatasets(sub_test_image_dataset, size, gamma=False, flip=False, shift=False, rotate=False)
    
      
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize, shuffle=True)
    validation_iter = chainer.iterators.SerialIterator(validation, 4, repeat=False, shuffle=False)
    
    #Train
    save_train_path = args.out
    os.makedirs(save_train_path, exist_ok=True)
    save_dataset(sub_train_dataset, os.path.join(save_train_path, "tsub-rain_dataset.csv"))
    save_dataset(sub_validation_dataset, os.path.join(save_train_path, "sub-validation_dataset.csv"))
    save_dataset(sub_test_dataset, os.path.join(save_train_path, "sub-test_dataset.csv"))
        
    train_model(args, model, train_iter, validation_iter, save_train_path)
        
     
    
    #Test
    best_validation_accuracy, best_model_path = load_best_model_path(save_train_path)
    save_test_path = save_train_path    
    os.makedirs(save_test_path, exist_ok=True)    
    sum_pred, sum_label, classification_result, name_list = test_model(args, model, best_model_path, test, test_dataset, save_test_path)
    
    ROC(sum_label, sum_pred, label_name=args.ModelName, line_color="darkorange", savepath=save_test_path)
    Result = test_classification(sum_label, sum_pred, classification_result, name_list, savedir=save_test_path)
    
    
    
    
    
    
def train_model(args, model, train_iter, validation_iter, save_train_path):
    
    
    
    if "model" in args.PretrainedModelPath:
        serializers.load_npz(args.PretrainedModelPath, model)
        print("------------------------------")
        print("load the pretrained model")
        print(args.PretrainedModelPath)
        print("------------------------------")
    
    
    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)  # Copy the model to the GPU
    
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(1e-3 * args.lr, beta1=0.5) # alpha = 1e-3
    
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.001), "hook_dec")
    
    
    # Set up an optimizer
    
    optimizer.setup(model)

    # Set up a trainer
    snapshot_interval = 1, "epoch"
    log_interval      = 1, "epoch"
    check_interval    = 10

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    stop_trigger = chainer.training.triggers.EarlyStoppingTrigger(
        monitor='validation/main/loss', check_trigger=(check_interval, 'epoch'),
        max_trigger=(args.epoch, 'epoch'))
    trainer = training.Trainer(updater,stop_trigger, out=save_train_path)
    
    evaluator = extensions.Evaluator(validation_iter, model, device=args.gpu)
    trainer.extend(evaluator, trigger=log_interval)
    trainer.extend(extensions.LogReport(), trigger=log_interval)
    trainer.extend(extensions.snapshot_object(model, "model_epoch_{.updater.epoch}.npz"), trigger=snapshot_interval)
    
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key="epoch", file_name='loss.svg'), trigger=log_interval)
    trainer.extend(extensions.PlotReport(["main/accuracy", "validation/main/accuracy"], x_key="epoch", file_name="accuracy.svg"), trigger=log_interval)
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',\
            'main/accuracy', 'validation/main/accuracy', 'elapsed_time', 'lr']), trigger=log_interval)
    
    
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.run()
    
def test_model(args, model, best_model_path, test, test_dataset, save_test_path):
    
    
    print(best_model_path)
    serializers.load_npz(best_model_path, model)
    
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        xp = chainer.cuda.cupy
        model.to_gpu(args.gpu)
    else:
        model.to_cpu()
        xp = np
    
    
    
    HeatmapPath = os.path.join(save_test_path, "GradCAM")
    TP_Path  = os.path.join(HeatmapPath,"TP")
    TN_Path = os.path.join(HeatmapPath, "TN")
    FP_Path = os.path.join(HeatmapPath, "FP")
    FN_Path = os.path.join(HeatmapPath, "FN")
    
    
    NameList = [TP_Path, TN_Path, FP_Path, FN_Path]
    
    
    for j in range(len(NameList)):
        try:
            os.makedirs(NameList[j])
        except:
            pass
     
    
    label_mat = np.zeros((test.__len__(), args.cls))
    pred_score_mat = np.zeros((test.__len__(), args.cls))
    classification_result = []
    classification_result.append(["Patient ID", "label", "pred" ])

    name_list = []
    
    TP , FN, FP, TN = [], [], [], []
    with chainer.using_config('train', False):
        for i in range(test.__len__()):
            x, t = test.get_example(i)
              
            name = test_dataset[i][0]
            name_list.append(name)
                
            label = int(t)
            convolved_image = model.convolution(Variable(xp.asarray(x[np.newaxis])))
                
            vs_norm, vs = model.output(convolved_image)    
            vs_norm = F.softmax(vs_norm)
            predicted_vector = chainer.cuda.to_cpu(vs_norm.data[0])
            pred_score_mat[i] = predicted_vector
            label_mat[i][t] = 1
            pred = np.argmax(predicted_vector)
             
                
            """---classfier the input image---"""
            if label == 0:
                if pred == 0:
                    TP.append([name, label]) # label=0 and pred=0
                    saveheatmap = TP_Path
                else:
                    FN.append([name, label]) # label=0 and pred=1
                    saveheatmap = FN_Path                        
                        
            else:
                if pred == 1:
                    TN.append([name, label]) # label=1 and pred=1
                    saveheatmap = TN_Path
                        
                else:
                    FP.append([name, label]) # label=1 and pred=0
                    saveheatmap = FP_Path
                        
            classification_result.append([name, label, pred])
                
            """---GradCAM ---"""
            raw_image = imread(test_dataset[i][0])
            split_name = name.split("/")
            ID = split_name[-1]
            
            grad_cam = GradCAM(model)
            gcam = grad_cam.generate(label=-1, convolved_image=convolved_image, vs_norm=vs_norm)
            gcam = chainer.cuda.to_cpu(gcam)
            grad_cam_image = superimpose_two_images(gcam, raw_image)
            grad_cam_image.save(os.path.join(saveheatmap, ID))
                
   
    return  pred_score_mat, label_mat, classification_result, name_list



def main():
    parser = argparse.ArgumentParser(description='Chest_Xray_Classification')
    
    parser.add_argument("--DataDirPath", "-p", type=str, default="path",
                        help="This is path to load training data")
    parser.add_argument("--Csvpath", "-csv", type=str, default="path",
                        help="This is path to load training data")
    parser.add_argument("--lr", "-lr", type=float, default=1.0,
                        help="This value is learning rate of Adam optimization")
    parser.add_argument("--PretrainedModelPath", "-pr", type=str, default="Nan",
                        help="This is path to load the parameters of pretrained model")
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--cls', '-c', type=int, default=2,
                        help='Number of classification class')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                       help='GPU ID (negative value indicates CPU)')
    parser.add_argument("--PAm_threshold", "-PAm", type=int, default=20,
                        help="This is a value of threshold for detecting hypertension")
    parser.add_argument("--k", "-k", type=int, default=1,
                        help="Number of k in k-fold cross validation")
    parser.add_argument("--l", "-l", type=int, default=1,
                        help="Number of l in nested cross validation")
    
    args = parser.parse_args()
    
    run(args)
    
if __name__ == '__main__':
    main()
