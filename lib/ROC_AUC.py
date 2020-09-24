#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:31:52 2019

@author: tsuji
"""


import os
import csv
import numpy as np
#import seaborn as sns
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from mpl_toolkits.mplot3d import Axes3D



def open_csv(filename):
    data = []
    with open(filename, 'r') as filename:
        reader = csv.reader(filename)
        # ヘッダ行は特別扱い
        #header = next(reader)
        # 中身
        for row in reader:
            data.append(row)
    return data
    
def load_csv(path):
    data = open_csv(path)
    data = np.array(data, dtype=np.float32)
    return data



def plot_scatter(pred_score_mat, label_mat, savedir):
    n_x, n_y, r_x, r_y = [], [], [], []
    for d, dd in zip(pred_score_mat, label_mat):
        if dd[0] == 1:
            r_x.append(d[0])
            r_y.append(d[1])
        else:
            n_x.append(d[0])
            n_y.append(d[1])
            
    fig = plt.figure(figsize=(6, 6), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(r_x, r_y, s=12, marker='o', c='red',  edgecolors='red',  label='Abnormal')
    ax.scatter(n_x, n_y, s=12, marker='o', c='blue', edgecolors='blue', label='Normal')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('Abnormal', fontsize=14)
    ax.set_ylabel('Normal', fontsize=14)
    ax.legend(loc=3)
    #ax.set_adjustable()

    savepath = os.path.join(savedir, "likelihood")
    
    Savefig(savepath)
    


def ROC(label_mat, pred_score_mat, label_name="CapsuleResNet_512", line_color="darkorange", savepath="result"):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(label_mat[:, i], pred_score_mat[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])
        #roc_auc[i] = roc_auc_score(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(label_mat.ravel(), pred_score_mat.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    #Plot of a ROC curve for a specific class
    lw = 2
    plt.figure(figsize=(10, 10), facecolor="white")
    plt.plot(fpr[0], tpr[0], color= line_color,
             lw=lw, label= label_name + ' (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(loc="lower right", fontsize=13)
    plt.grid(True)
    
    Savefig(os.path.join(savepath, "ROC_AUC"))
       
    
def test_classification(sum_label, sum_pred, result_list, name_list, savedir):

    plot_scatter(sum_pred, sum_label, savedir)
    
    size = sum_pred.shape[1] 
    ConfusionMatrix = np.zeros((size, size))
    for p, l in zip(sum_pred, sum_label):
        pred = np.argmax(p)
        label = np.argmax(l)
        # i is True label 
        ConfusionMatrix[label,pred] += 1
    
    Acc = (ConfusionMatrix[0][0] + (ConfusionMatrix[1][1])) / len(sum_label) * 100
    TP  =  ConfusionMatrix[0][0] / (ConfusionMatrix[0][0] + ConfusionMatrix[0][1]) * 100 
    TN  =  ConfusionMatrix[1][1] / (ConfusionMatrix[1][0] + ConfusionMatrix[1][1]) * 100

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(sum_label[:, i], sum_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(sum_label.ravel(), sum_pred.ravel())
    #AUC = auc(fpr["micro"], tpr["micro"])
    AUC_micro = auc(fpr["micro"], tpr["micro"])
    
    AUC = auc(fpr[0], tpr[0])
    Result = [Acc, TP, TN, AUC]

    print("-----------------------------------------------")
    print(ConfusionMatrix)
    print("Accuracy : " + str(Acc) + " %")
    print("True Positive : " + str(TP) + "%")
    print("True negative : " + str(TN) + "%")
    print("Area under the curve (macro) : " + str(AUC))
    #print("Area under the curve (micro) : " + str(AUC_micro))
    print("---------------------------------")
    
    
    name_data = np.array(name_list).reshape(len(name_list), 1)
    sum_pred = np.hstack([name_data, sum_pred])
    sum_label = np.hstack([name_data, sum_label])    
    
    #save pred & label
    data = [sum_label,sum_pred, result_list]
    data_name = ["sum_label","sum_pred", "classification_result"]

   
    for i in range(len(data)):
        savepath = os.path.join(savedir, data_name[i] + ".csv")
        f = open(savepath,"w")
        for d in data[i]:    
            writer = csv.writer(f)
            writer.writerow(d)
        f.close()
    
    return Result





def Savefig(savename):    
    plt.savefig(savename + ".svg", dpi=500, facecolor="white")
    plt.savefig(savename + ".png", dpi=500, facecolor="white")
    plt.close()
    
        
if __name__=='__main__':
    path  = "/media/tsuji/Tokushima_Project/result/PCWm/hirosemachine/classification/5ensemble/Resnet50/2/0.00018456840873387327/test_newdata"
    path = "/media/tsuji/bd4e7470-cf69-40b7-af32-cd624560165b/Tokushima_Project/result/PCWm/hirosemachine/classification/5ensemble/Resnet50/2/0.00018456840873387327/test_newdata"
    pred_path = os.path.join(path, "sum_pred.csv")
    label_path = os.path.join(path, "sum_label.csv")
    
    
    sum_pred  = np.array(open_csv(pred_path))[:, 1:].astype(np.float32)
    sum_label = np.array(open_csv(label_path))[:, 1:].astype(np.float32)
    
    #ROC(sum_label, sum_pred)
    
    

    
    
    