#!/bin/bash

GPU=1
CLS=2


PATHDIR="/data/HDD1/Tokushima_Project/Data/XP_Tokushima_png"
CSVPATH="/home/tsuji/Dropbox/deep_learning/Tokushima_project/PAm/chainer/catheter_data/PAm_data_20200916.csv"
SAVEBASEPATH="/data/HDD1/Tokushima_Project/result/PAm/revise/RSNA/TrasnferLearning/2class/nested_crossvalidation"

#RSNA pretrained model
MODELPATH="/data/HDD1/kaggle/RSNA_PneumoniaDetection/result/chainer/Reedbush/2class/CapsuleResNet_512/model_epoch_50.npz"


BATCHSIZE=4
EPOCH=5

MODEL="CapResNet"
for k in 1  #2 3 4 5 6 7 8 9 10
do
    for l in 1  #2 3 4 5 6 7 8 9 10
    do
	for step in 0.01  #0.05 0.1 0.5 1.0
	do

	    #SAVEDIRPATH="${SAVEBASEPATH}/${MODEL}/${k}/${l}/${step}"
	    SAVEMODELPATH="${SAVEBASEPATH}/${MODEL}/${k}/${l}/${step}"
	    python trainer_test_nested_crossvalidation.py -g ${GPU} -c ${CLS} -m ${MODEL} -pr ${MODELPATH} -b ${BATCHSIZE} -lr ${step} -e ${EPOCH} -o ${SAVEMODELPATH} -p ${PATHDIR} -csv ${CSVPATH} -k ${k} -l${l} 
	    rm ${SAVEMODELPATH}/model_epoch*.npz
	    #mv ${TrainLog} ${SAVEMODELPATH}
	done
    done    
done
