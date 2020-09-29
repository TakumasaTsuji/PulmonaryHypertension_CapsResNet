#!/bin/bash

GPU=0
CLS=2

PATHDIR="./Chest_Xray_Dataset"
CSVPATH="./Catheter_data.csv"
SAVEBASEPATH="./result"

#RSNA pretrained model                                                                                                                                                                                      
MODELPATH="./RSNA_pretrained_model.npz"
BATCHSIZE=32
EPOCH=100

MODEL="CapResNet"
for k in 1 2 3 4 5 6 7 8 9 10
do
    for l in 1 2 3 4 5 6 7 8 9 10
    do
	for step in 0.01 0.05 0.1 0.5 1.0
	do

	    SAVEMODELPATH="${SAVEBASEPATH}/${MODEL}/${k}/${l}/${step}"
	    python trainer_test_nested_crossvalidation.py -g ${GPU} -c ${CLS} -pr ${MODELPATH} -b ${BATCHSIZE} -lr ${step} -e ${EPOCH} -o ${SAVEMODELPATH} -p ${PATHDIR} -csv ${CSVPATH} -k ${k} -l${l} 
	    rm ${SAVEMODELPATH}/model_epoch*.npz
	done
    done    
done
