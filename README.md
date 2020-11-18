# Pulmonary hypertension classification using CapsResNet
* This repository contains Chainer implementation of the following paper: [Deep learning to predict elevated pulmonary artery pressure in patients with suspected pulmonary hypertension using standard chest X ray](https://www.nature.com/articles/s41598-020-76359-w)
---
[![DOI](https://zenodo.org/badge/298153172.svg)](https://zenodo.org/badge/latestdoi/298153172)
### Key Details
* Based on the work [Deep Residual Learning for Image Recognition](https://www.cvfoundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) and [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)
* Thanks to [soskek](https://github.com/soskek/dynamic_routing_between_capsules) for the [chainer](https://github.com/chainer/chainer) implementation of Capsule Network.



|Item| Details|
|---|---|
|**Input**|512 x 512 grayscale image|



---
### Dependencies
* Chainer 7.2.0
* Cupy 7.3.0
* Cuda 10.0

---
### Runining CapsResNet
```
$ bash run_crossvalidation.sh
$ bash run_nested_crossvalidation.sh
```
