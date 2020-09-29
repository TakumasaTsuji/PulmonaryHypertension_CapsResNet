#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:40:14 2020

@author: tsuji
"""


import numpy as np

from PIL import Image
from skimage.transform import resize
import matplotlib 
matplotlib.use("Agg")
from matplotlib import pyplot as plt


class BaseBackprop(object):

    def __init__(self, model):
        self.model = model
        self.xp = model.xp

    def backward(self, vs_norm, label):
        vs_norm.grad = self.xp.zeros_like(vs_norm.data)

        if label == -1:
            vs_norm.grad[:, vs_norm.data.argmax()] = 1
        else:
            vs_norm.grad[:, label] = 1
        
        
        self.model.cleargrads()
        vs_norm.backward(retain_grad=True)

        return vs_norm

class GradCAM(BaseBackprop):

    def __init__(self, model):
        super(GradCAM, self).__init__(model)

    def generate(self, label, convolved_image, vs_norm):
        
        vs_norm = self.backward(vs_norm, label)        
        weights = np.mean(convolved_image.grad, axis=(2, 3))
        gcam = np.tensordot(weights[0], convolved_image.data[0], axes=(0, 0))
        gcam = np.maximum(0, gcam)

        return gcam
    
def superimpose_two_images(gcam, x):
    im1 = Image.fromarray(np.uint8(x / x.max() * 255))
    gcam = resize(gcam, im1.size, mode="reflect", order=1)
    gray_to_jet = plt.get_cmap("jet")
    gcam = gray_to_jet(np.uint8(gcam / gcam.max() * 255))
    im2 = Image.fromarray((gcam[:, :, :3]*255).astype(np.uint8))
    mask = Image.new("L", im1.size, 128) # 128 
    grad_cam_image = Image.composite(im1, im2, mask)
    return grad_cam_image


