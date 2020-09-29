#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:31:38 2019

@author: tsuji
"""
import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import initializers

from skimage.transform import resize


def _count_params(m, n_grids=6):
    print('# of params', sum(param.size for param in m.params()))
    # The number of parameters in the paper (11.36M) might be
    # of the model with unshared matrices over primary capsules in a same grid
    # when input data are 36x36 images of MultiMNIST (n_grids = 10).
    # Our model with n_grids=10 has 11.349008M parameters.
    # (In the Sec. 4, the paper says "each capsule in the [6, 6] grid
    # is sharing their weights with each other.")
    print('# of params if unshared',
          sum(param.size for param in m.params()) +
          sum(param.size for param in m.Ws.params()) *
          (n_grids * n_grids - 1))


def squash(ss):
    ss_norm2 = F.sum(ss ** 2, axis=1, keepdims=True)
    """
    # ss_norm2 = F.broadcast_to(ss_norm2, ss.shape)
    # vs = ss_norm2 / (1. + ss_norm2) * ss / F.sqrt(ss_norm2): naive
    """
    norm_div_1pnorm2 = F.sqrt(ss_norm2) / (1. + ss_norm2)
    norm_div_1pnorm2 = F.broadcast_to(norm_div_1pnorm2, ss.shape)
    vs = norm_div_1pnorm2 * ss  # :efficient
    # (batchsize, 16, 10)
    return vs


def get_norm(vs):
    return F.sqrt(F.sum(vs ** 2, axis=1))

    

class ResBlock(chainer.Chain):
    def __init__(self, ch,activation=F.relu):
        self.activation = activation
        layers = {}
        layers['c0'] = L.Convolution2D(ch, ch, 3, 1, 1)
        layers['c1'] = L.Convolution2D(ch, ch, 3, 1, 1)
        layers['bn0'] = L.BatchNormalization(ch)
        layers['bn1'] = L.BatchNormalization(ch)
        super(ResBlock, self).__init__(**layers)

    def __call__(self, x):
        h = self.bn0(x)
        h = self.activation(h)
        h = self.c0(h)
        h = self.bn1(h)
        h = self.activation(h)
        h = self.c1(h)
        return h + x

        
        
        
        
        
init = chainer.initializers.Uniform(scale=0.05)


class CapsResNet(chainer.Chain):

    def __init__(self, cls=2):
        super(CapsResNet, self).__init__()
        self.n_iterations = 3  # dynamic routing
        self.n_grids = 6  # grid width of primary capsules layer
        self.n_raw_grids = self.n_grids
        self.cls = cls
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(  1, 256, ksize=4,  stride=2, pad=1, initialW=init)
            self.conv2_1 = L.Convolution2D(256, 128, ksize=4,  stride=2, pad=1, initialW=init)
            self.conv3_1 = L.Convolution2D(128,  64, ksize=4,  stride=2, pad=1, initialW=init)
            self.conv4_1 = L.Convolution2D(  64, 64, ksize=4,  stride=2, pad=1, initialW=init)
            self.conv5_1 = L.Convolution2D( 64, 128, ksize=4,  stride=2, pad=1, initialW=init)
            self.conv6_1 = L.Convolution2D(128, 256, ksize=5,  stride=2, pad=0, initialW=init)
            
            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(128)
            self.bn3 = L.BatchNormalization(64)
            self.bn4 = L.BatchNormalization(64)
            self.bn5 = L.BatchNormalization(128)
            self.bn6 = L.BatchNormalization(256)
            
            self.conv1_2 = ResBlock(256, activation=F.relu)
            self.conv2_2 = ResBlock(128, activation=F.relu)
            self.conv3_2 = ResBlock( 64, activation=F.relu)
            self.conv4_2 = ResBlock( 64, activation=F.relu)
            self.conv5_2 = ResBlock(128, activation=F.relu)
            self.conv6_2 = ResBlock(256, activation=F.relu)
          
            self.Ws = chainer.ChainList(
                *[L.Convolution2D(8, 16 * cls, ksize=1, stride=1, initialW=init)
                  for i in range(32)])
            


        _count_params(self, n_grids=self.n_grids)
        self.results = {'N': 0., 'loss': [], 'correct': [],
                        'cls_loss': [], 'rcn_loss': []}

    def pop_results(self):
        merge = dict()
        merge['mean_loss'] = sum(self.results['loss']) / self.results['N']
        merge['cls_loss'] = sum(self.results['cls_loss']) / self.results['N']
        merge['rcn_loss'] = sum(self.results['rcn_loss']) / self.results['N']
        merge['accuracy'] = sum(self.results['correct']) / self.results['N']
        self.results = {'N': 0., 'loss': [], 'correct': [],
                        'cls_loss': [], 'rcn_loss': []}
        return merge

    def __call__(self, x, t):
        convolved_image = self.convolution(x)
        vs_norm, vs = self.output(convolved_image)
        self.loss = self.calculate_loss(vs_norm, t, vs, x)

        self.results['loss'].append(self.loss.data * t.shape[0])
        self.results['correct'].append(self.calculate_correct(vs_norm, t))
        self.results['N'] += t.shape[0]
        self.accuracy = sum(self.results['correct']) / self.results['N']
              
        mean_loss = self.loss / self.results["N"]
        acc  = self.accuracy
        chainer.report({'loss': mean_loss, 'accuracy': acc}, self)

        return self.loss
    
    def convolution(self, x):
        h = self.conv1_2(F.relu(self.bn1(self.conv1_1(x))))
        h = self.conv2_2(F.relu(self.bn2(self.conv2_1(h))))
        h = self.conv3_2(F.relu(self.bn3(self.conv3_1(h))))
        h = self.conv4_2(F.relu(self.bn4(self.conv4_1(h))))
        h = self.conv5_2(F.relu(self.bn5(self.conv5_1(h))))
        h = self.conv6_2(F.relu(self.bn6(self.conv6_1(h))))
        
        return h
        
    def output(self, x):
        
        cls = self.cls
        batchsize = x.shape[0]
        n_iters = self.n_iterations
        gg = self.n_grids * self.n_grids
        
        pr_caps = F.split_axis(x, 32, axis=1)
       

        Preds = []
        for i in range(32):
            pred = self.Ws[i](pr_caps[i])
            Pred = pred.reshape((batchsize, 16, cls, gg))
            Preds.append(Pred)
        Preds = F.stack(Preds, axis=3)
        assert(Preds.shape == (batchsize, 16, cls, 32, gg))

        bs = self.xp.zeros((batchsize, cls, 32, gg), dtype='f')
        for i_iter in range(n_iters):
            cs = F.softmax(bs, axis=1)
            Cs = F.broadcast_to(cs[:, None], Preds.shape)
            assert(Cs.shape == (batchsize, 16, cls, 32, gg))
            ss = F.sum(Cs * Preds, axis=(3, 4))
            vs = squash(ss)
            assert(vs.shape == (batchsize, 16, cls))

            if i_iter != n_iters - 1:
                Vs = F.broadcast_to(vs[:, :, :, None, None], Preds.shape)
                assert(Vs.shape == (batchsize, 16, cls, 32, gg))
                bs = bs + F.sum(Vs * Preds, axis=1)
                assert(bs.shape == (batchsize, cls, 32, gg))

        vs_norm = get_norm(vs)
        return vs_norm, vs

   
       

    def calculate_loss(self, vs_norm, t, vs, x):
        class_loss = self.calculate_classification_loss(vs_norm, t)
        
        self.results['cls_loss'].append(class_loss.data * t.shape[0])
        return class_loss

    def calculate_classification_loss(self, vs_norm, t):
        xp = self.xp
        batchsize = t.shape[0]
        I = xp.arange(batchsize)
        T = xp.zeros(vs_norm.shape, dtype='f')
        T[I, t] = 1.
        m = xp.full(vs_norm.shape, 0.1, dtype='f')
        m[I, t] = 0.9

        loss = T * F.relu(m - vs_norm) ** 2 + \
            0.5 * (1. - T) * F.relu(vs_norm - m) ** 2
        return F.sum(loss) / batchsize 

    
  
        def calculate_correct(self, v, t):
        return (self.xp.argmax(v.data, axis=1) == t).sum()

        
        
if __name__=='__main__':
    import numpy as np
    import time
    model = CapsResNet(cls=2)
    
    size = 512
    batch_size = 1
    
    x = np.ones((batch_size,1, size, size)).astype(np.float32)
    x = x.reshape(1,1, size, size)
    t = np.ones((1,1)).astype(np.int32)
    
    s = time.time()
    convolved_image = model.convolution(x)
    vs_norm, vs = model.output(convolved_image)
   
    e = time.time()
    print(e-s, "sec")
    
