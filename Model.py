'''
Author: Galazxhy galazxhy@163.com
Date: 2025-02-21 16:16:43
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-04-14 10:32:11
FilePath: /GPM/Model.py
Description: Graph Pseudo-label Propagation Model 

Copyright (c) 2025 by Astroyd, All Rights Reserved. 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import Utils

class plpLayer(nn.Module):
    '''
    description: Pseudo label propagation layer
    method {Initialization} __init__()
    method {Pseudo label propagation} prop(self, adjMatrix, pLabel)
    method {Forward calculation} forward(self, adjMatrix, x)
    '''   
    '''
    description: description
    param {Class plpLayer} self
    param {Number of input features} nFeature
    param {Number of classes} nClass
    param {Range of propagation} propRange
    param {Ensembling mode} mode
    '''  
    def __init__(self, nFeature, nClass, propRange, mode, alpha, beta):
         
        super(plpLayer, self).__init__()
        self.predSeq = nn.Sequential(
            nn.Linear(nFeature, 32),
            nn.ReLU(),
            nn.Linear(32, nClass)
        )
        self.softmax = nn.Softmax(dim=1)
        self.propRange = propRange
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
    
    '''
    description: Propagation process
    param {Class plpLayer} self
    param {Propagation matrix} adjMatrix
    param {Pseudo labels} pLabel
    return {Propagated pseudo labels}
    '''
    def prop(self, propMatrix, pLabel):
        if self.mode != 'residual':
            pLabelRe = pLabel
            for i in range(self.propRange):
                pLabel = Utils.soft_logic_mm(propMatrix, pLabelRe, self.alpha, self.beta)
            return self.softmax(pLabel)
        else:
            pLabelAll = []
            pLabelRe = pLabel
            for i in range(self.propRange):
                pLabelRe = Utils.soft_logic_mm(propMatrix, pLabelRe, self.alpha, self.beta)
                pLabelAll.append(self.softmax(pLabelRe))
            return pLabelAll

    '''
    description: Forward calculation
    param {Class plpLayer} self
    param {Propagation matrix} propMatrix
    param {Input} x
    return {Results} y
    '''
    def forward(self, input):
        propMatrix, x = input[0], input[1]
        pLabel = self.softmax(self.predSeq(x))
        y = self.prop(propMatrix, pLabel)

        return y
    
class GPPM(nn.Module):
    '''
    description: Graph propagation model
    method {Initialization} __init__()
    method {Forward calculation} forward()
    '''
    '''
    description: description
    param {Class GPPM} self
    param {Number of features} nFeature
    param {Number of classes} nClass
    param {Range of propagation} propRange
    param {Mode of } mode
    '''    
    def __init__(self, nFeature, nClass, propRange, alpha, beta, mode='ensemble'):
        super(GPPM, self).__init__()
        if mode == 'None':
            self.plpList = nn.ModuleList([
                plpLayer(nFeature, nClass, propRange, mode, alpha, beta)
            ])
        elif mode == 'voting':
            self.plpList = nn.ModuleList()
            for i in range(propRange):
                self.plpList.append(plpLayer(nFeature, nClass, i+1, mode, alpha, beta))
        else:
            self.plpList = nn.ModuleList([
                plpLayer(nFeature, nClass, propRange, mode, alpha, beta)
            ])
        
        self.propRange = propRange
        self.mode = mode
    
    '''
    description: Forward calculation
    param {Class GPPM} self
    param {Node features} x
    param {Edge indices} edgeIndex
    return {Results}
    '''    
    def forward(self, x, edgeIndex):
        propMatrix = torch.sparse_coo_tensor(
            edgeIndex,
            torch.ones(edgeIndex.shape[1]).to(x.device)
        ) + torch.sparse_coo_tensor(
            [range(0, x.shape[0], 1), range(0, x.shape[0], 1)],
            torch.ones(x.shape[0])
        ).to(x.device)
        
        
        if self.mode == 'None':
            y = self.plpList[0]((propMatrix, x))
        elif self.mode == 'voting':
            y = None
            for i in range(self.propRange):
                y = Utils.ts_append(y, self.plpList[i]((propMatrix, x)))
            y = y.sum(dim=0)
        elif self.mode == 'residual':
            y = self.plpList[0]((propMatrix, x))
            y = sum(y)
        
        return y
            