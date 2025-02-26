'''
Author: Galazxhy galazxhy@163.com
Date: 2025-02-20 21:09:59
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-02-26 19:32:07
FilePath: /GPM/Utils.py
Description: Tool functions 

Copyright (c) 2025 by Astroyd, All Rights Reserved. 
'''
import os
import torch
import numpy as np

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import Coauthor


######################################################################
#                               Data                                 #
######################################################################

'''
description: Get Split Dataset
param {Name of dataset} name
param {Number of labels for training per class} trainPerClass
param {Ratio of validation ratio} validRatio
return {Split dataset} dataset
'''
def getData(name, trainPerClass, validRatio):
    if(name == 'Cora' or name == 'Citeseer' or name == 'Pubmed'):
        dataset = Planetoid(root='./data/'+name, name=name)
        data = dataset[0]
    elif(name == 'Computers' or name == 'Photo'):
        dataset = Amazon(root='./data/'+name, name=name)
        data = random_splits(dataset[0], dataset.num_classes, trainPerClass, validRatio)
    elif(name == 'CS' or name == 'Physics'):
        dataset = Coauthor(root='./data/'+name, name=name)
        data = random_splits(dataset[0], dataset.num_classes, trainPerClass, validRatio)
    
    return data, dataset.num_node_features, dataset.num_classes


######################################################################
#                               Tools                                #
######################################################################
'''
description: Generate mask with indices
param {The indices to be masked} index
param {Size of the mask} size
return {Mask}
'''
def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

'''
description: plit Dataset With Random Seed
param {Dataset to be split} data
param {Number of classes of the dataset} num_classes
param {Number of semi-supervised train labels of each class} percls_trn
param {Ratio of validation dataset} val_lb
param {Random seed} seed
return {Split dataset}
'''
def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
    index=[i for i in range(0,data.y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    return data

'''
description: List like 'Append' tool for tensor datatype
param {append a with b} a
param {append a with b} b
return {Appending result}
'''
def ts_append(a, b):
    if a is None:
        return b.unsqueeze(0)
    else:
        return torch.cat([a, b.unsqueeze(0)], dim=0)
    
'''
description: Sigmoid function
param {Input} x
param {Stepness} alpha
param {Bias} beta
return {Result of sigmoid}
'''
def sigmoid(x, alpha, beta):
    return 1 / (1 + torch.exp(-alpha*(x-beta)))

'''
description: Soft Logical Matrix Multiplication
param {matrix A} A
param {matrix B} B
return {Logical multiplication result of A and B}
'''
def soft_logic_mm(A, B):
    # return torch.mm(A, B)
    return sigmoid(torch.mm(A, B), 1, 2)

'''
description: Save to file
param {Model to be saved ".pt file"} nets
param {Training results to be saved ".txt file"} results
'''
def saveToFile(nets, results):
    i = 0
    while(os.path.exists('./save/exp'+str(i))):
        i+=1
    for j in range(len(nets)):
        os.makedirs(f'./save/exp'+str(i)+'/rep'+str(j))
    for j in range(len(nets)):
        torch.save(nets[j], f'./save/exp'+str(i)+'/rep'+str(j)+'/model.pt')
        with open(f'./save/exp'+str(i)+'/rep'+str(j)+'/result.txt', 'w') as f:
            f.write('Train Accuracy:'+ str(results[j,0])+ '\n')
            f.write('Validation Accuracy'+ str(results[j,1])+ '\n')
            f.write('Test Accuracy:'+ str(results[j,2])+ '\n')
