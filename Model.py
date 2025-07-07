"""
Author: Galazxhy galazxhy@163.com
Date: 2025-02-21 16:16:43
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-04-17 16:05:24
FilePath: /GPM/Model.py
Description: Graph Pseudo-label Propagation Model

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import Utils


class plpLayer(nn.Module):
    """
    description: Pseudo label propagation layer
    method {Initialization} __init__()
    method {Pseudo label propagation} prop(self, adjMatrix, pLabel)
    method {Forward calculation} forward(self, adjMatrix, x)
    """

    """
    description: description
    param {Class plpLayer} self
    param {Number of input features} nFeature
    param {Number of classes} nClass
    param {Range of propagation} propRange
    param {Ensembling mode} mode
    """

    def __init__(self, nFeature, nClass, propRange, mode, alpha, beta, nlFunc):

        super(plpLayer, self).__init__()
        self.predSeq = nn.Sequential(
            nn.Linear(nFeature, 32), nn.ReLU(), nn.Linear(32, nClass)
        )
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)
        self.propRange = propRange
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.nlFunc = nlFunc

    """
    description: Propagation process
    param {Class plpLayer} self
    param {Propagation matrix} adjMatrix
    param {Pseudo labels} pLabel
    return {Propagated pseudo labels}
    """

    def prop(self, propMatrix, pLabel):
        if self.mode == "none":
            for _ in range(self.propRange):
                pLabel = getattr(Utils.nl_mm(), self.nlFunc + "_mm")(
                    propMatrix, pLabel, self.alpha, self.beta
                )
                pLabel = self.dropout(pLabel)
            return self.softmax(pLabel)
        elif self.mode == "late_fusion":
            pLabelAll = []
            pLabelRe = pLabel
            for _ in range(self.propRange):
                pLabelRe = getattr(Utils.nl_mm(), self.nlFunc + "_mm")(
                    propMatrix, pLabelRe, self.alpha, self.beta
                )
                pLabelRe = self.dropout(pLabelRe)
                pLabelAll.append(self.softmax(pLabelRe))
            return pLabelAll
        elif self.mode == "residual":
            pLabelRe = pLabel
            for _ in range(self.propRange - 1):
                pLabel = (
                    getattr(Utils.nl_mm(), self.nlFunc + "_mm")(
                        propMatrix, pLabel, self.alpha, self.beta
                    )
                    + pLabelRe
                )
                # pLabelRe = pLabel
                pLabelRe = self.dropout(pLabel)
            return self.softmax(pLabelRe)

    """
    description: Forward calculation
    param {Class plpLayer} self
    param {Propagation matrix} propMatrix
    param {Input} x
    return {Results} y
    """

    def forward(self, input):
        propMatrix, x = input[0], input[1]
        pLabel = self.softmax(self.predSeq(x))
        y = self.prop(propMatrix, pLabel)

        return y


class GPPM(nn.Module):
    """
    description: Graph propagation model
    method {Initialization} __init__()
    method {Forward calculation} forward()
    """

    """
    description: description
    param {Class GPPM} self
    param {Number of features} nFeature
    param {Number of classes} nClass
    param {Range of propagation} propRange
    param {Mode of } mode
    """

    def __init__(
        self, nFeature, nClass, propRange, alpha, beta, mode="none", nlFunc="sigmoid"
    ):
        super(GPPM, self).__init__()
        self.plpList = plpLayer(nFeature, nClass, propRange, mode, alpha, beta, nlFunc)

        self.propRange = propRange
        self.mode = mode

    """
    description: Forward calculation
    param {Class GPPM} self
    param {Node features} x
    param {Edge indices} edgeIndex
    return {Results}
    """

    def forward(self, x, edgeIndex):
        propMatrix = torch.sparse_coo_tensor(
            edgeIndex, torch.ones(edgeIndex.shape[1]).to(x.device)
        ) + torch.sparse_coo_tensor(
            [range(0, x.shape[0], 1), range(0, x.shape[0], 1)], torch.ones(x.shape[0])
        ).to(
            x.device
        )

        y = self.plpList((propMatrix, x))
        if self.mode == "late_fusion":
            y = sum(y)

        return y
