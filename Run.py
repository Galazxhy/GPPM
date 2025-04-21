"""
Author: Galazxhy galazxhy@163.com
Date: 2025-02-21 18:33:07
LastEditors: Galazxhy galazxhy@163.com
LastEditTime: 2025-04-14 10:32:04
FilePath: /GPM/Run.py
Description: Code Running Script

Copyright (c) 2025 by Astroyd, All Rights Reserved.
"""

import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import Utils
import Model

"""
description: Run code
param {Running parameters} args
"""


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("-------------- Loading Data --------------")
    data, nFeatures, nClasses = Utils.getData(
        args.data, args.trn_per_class, args.val_rt
    )
    data = data.to(device)
    earlyStopping = Utils.EarlyStopping(patience=args.patience, delta=args.delta)
    bestNets = []
    bestResults = []

    for i in range(args.rep):
        print(f"Repetition {i}")
        print("------------ Initializing model -------------")
        earlyStopping.reset()
        best = [0, 0, 0, 0]
        bestNet = None
        net = Model.GPPM(
            nFeatures, nClasses, args.max_prop, args.alpha, args.beta, args.mode
        ).to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
        print(f"---------------- Epoch {i} -------------------")
        tbar = tqdm.tqdm(total=args.epoch, position=0, leave=True)

        for j in range(args.epoch):
            net.train()
            tbar.set_description(f"Model {args.model} training on {args.data}")
            optimizer.zero_grad()
            out = net(data.x.to(device), data.edge_index.to(device))
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            net.eval()
            logits = net(data.x.to(device), data.edge_index.to(device))
            val_loss = F.cross_entropy(logits[data.val_mask], data.y[data.val_mask])
            accs = []
            for _, mask in data("train_mask", "val_mask"):
                pred = logits[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                accs.append(acc)
            pred = logits[data.test_mask].max(1)[1]
            acc = (
                pred.eq(data.y[data.test_mask]).sum().item()
                / data.test_mask.sum().item()
            )
            accs.append(acc)
            earlyStopping(val_loss, accs)
            if earlyStopping.earlyStop:
                best = accs
                bestNet = net
                print(f"Early stopping at epoch {j}")
                break
            tbar.set_postfix(
                Loss="{:.5f}".format(loss.item()),
                trainAcc="{:.5f}".format(accs[0]),
                valAcc="{:.5f}".format(accs[1]),
                testAcc="{:.5f}".format(accs[2]),
            )
            tbar.update()
        bestNets.append(bestNet)
        bestResults.append(best)
    bestResults = np.array(bestResults)
    Utils.saveToFile(bestNets, np.array(bestResults))

    print(
        "Accuracy:",
        bestResults[:, 2],
        "\nAverage",
        sum(bestResults[:, 2]) / len(bestResults[:, 2]),
        "\nStd",
        np.std(np.array(bestResults[:, 1]), ddof=1),
    )


"""
description: Main function
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Parameters")
    # env
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device")

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="Cora",
        choices=["Cora", "Citeseer", "Pubmed", "Computers", "Photo", "CS", "Physics"],
        help="Dataset",
    )
    parser.add_argument(
        "--val_rt",
        type=float,
        default="0.2",
        help="validation ratio of splitting dataset",
    )
    parser.add_argument(
        "--trn_per_class",
        type=int,
        default="10",
        help="Number of training labels per class",
    )

    # Training
    parser.add_argument(
        "--model", type=str, default="GPPM", choices=["GPPM"], help="Ensemble mode"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="voting",
        choices=["None", "voting", "residual"],
        help="Ensemble mode",
    )
    parser.add_argument("--rep", type=int, default=20, help="Repetition times")
    parser.add_argument("--epoch", type=int, default=500, help="Training epoch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--max_prop",
        type=int,
        default=4,
        help="Maximum propagation range (Not supporting customized range like (1, 3, 5))",
    )
    parser.add_argument(
        "--alpha", type=float, default=1, help="Alpha for sigmoid function"
    )
    parser.add_argument(
        "--beta", type=float, default=2, help="Beta for sigmoid function"
    )

    # Early stopping
    parser.add_argument(
        "--patience", type=int, default=20, help="Patience of early stopping"
    )
    parser.add_argument(
        "--delta", type=float, default=0, help="Minimum change of early stopping"
    )

    args = parser.parse_args()
    run(args)
