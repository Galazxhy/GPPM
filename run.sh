###
 # @Author: Galazxhy galazxhy@163.com
 # @Date: 2025-02-24 20:24:11
 # @LastEditors: Galazxhy galazxhy@163.com
 # @LastEditTime: 2025-04-17 16:30:10
 # @FilePath: /GPM/run.sh
 # @Description: Run code script
 # 
 # Copyright (c) 2025 by Astroyd, All Rights Reserved. 
### 
#!/bin/bash
#Settings
device=cuda:0
#Data
data=Cora
#If full-supervised
val_rt=0.2
trn_per_class=10
#Training
model=GPPM
mode=residual
rep=20
epoch=500
lr=5e-3
wd=1e-4
max_prop=4
alpha=1
beta=1
#Early stopping
patience=11
delta=1e-3

python Run.py --device $device --data $data --val_rt $val_rt --trn_per_class $trn_per_class --model $model --mode $mode --rep $rep --epoch $epoch --lr $lr --wd $wd --patience $patience --delta $delta --max_prop $max_prop --alpha $alpha --beta $beta