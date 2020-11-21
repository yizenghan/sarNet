import os
import pandas as pd
import numpy as np

path1_1 = '/home/hanyz/code/sarNet/log/cifar100/sar_resnet32x2_alphaBase_cifar/_round1_optimRate_g4_a2b2_s2/t0_1_0_tLast0_01_tempScheduler_exp_target0_5_optimizeFromEpoch100to250_dr2_lambda_1_0/log.txt'
res1_1 = pd.read_csv(path1_1)
rate1_1 = np.array(res1_1['val_acc_top1'])[-6:-1]
# rate1_1 = [r.replace('tensor(','') for r in rate1_1]
# rate1_1 = [r.replace(", device='cuda:0')",'') for r in rate1_1]
# rate1_1 = [float(r) for r in rate1_1]
# rate1_1 = np.array(rate1_1)

path1_2 = '/home/hanyz/code/sarNet/log/cifar100/sar_resnet32x2_alphaBase_cifar/_round2_optimRate_g4_a2b2_s2/t0_1_0_tLast0_01_tempScheduler_exp_target0_5_optimizeFromEpoch100to250_dr2_lambda_1_0/log.txt'
res1_2 = pd.read_csv(path1_2)
rate1_2 = np.array(res1_2['val_acc_top1'])[-6:-1]
# rate1_2 = [r.replace('tensor(','') for r in rate1_2]
# rate1_2 = [r.replace(", device='cuda:0')",'') for r in rate1_2]
# rate1_2 = [float(r) for r in rate1_2]
# rate1_2 = np.array(rate1_2)


rate = np.hstack((rate1_1,rate1_2))
std = np.std(rate)
print(std)