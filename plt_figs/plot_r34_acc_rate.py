import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

epochs = np.array([e for e in range(30,109)])


fig = plt.figure()
path1_1 = '../log/___31/_ImageNet/sar_resnet34_alphaBase_4stage_imgnet_OptimRate/g2_a2b1_s2/t0_1_0_tLast0_01_tempScheduler_exp_target0_75_optimizeFromEpoch30to60_dr2_lambda_1_0/log.txt'
res1_1 = pd.read_csv(path1_1)
acc1_1 = np.array(res1_1['val_acc_top1'])[30:]
rate1_1 = np.array(res1_1['val_act_rate'])[30:]

fig = plt.figure()
path1_2= '../log/___31resume/_ImageNet/sar_resnet34_alphaBase_4stage_imgnet_OptimRate/g2_a2b1_s2/t0_1_0_tLast0_01_tempScheduler_exp_target1_0_optimizeFromEpoch0to110_dr1_lambda_1_0/log.txt'
res1_2 = pd.read_csv(path1_2)
acc1_2 = np.array(res1_2['val_acc_top1'])[30:109]
rate1_2 = np.array(res1_2['val_act_rate'])[30:109]


plt.plot(epochs, rate1_1, c = '#ff3300', label='lr, t0=1.0, target=0.75')
plt.plot(epochs, rate1_2, c = '#ff99cc', label='lr, t0=1.0, target=1.0')
xlabel(r'$epoch$')
ylabel(r'$activation rate$')
plt.legend(loc=2)

twinx()

plt.plot(epochs, acc1_1, c = '#0066ff', label='train_loss, t0=1.0, target=0.75')
plt.plot(epochs, acc1_2, c = '#33ccff', label='train_loss, t0=1.0, target=1.0')
ylabel(r'$acc1$')
plt.legend(loc=3)
plt.savefig('r34_loss_lr.png')