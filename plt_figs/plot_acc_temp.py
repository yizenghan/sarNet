import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

epochs = np.array([e for e in range(100)])


fig = plt.figure()
path1_1 = '/home/hanyz/code/sarNet/log/cifar10/sar_resnet32_alphaBase_cifar/0_02/_round1_optimRate_g1_a1b1_s2/t0_3_0_tLast0_01_tempScheduler_exp_target0_5_optimizeFromEpoch0to100_dr0_lambda_0_5/log.txt'
res1_1 = pd.read_csv(path1_1)
acc1_1 = np.array(res1_1['val_acc_top1'])[:-1]
rate1_1 = np.array(res1_1['val_act_rate'])
rate1_1 = rate1_1[:-1]
# rate1_1 = [r.replace('tensor(','') for r in rate1_1]
# rate1_1 = [r.replace(", device='cuda:0')",'') for r in rate1_1]
# rate1_1 = [float(r) for r in rate1_1]
# rate1_1 = np.array(rate1_1)

path1_2 = '/home/hanyz/code/sarNet/log/cifar10/sar_resnet32_alphaBase_cifar/0_02/_round3_optimRate_g1_a1b1_s2/t0_3_0_tLast0_01_tempScheduler_exp_target0_5_optimizeFromEpoch0to100_dr0_lambda_0_5/log.txt'
res1_2 = pd.read_csv(path1_2)
acc1_2 = np.array(res1_2['val_acc_top1'])[:-1]
rate1_2 = np.array(res1_2['val_act_rate'])[:-1]
# rate1_2 = [r.replace('tensor(','') for r in rate1_2]
# rate1_2 = [r.replace(", device='cuda:0')",'') for r in rate1_2]
# rate1_2 = [float(r) for r in rate1_2]
# rate1_2 = np.array(rate1_2)


plt.plot(epochs, rate1_1, c = '#ff3300', label='act_rate, lr=0.02')
plt.plot(epochs, rate1_2, c = '#ff99cc', label='act_rate, lr=0.02, weight')
xlabel(r'$epoch$')
ylabel(r'$act_rate$')
plt.legend(loc=2)

twinx()

plt.plot(epochs, acc1_1, c = '#0066ff', label='acc1, lr=0.02')
plt.plot(epochs, acc1_2, c = '#33ccff', label='acc1, lr=0.02, weight')
ylabel(r'$acc1$')
plt.legend(loc=3)
plt.savefig('cifar10_finetune.png')


# -----------------------------------------------------------------------------------------


'''fig = plt.figure()
path2_1 = '/data/hanyz/code/SAR_zyt/log/__tempalphaBase_cifar16x4_g2_a2_s2_t0_5_0_target0_0_optimizeFromEpoch201_lambda_0_0_dynamicRate0/log.txt'
res2_1 = pd.read_csv(path2_1)
acc2_1 = np.array(res2_1['val_acc_top1'])[:-1]
rate2_1 = np.array(res2_1['val_act_rate'])
rate2_1 = rate2_1[:-1]
rate2_1 = [r.replace('tensor(','') for r in rate2_1]
rate2_1 = [r.replace(", device='cuda:0')",'') for r in rate2_1]
rate2_1 = [float(r) for r in rate2_1]
rate2_1 = np.array(rate2_1)

path2_2 = '/data/hanyz/code/SAR_zyt/log/__tempalphaBase_cifar16x4_g2_a2_s2_t0_5_0_target0_25_optimizeFromEpoch100_lambda_0_1_dynamicRate0/log.txt'
res2_2 = pd.read_csv(path2_2)
acc2_2 = np.array(res2_2['val_acc_top1'])[:-1]
rate2_2 = np.array(res2_2['val_act_rate'])[:-1]
rate2_2 = [r.replace('tensor(','') for r in rate2_2]
rate2_2 = [r.replace(", device='cuda:0')",'') for r in rate2_2]
rate2_2 = [float(r) for r in rate2_2]
rate2_2 = np.array(rate2_2)

plt.plot(epochs, rate2_1, c = '#ff3300', label='activation rate, t0=5.0, no constraint')
plt.plot(epochs, rate2_2, c = '#ff99cc', label='activation rate, t0=5.0, target=0.25')
xlabel(r'$epoch$')
ylabel(r'$activation rate$')
plt.legend(loc=2)

twinx()

plt.plot(epochs, acc2_1, c = '#0066ff', label='acc1, t0=5.0, no constraint')
plt.plot(epochs, acc2_2, c = '#33ccff', label='acc1, t0=5.0, target=0.25')
ylabel(r'$acc1$')
plt.legend(loc=3)
plt.savefig('model1_temp5.png')


# -----------------------------------------------------------------------------------------


fig = plt.figure()
path3_1 = '/data/hanyz/code/SAR_zyt/log/__tempalphaBase_cifar16x4_g2_a2_s2_t0_10_0_target0_0_optimizeFromEpoch201_lambda_0_0_dynamicRate0/log.txt'
res3_1 = pd.read_csv(path3_1)
acc3_1 = np.array(res3_1['val_acc_top1'])[:-1]
rate3_1 = np.array(res3_1['val_act_rate'])
rate3_1 = rate3_1[:-1]
rate3_1 = [r.replace('tensor(','') for r in rate3_1]
rate3_1 = [r.replace(", device='cuda:0')",'') for r in rate3_1]
rate3_1 = [float(r) for r in rate3_1]
rate3_1 = np.array(rate3_1)

path3_2 = '/data/hanyz/code/SAR_zyt/log/__tempalphaBase_cifar16x4_g2_a2_s2_t0_10_0_target0_25_optimizeFromEpoch100_lambda_0_1_dynamicRate0/log.txt'
res3_2 = pd.read_csv(path3_2)
acc3_2 = np.array(res3_2['val_acc_top1'])[:-1]
rate3_2 = np.array(res3_2['val_act_rate'])[:-1]
rate3_2 = [r.replace('tensor(','') for r in rate3_2]
rate3_2 = [r.replace(", device='cuda:0')",'') for r in rate3_2]
rate3_2 = [float(r) for r in rate3_2]
rate3_2 = np.array(rate3_2)

plt.plot(epochs, rate2_1, c = '#ff3300', label='activation rate, t0=10.0, no constraint')
plt.plot(epochs, rate2_2, c = '#ff99cc', label='activation rate, t0=10.0, target=0.25')
xlabel(r'$epoch$')
ylabel(r'$activation rate$')
plt.legend(loc=2)

twinx()

plt.plot(epochs, acc2_1, c = '#0066ff', label='acc1, t0=10.0, no constraint')
plt.plot(epochs, acc2_2, c = '#33ccff', label='acc1, t0=10.0, target=0.25')
ylabel(r'$acc1$')

plt.legend(loc=3)
plt.savefig('model1_temp10.png')'''