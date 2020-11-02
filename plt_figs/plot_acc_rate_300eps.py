import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

epochs = np.array([e for e in range(300)])


fig = plt.figure()
path1_1 = '//home/hanyz/code/sarNet/log/cifar10/sar_resnet32x2_alphaBase_cifar/Epochs300_interval5_target0_4_optimizeFromEpoch0to200_dr0_lambda_1_0/_optimRate_g2_a2b1_s2/t0_1_0_tLast0_01_tempScheduler_exp/log.txt'
res1_1 = pd.read_csv(path1_1)
acc1_1 = np.array(res1_1['val_acc_top1'])[:-1]
rate1_1 = np.array(res1_1['val_act_rate'])[:-1]
path1_2 = '//home/hanyz/code/sarNet/log/cifar10/sar_resnet32x2_alphaBase_cifar/Epochs300_interval5_target0_4_optimizeFromEpoch0to300_dr0_lambda_1_0/_optimRate_g2_a2b1_s2/t0_1_0_tLast0_01_tempScheduler_exp/log.txt'
res1_2 = pd.read_csv(path1_2)
acc1_2 = np.array(res1_2['val_acc_top1'])[:-1]
rate1_2 = np.array(res1_2['val_act_rate'])[:-1]

plt.plot(epochs, rate1_1, c = '#ff3300', label='valid_rate, 0 to 200')
xlabel(r'$epoch$')
ylabel(r'$activation rate$')
plt.plot(epochs, rate1_2, c = '#ffccff', label='valid_rate, 0 to 300')
plt.legend(loc=2)

twinx()

plt.plot(epochs, acc1_1, c = '#0066ff', label='valid_acc1, 0 to 200')
ylabel(r'$acc1$')
plt.plot(epochs, acc1_2, c = '#00ccff', label='valid_acc1, 0 to 300')
plt.legend(loc=3)
plt.savefig('internal5_t1_exp_ta04_0to200.png')

fig = plt.figure()
path1_1 = '//home/hanyz/code/sarNet/log/cifar10/sar_resnet32x2_alphaBase_cifar/Epochs300_interval5_target0_6_optimizeFromEpoch0to200_dr0_lambda_1_0/_optimRate_g2_a2b1_s2/t0_1_0_tLast0_01_tempScheduler_exp/log.txt'
res1_1 = pd.read_csv(path1_1)
acc1_1 = np.array(res1_1['val_acc_top1'])[:-1]
rate1_1 = np.array(res1_1['val_act_rate'])[:-1]
path1_2 = '//home/hanyz/code/sarNet/log/cifar10/sar_resnet32x2_alphaBase_cifar/Epochs300_interval5_target0_6_optimizeFromEpoch0to300_dr0_lambda_1_0/_optimRate_g2_a2b1_s2/t0_1_0_tLast0_01_tempScheduler_exp/log.txt'
res1_2 = pd.read_csv(path1_2)
acc1_2 = np.array(res1_2['val_acc_top1'])[:-1]
rate1_2 = np.array(res1_2['val_act_rate'])[:-1]

plt.plot(epochs, rate1_1, c = '#ff3300', label='valid_rate, 0 to 200')
xlabel(r'$epoch$')
ylabel(r'$activation rate$')
plt.plot(epochs, rate1_2, c = '#ffccff', label='valid_rate, 0 to 300')
plt.legend(loc=2)

twinx()

plt.plot(epochs, acc1_1, c = '#0066ff', label='valid_acc1, 0 to 200')
ylabel(r'$acc1$')
plt.plot(epochs, acc1_2, c = '#00ccff', label='valid_acc1, 0 to 300')
plt.legend(loc=3)
plt.savefig('internal5_t1_exp_ta06_0to200.png')

fig = plt.figure()
path1_1 = '//home/hanyz/code/sarNet/log/cifar10/sar_resnet32x2_alphaBase_cifar/Epochs300_interval5_target0_25_optimizeFromEpoch0to200_dr0_lambda_1_0/_optimRate_g2_a2b1_s2/t0_1_0_tLast0_01_tempScheduler_exp/log.txt'
res1_1 = pd.read_csv(path1_1)
acc1_1 = np.array(res1_1['val_acc_top1'])[:-1]
rate1_1 = np.array(res1_1['val_act_rate'])[:-1]
path1_2 = '//home/hanyz/code/sarNet/log/cifar10/sar_resnet32x2_alphaBase_cifar/Epochs300_interval5_target0_25_optimizeFromEpoch0to300_dr0_lambda_1_0/_optimRate_g2_a2b1_s2/t0_1_0_tLast0_01_tempScheduler_exp/log.txt'
res1_2 = pd.read_csv(path1_2)
acc1_2 = np.array(res1_2['val_acc_top1'])[:-1]
rate1_2 = np.array(res1_2['val_act_rate'])[:-1]

plt.plot(epochs, rate1_1, c = '#ff3300', label='valid_rate, 0 to 200')
xlabel(r'$epoch$')
ylabel(r'$activation rate$')
plt.plot(epochs, rate1_2, c = '#ffccff', label='valid_rate, 0 to 300')
plt.legend(loc=2)

twinx()

plt.plot(epochs, acc1_1, c = '#0066ff', label='valid_acc1, 0 to 200')
ylabel(r'$acc1$')
plt.plot(epochs, acc1_2, c = '#00ccff', label='valid_acc1, 0 to 300')
plt.legend(loc=3)
plt.savefig('internal5_t1_exp_ta025_0to200.png')