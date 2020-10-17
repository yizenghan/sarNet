import math
import numpy
import matplotlib.pyplot as plt

args_target_rate = 0.3
args_epochs = 110

def adjust_target_rate(epoch):
    if epoch < args_epochs // 4:
        target_rate = 1.0
    elif epoch < args_epochs // 2:
        target_rate = 0.8
    elif epoch < args_epochs // 4 * 3:
        target_rate = (args_target_rate-0.8) / (args_epochs//4) * (epoch - args_epochs // 2) + 0.8
    else:
        target_rate = args_target_rate
    return target_rate

t0 = 5.0
len_epoch = 1252
def adjust_gs_temperature(epoch, step, len_epoch,temp_scheduler='linear'):
    T_total = args_epochs * len_epoch
    T_cur = epoch * len_epoch + step
    if temp_scheduler == 'exp':
        alpha = math.pow(0.01/t0, 1/(args_epochs*len_epoch))
        temp = math.pow(alpha, epoch*len_epoch+step)*t0
    elif temp_scheduler == 'linear':
        temp = (t0 - 0.01) * (1 - T_cur / T_total) + 0.01
    else:
        temp = 0.5 * (t0-0.01) * (1 + math.cos(math.pi * T_cur / T_total)) + 0.01
    return temp

t = []

for i in range(args_epochs):
    for j in range(len_epoch):
        t.append(adjust_gs_temperature(i, j, len_epoch,temp_scheduler='cosine'))

print(t[-1])
plt.plot(t)
plt.show()
plt.savefig('tem.png')
# for i in range(110):
#     print(adjust_target_rate(i))