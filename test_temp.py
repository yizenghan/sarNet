import math
import numpy
import matplotlib.pyplot as plt

args_target_rate = 0.25
args_epochs = 300
args_temp_scheduler = 'cosine'
args_t0 = 5.0 
args_t_last = 1e-2
len_epoch = 782

def adjust_gs_temperature(epoch, step, len_epoch):
    T_total = args_epochs * len_epoch
    T_cur = epoch * len_epoch + step
    if args_temp_scheduler == 'exp':
        alpha = math.pow(args_t_last/args_t0, 1/(args_epochs*len_epoch))
        args_temp = math.pow(alpha, epoch*len_epoch+step)*args_t0
    elif args_temp_scheduler == 'linear':
        if epoch < args_epochs // 2:
            args_temp = (args_t0 - args_t_last) * (1 - T_cur / (T_total/2)) + args_t_last
        else:
            args_temp = args_t_last
    else:
        if epoch < args_epochs // 2:
            args_temp = 0.5 * (args_t0-args_t_last) * (1 + math.cos(math.pi * T_cur / (T_total/2))) + args_t_last
        else:
            args_temp = args_t_last
    return args_temp

def adjust_target_rate(epoch):
    if epoch < args_epochs // 6 :
        target_rate = 0.9
    elif epoch < args_epochs // 3:
        target_rate = args_target_rate + (0.9 - args_target_rate) / 2
    else:
        target_rate = args_target_rate
    return target_rate

t = []

for i in range(args_epochs):
    for j in range(len_epoch):
        t.append(adjust_gs_temperature(i, j, len_epoch))

# print(t[-1])
# plt.plot(t)
# plt.show()
# plt.savefig('tem.png')

ta = []
for i in range(args_epochs):
    ta.append(adjust_target_rate(i))
plt.plot(ta)
plt.show()
plt.savefig('target_rate.png')