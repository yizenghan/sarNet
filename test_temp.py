import math
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='PyTorch SARNet')
parser.add_argument('--t_last_epoch', default=300, type=int)
parser.add_argument('--t_last', default=1e-2, type=int)
# parser.add_argument('--epochs', default=160, type=int)
parser.add_argument('--t0', default=3, type=int)
parser.add_argument('--dynamic_rate', default=2, type=int)
parser.add_argument('--ta_begin_epoch', default=45, type=int)
parser.add_argument('--ta_last_epoch', default=90, type=int)
parser.add_argument('--target_rate', default=0.3, type=float, metavar='M', help='momentum')
parser.add_argument('--temp', default=0.1, type=float, metavar='M', help='momentum')
parser.add_argument('--temp_scheduler', default='cosine', type=str)
args = parser.parse_args()


def adjust_gs_temperature(epoch, step, len_epoch, args):
    if epoch >= args.t_last_epoch:
        return args.t_last
    else:
        T_total = args.t_last_epoch * len_epoch
        T_cur = epoch * len_epoch + step
        if args.temp_scheduler == 'exp':
            alpha = math.pow(args.t_last / args.t0, 1 / T_total)
            args.temp = math.pow(alpha, T_cur) * args.t0
        elif args.temp_scheduler == 'linear':
            args.temp = (args.t0 - args.t_last) * (1 - T_cur / T_total) + args.t_last
        else:
            args.temp = 0.5 * (args.t0-args.t_last) * (1 + math.cos(math.pi * T_cur / (T_total))) + args.t_last

def adjust_target_rate(epoch, args):
    if args.dynamic_rate == 0:
        return args.target_rate
    elif args.dynamic_rate == 1:
        if epoch < args.ta_last_epoch // 2:
            target_rate = 1.0
        else:
            target_rate = args.target_rate
    else:
        if epoch < args.ta_begin_epoch :
            target_rate = 1.0
        elif epoch < args.ta_begin_epoch + (args.ta_last_epoch-args.ta_begin_epoch)//2:
            target_rate = args.target_rate + (1.0 - args.target_rate)/3*2
        # elif epoch < args.ta_last_epoch:
        #     Ta_total = (args.ta_last_epoch-args.ta_begin_epoch)* len_epoch
        #     Ta_cur = (epoch-args.ta_begin_epoch)* len_epoch + step
        #     alpha = math.pow(args.target_rate / 1, 1 / Ta_total)
        #     target_rate = math.pow(alpha, Ta_cur)
            # target_rate = args.target_rate + (1.0 - args.target_rate)/3
        else:
            target_rate = args.target_rate
    return target_rate

temp_list = []
tar_list = []
epochs = 300
for epoch in range(epochs):
    target = adjust_target_rate(epoch, args)
    tar_list.append(target)
    len_epoch = 100
    for i in range(len_epoch):
        adjust_gs_temperature(epoch, i, len_epoch, args)
        temp_list.append(args.temp)
        # target = adjust_target_rate(epoch, i, len_epoch, args)
        # tar_list.append(target)

print(temp_list[200*len_epoch])
x = range(epochs*len_epoch)
plt.plot(x, temp_list, color='green', label='temp')
plt.savefig("temp.jpg")