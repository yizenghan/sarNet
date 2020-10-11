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


for i in range(110):
    print(adjust_target_rate(i))