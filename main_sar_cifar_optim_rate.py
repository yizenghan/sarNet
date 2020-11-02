import os
import argparse
import time
import random
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
import pandas as pd
import sys
# current_path = os.path.dirname('obs://d-cheap-net-shanghai/hanyz/sarNet/main_sar.py')
# sys.path.append(current_path)
# import double_checked_models
import models
from utils import *
from optimizer import get_optimizer
from criterion import get_criterion
from scheduler import get_scheduler
from transform import get_transform
import glob
import math
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as pytorchmodels
import torchvision.transforms as transforms
from queue_jump import check_gpu_memory

parser = argparse.ArgumentParser(description='PyTorch SARNet')
parser.add_argument('--config', help='train config file path')
parser.add_argument('--data_url', type=str, metavar='DIR', default='/home/hanyz/data/',
                    help='path to dataset')
parser.add_argument('--train_url', type=str, metavar='PATH', default='./log/',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'],
                    help='dataset')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# hyperparameters
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--scheduler', default='cosine', type=str, metavar='T',
                    help='learning rate strategy (default: multistep)',
                    choices=['cosine', 'multistep', 'linear'])                    
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='SGD', type=str)
# structure
parser.add_argument('--type', default='bottleneck', type=str)
parser.add_argument('--arch', default='sar_resnet_cifar4stage_alphaBase', type=str)
parser.add_argument('--arch_config', default='sar_resnet50_alphaBase_4stage_cifar', type=str)
parser.add_argument('--patch_groups', default=2, type=int)
parser.add_argument('--mask_size', default=4, type=int)
parser.add_argument('--alpha', default=2, type=int)
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--width', default=1, type=int)
parser.add_argument('--base_scale', default=2, type=int)
# mask control
parser.add_argument('--t0', default=5.0, type=float, metavar='M', help='momentum')
parser.add_argument('--t_last', default=1e-3, type=float, metavar='M', help='momentum')
parser.add_argument('--target_rate', default=0, type=float, metavar='M', help='momentum')
parser.add_argument('--lambda_act', default=0.1, type=float, metavar='M', help='momentum')
parser.add_argument('--temp', default=0.1, type=float, metavar='M', help='momentum')
parser.add_argument('--lrfact', default=1, type=float, help='learning rate factor')
parser.add_argument('--temp_scheduler', default='cosine', type=str)

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--labelsmooth', default=0, type=int)
parser.add_argument('--mixup', default=0.0, type=float)
parser.add_argument('--warmup_epoch', default=None, type=int, metavar='N',
                    help='number of epochs to warm up')
parser.add_argument('--warmup_lr', default=0.1, type=float,
                    metavar='LR', help='initial warm up learning rate (default: 0.1)')
parser.add_argument('--weigh_decay_apply_on_all', default=True, type=str)
parser.add_argument('--nesterov', default=True, type=str)                    
parser.add_argument('--print_freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set (default: false)')
parser.add_argument('--evaluate_from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')

# huawei cloud
parser.add_argument('--no_train_on_cloud', dest='train_on_cloud', action='store_false', default=False,
                    help='whether to run the code on huawei cloud')
parser.add_argument('--init_method', type=str, default='',
                    help='an argument needed in huawei cloud, but i do not know its usage')
parser.add_argument('--test_code', default=0, type=int,
                    help='whether to test the code')

parser.add_argument('--start_eval_epoch', default=0, type=int)
parser.add_argument('--round', default=0, type=int)
parser.add_argument('--dynamic_rate', default=0, type=int)

parser.add_argument('--t_last_epoch', default=160, type=int)

parser.add_argument('--ta_begin_epoch', default=80, type=int)
parser.add_argument('--ta_last_epoch', default=120, type=int)

args = parser.parse_args()

if args.train_on_cloud:
    import moxing as mox
    mox.file.shift('os', 'mox')

best_acc1 = 0
best_acc1_corresponding_acc5 = 0
val_acc_top1 = []
val_acc_top5 = []
val_act_rate = []
val_FLOPs = []
tr_acc_top1 = []
tr_acc_top5 = []
train_loss = []
valid_loss = []
lr_log = []
epoch_log = []


def main():
    # check_gpu_memory()
    if args.dataset == 'cifar10':
        args.num_classes = 10
    else:
        args.num_classes = 100
    if not args.train_on_cloud:
        if not os.path.exists(args.train_url):
            os.makedirs(args.train_url)
        else:
            
            str_t0 = str(args.t0).replace('.', '_')
            str_lambda = str(args.lambda_act).replace('.', '_')
            str_ta = str(args.target_rate).replace('.', '_')
            str_t_last = str(args.t_last).replace('.', '_')
            save_path = f'{args.train_url}{args.dataset}/{args.arch_config}/_round{args.round}_optimRate_g{args.patch_groups}_a{args.alpha}b{args.beta}_s{args.base_scale}/t0_{str_t0}_tLast{str_t_last}_tempScheduler_{args.temp_scheduler}_target{str_ta}_optimizeFromEpoch{args.ta_begin_epoch}to{args.ta_last_epoch}_dr{args.dynamic_rate}_lambda_{str_lambda}/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            args.train_url = save_path
            print(args.train_url)
            # assert(0==1)
    

    args.IMAGE_SIZE = 32

    main_worker(args)


def main_worker(args):
    global best_acc1
    global best_acc1_corresponding_acc5
    global val_acc_top1
    global val_acc_top5
    global tr_acc_top1
    global tr_acc_top5
    global train_loss
    global valid_loss
    global lr_log
    global epoch_log
    global val_act_rate
    global val_FLOPs
    print(args)

    if args.train_on_cloud:
        with mox.file.File(args.train_url+'train_configs.txt', "w") as f:
            f.write(str(args))
    else:
        with open(args.train_url+'train_configs.txt', "w") as f:
            f.write(str(args))


    ### Create model
    model = eval(f'models.{args.arch}.{args.arch_config}')(args)
    
    print('Model Struture:', str(model))
    if args.train_on_cloud:
        with mox.file.File(args.train_url+'model_arch.txt', "w") as f:
            f.write(str(model))
    else:
        with open(args.train_url+'model_arch.txt', "w") as f:
            f.write(str(model))
    model.eval()
    rand_inp = torch.rand(1, 3, 32, 32)
    _, _, args.full_flops = model.forward_calc_flops(rand_inp, temperature=1e-8)
    args.full_flops /= 1e9
    print(f'FULL FLOPs: {args.full_flops} G')
    # assert(0==1)
    ### Optionally evaluate from a model
    if args.evaluate_from is not None:
        args.evaluate = True
        state_dict = torch.load(args.evaluate_from, map_location='cpu')['state_dict']

        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    ### Define loss function (criterion) and optimizer
    criterion = get_criterion(args).cuda()

    # optimizer_backbone = torch.optim.SGD(model.get_backbone_params(), args.lr,
    #                                momentum=args.momentum,
    #                                weight_decay=args.weight_decay,
    #                                nesterov=args.nesterov)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args)

    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    # optionally resume from a checkpoint
    # args.gpu = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc1_corresponding_acc5 = ['best_acc1_corresponding_acc5']
            model.load_state_dict(checkpoint['state_dict'])
            if not args.evaluate:
                optimizer.load_state_dict(checkpoint['optimizer'])
            val_acc_top1 = checkpoint['val_acc_top1']
            val_acc_top5 = checkpoint['val_acc_top5']
            val_act_rate = checkpoint['val_act_rate']
            val_FLOPs = checkpoint['val_FLOPs']
            tr_acc_top1 = checkpoint['tr_acc_top1']
            tr_acc_top5 = checkpoint['tr_acc_top5']
            train_loss = checkpoint['train_loss']
            valid_loss = checkpoint['valid_loss']
            lr_log = checkpoint['lr_log']
            try:
                epoch_log = checkpoint['epoch_log']
            except:
                print('There is no epoch_log in checkpoint!')
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        # assert(0==1)

    cudnn.benchmark = True

    if args.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                         std=[0.2471, 0.2435, 0.2616])
        trainset = datasets.CIFAR10(args.data_url, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize
                                     ]))
        valset = datasets.CIFAR10(args.data_url, train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       normalize
                                   ]))
    elif args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        trainset = datasets.CIFAR100(args.data_url, train=True, download=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize
                                      ]))
        valset = datasets.CIFAR100(args.data_url, train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        normalize
                                    ]))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    if args.evaluate:
        # target_rate = args.target_rate
        validate(val_loader, model, criterion, args)
        return

    epoch_time = AverageMeter('Epoch Tiem', ':6.3f')
    start_time = time.time()
    acc_avg = []
    rate_avg = []
    flops_avg = []
    for epoch in range(args.start_epoch, args.epochs):
        ### Train for one epoch
        target_rate = adjust_target_rate(epoch, args)
        print(f'Target rate: {target_rate}')
        tr_acc1, tr_acc5, tr_loss, lr = \
            train(train_loader, model, criterion, optimizer, scheduler, epoch, args, target_rate)
        # tr_acc1, tr_acc5, tr_loss, lr = 0,0,0,0
        if epoch % 10 == 0 or epoch >= args.start_eval_epoch:
            ### Evaluate on validation set
            val_acc1, val_acc5, val_loss, val_rate, val_flops = validate(val_loader, model, criterion, args, target_rate)
            # assert(0==1)
            ### Remember best Acc@1 and save checkpoint
            is_best = val_acc1 > best_acc1
            if is_best:
                best_acc1_corresponding_acc5 = val_acc5
            best_acc1 = max(val_acc1, best_acc1)

            val_acc_top1.append(val_acc1)
            val_acc_top5.append(val_acc5)
            val_act_rate.append(val_rate)
            val_FLOPs.append(val_flops)
            tr_acc_top1.append(tr_acc1)
            tr_acc_top5.append(tr_acc5)
            train_loss.append(tr_loss)
            valid_loss.append(val_loss)
            lr_log.append(lr)
            epoch_log.append(epoch)
            print('val_act_rate',len(val_act_rate))
            print('lr_log',len(lr_log))
            print('epoch_log',len(epoch_log))

            if epoch >= args.epochs - 5:
                acc_avg.append(val_acc1)
                rate_avg.append(val_rate)
                flops_avg.append(val_flops)

            df = pd.DataFrame({'val_acc_top1': val_acc_top1, 'val_acc_top5': val_acc_top5, 
                                'val_act_rate': val_act_rate, 'val_FLOPs': val_FLOPs, 
                                'tr_acc_top1': tr_acc_top1, 'tr_acc_top5': tr_acc_top5, 
                                'train_loss': train_loss, 'valid_loss': valid_loss,
                               'lr_log': lr_log, 'epoch_log': epoch_log})
            log_file = args.train_url + 'log.txt'
            if args.train_on_cloud:
                with mox.file.File(log_file, "w") as f:
                    df.to_csv(f)
            else:
                with open(log_file, "w") as f:
                    df.to_csv(f)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.arch_config,
                'hyper_set': str(args),
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc1_corresponding_acc5': best_acc1_corresponding_acc5,
                'optimizer': optimizer.state_dict(),
                'val_acc_top1': val_acc_top1,
                'val_acc_top5': val_acc_top5,
                'val_act_rate': val_act_rate,
                'val_FLOPs': val_FLOPs,
                'tr_acc_top1': tr_acc_top1,
                'tr_acc_top5': tr_acc_top5,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'lr_log': lr_log,
                'epoch_log': epoch_log,
            }, args, is_best, filename='checkpoint.pth.tar')

        epoch_time.update(time.time() - start_time, 1)
        print('Duration: %4f H, Left Time: %4f H' % (
            epoch_time.sum / 3600, epoch_time.avg * (args.epochs - epoch - 1) / 3600))
        start_time = time.time()

    fout = open(os.path.join(args.train_url, 'log.txt'), mode='a', encoding='utf-8')
    fout.write("%.6f\t%.6f\t%.6f" % (sum(acc_avg)/5, sum(rate_avg)/5, sum(flops_avg)/5))
    print(' * Best Acc@1 {best_acc1:.3f} Acc@5 {best_acc1_corresponding_acc5:.3f}'
          .format(best_acc1=best_acc1, best_acc1_corresponding_acc5=best_acc1_corresponding_acc5))
    return


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args, target_rate):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_cls = AverageMeter('Loss_cls', ':.4e')
    losses_act = AverageMeter('Loss_activate', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    act_rates = AverageMeter('Activation rate', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    train_batches_num = len(train_loader)

    train_progress = ProgressMeter(
        train_batches_num,
        [batch_time, data_time,act_rates, losses, losses_cls, losses_act, top1, top5],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        ### Adjust learning rate
        lr = scheduler.step(optimizer, epoch, batch=i, nBatch=len(train_loader))

        ### Measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        ### Compute output
        adjust_gs_temperature(epoch, i, train_batches_num, args)
        if args.mixup > 0.0:
            input, target_a, target_b, lam = mixup_data(input, target, args.mixup)
            output, _masks = model(input, temperature=args.temp, inference=False)
            loss_cls = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            output, _masks = model(input, temperature=args.temp, inference=False)
            loss_cls = criterion(output, target)

        ### Measure accuracy and record loss
        if args.mixup > 0.0:
            acc1_a, acc5_a = accuracy(output.data, target_a, topk=(1, 5))
            acc1_b, acc5_b = accuracy(output.data, target_b, topk=(1, 5))
            acc1 = lam * acc1_a + (1 - lam) * acc1_b
            acc5 = lam * acc5_a + (1 - lam) * acc5_b
        else:
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        
        act_rate = 0.0
        loss_act_rate = 0.0
        # print(len(_masks))
        # assert(0==1)
        for act in _masks:
            act_rate += torch.mean(act)
            loss_act_rate += torch.pow(target_rate-torch.mean(act), 2)
        act_rate = torch.mean(act_rate/len(_masks))
        loss_act_rate = torch.mean(loss_act_rate/len(_masks))
        loss_act_rate = args.lambda_act * loss_act_rate
        # print(loss_cls, loss_act_rate)
        if args.dynamic_rate > 0:
            loss = loss_cls + loss_act_rate
        else:
            loss = loss_cls + loss_act_rate if epoch >= args.ta_begin_epoch else loss_cls
        
        act_rates.update(act_rate.item(), input.size(0))
        losses_act.update(loss_act_rate.item(),input.size(0))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        ### Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            train_progress.display(i)
            print('LR: %6.4f' % (lr))

    return top1.avg, top5.avg, losses.avg, lr

def validate(val_loader, model, criterion, args, target_rate):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_cls = AverageMeter('Loss_cls', ':.4e')
    losses_act = AverageMeter('Loss_activate', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    act_rates = AverageMeter('Activation rate', ':.2e')
    FLOPs = AverageMeter('FLOPs', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, act_rates, FLOPs, losses, losses_cls, losses_act, top1, top5],
        prefix='Test: ')

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            ### Compute output single crop
            # output = model(input)
            output, _masks, flops = model.module.forward_calc_flops(input, temperature=args.t_last, inference=False)
            flops /= 1e9
            loss_cls= criterion(output, target)
            act_rate = 0.0
            loss_act_rate = 0.0
            for act in _masks:
                act_rate += torch.mean(act)
                loss_act_rate += torch.pow(target_rate-torch.mean(act), 2)
            act_rate = torch.mean(act_rate/len(_masks))
            loss_act_rate = torch.mean(loss_act_rate/len(_masks))
            loss_act_rate = args.lambda_act * loss_act_rate
            loss = loss_cls + loss_act_rate
            
            FLOPs.update(flops.item(), input.size(0))
            act_rates.update(act_rate.item(), input.size(0))
            losses_act.update(loss_act_rate.item(),input.size(0))
            losses_cls.update(loss_cls.item(), input.size(0))


            ### Measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            ### Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} FLOPs {flops.avg}'
          .format(top1=top1, top5=top5, flops =FLOPs))

    return top1.avg, top5.avg, losses.avg, act_rates.avg, FLOPs.avg

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
        elif epoch < args.ta_last_epoch:
            target_rate = args.target_rate + (1.0 - args.target_rate)/3
        else:
            target_rate = args.target_rate
    return target_rate

if __name__ == '__main__':
    main()
