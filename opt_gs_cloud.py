import moxing as mox
mox.file.shift('os', 'mox')

import argparse
import itertools
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import os
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
import shutil
import warnings
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import math

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

parser = argparse.ArgumentParser(description='PyTorch SE Training')
parser.add_argument('--epochs', default=90, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=1024, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('--lr', '--learning_rate', default=0.4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--train_url', default='./log', type=str, metavar='Logdir', help='log dir')
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--lr_scheduler',  default = 'cosine', type = str)
parser.add_argument('--data_url', default = '/home/hanyz/data', type=str)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--arch', default = 'densenet_bc', type=str)
parser.add_argument('--arch_config', default='densenet_k12_d100_cifar',type=str)
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--init_method', type=str, default='',
                    help='an argument needed in huawei cloud, but i do not know its usage')
parser.add_argument('--schedule_iter', action='store_true', default=False)
parser.add_argument('--visible_gpus', type=str, default='0', help='visible gpus')
parser.add_argument('--t0', default=0.5, type=float, metavar='M', help='momentum')
parser.add_argument('--target_rate', default=0.5, type=float, metavar='M', help='momentum')
parser.add_argument('--lambda_act', default=1.0, type=float, metavar='M', help='momentum')
parser.add_argument('--temp', default=0.1, type=float, metavar='M', help='momentum')
parser.add_argument('--lrfact', default=1, type=float,
                    help='learning rate factor')
parser.add_argument('--dynamic_rate', default=0, type=int)
parser.add_argument('--patch_groups', default=1, type=int)
parser.add_argument('--optimize_rate_begin_epoch', default=45, type=int)
best_acc1 = 0.0
val_acc_top1 = []
val_acc_top5 = []
val_loss = []

def main():
    args = parser.parse_args()
    args.multiprocessing_distributed = True 
    args.schedule_iter = True
    args.num_classes = 1000
    args.dynamic_rate = True if args.dynamic_rate == 1 else False
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                    'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    
    with mox.file.File(args.train_url+'train_configs.txt', "w") as f:
        f.write(str(args))
    
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    model = eval(f'models.{args.arch}.{args.arch_config}')(args)
    model_arch_str = f'{args.arch}\n{args.arch_config}\n{str(model)}'
    with mox.file.File(args.train_url + 'model_arch.txt', 'w') as output_file: 
        output_file.write(model_arch_str) 
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = torch.tensor(best_acc1).to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            assert('No checkpoint!!!')
    cudnn.benchmark = True

    traindir = args.data_url + 'train/'
    valdir = args.data_url + 'val/'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    if args.eval:
        validate(val_loader, model, criterion, args, args.target_rate)
        return
    
    for epoch in range(args.start_epoch, args.epochs):  
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if not args.schedule_iter:
            adjust_learning_rate(optimizer, epoch, args)
        target_rate = adjust_target_rate(epoch, args)
        train(train_loader, model,criterion, optimizer, epoch, args, target_rate)
        if epoch >= args.epochs - 10:
            val_loss,val_acc1,val_acc5 = validate(val_loader, model,criterion, args, target_rate)
            is_best = val_acc1 > best_acc1
            best_acc1 = max(val_acc1,best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
                val_acc_top1.append(val_acc1)
                val_acc_top5.append(val_acc5)
                acc_file = args.train_url + 'val_accuracy.txt'
                acc_df = pd.DataFrame({'val_acc_top1@1': val_acc_top1, 'val_acc_top1@5': val_acc_top5})
                with mox.file.File(acc_file, "w") as acc_f:
                    acc_df.to_csv(acc_f)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, args, epoch,filename='checkpoint.pth.tar')
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, False, args, epoch,filename='checkpoint.pth.tar')


def train(train_loader, model,criterion, optimizer, epoch, args, target_rate):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_cls = AverageMeter('Loss_cls', ':.4e')
    losses_act = AverageMeter('Loss_activate', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    act_rates = AverageMeter('Activation rate', ':.2e')
    acc1 = AverageMeter('Acc@1', ':6.2f')
    acc5 = AverageMeter('Acc@5', ':6.2f')
    train_batches_num = len(train_loader)

    progress = ProgressMeter(
        train_batches_num,
        [batch_time, data_time,act_rates, losses, losses_cls, losses_act, acc1, acc5],
        prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        if args.gpu is not None:
            input = input.cuda(args.gpu)
        target = target.cuda(args.gpu)
        data_time.update(time.time() - end)

        if args.schedule_iter:
            adjust_learning_rate_iter_warmup(optimizer, epoch, i, train_batches_num, args)

        adjust_gs_temperature(epoch, i, train_batches_num, args)
        output, gate_activation = model(input, temperature=args.temp, inference=False)
        gate_activation = [a for a in gate_activation if a is not None]
        loss_cls = criterion(output, target)

        act_rate = 0.0
        loss_act_rate = 0.0
        for act in gate_activation:
            act_rate += torch.mean(act)
            loss_act_rate += torch.pow(target_rate-torch.mean(act), 2)
        act_rate = torch.mean(act_rate/len(gate_activation))
        loss_act_rate = torch.mean(loss_act_rate/len(gate_activation))
        loss_act_rate = args.lambda_act * loss_act_rate
        loss = loss_cls + loss_act_rate if epoch >= args.optimize_rate_begin_epoch else loss_cls

        if math.isnan(loss.item()):
            optimizer.zero_grad()
            continue 
        elif math.isnan(loss_act_rate.item()):
            print(gate_activation)
            optimizer.zero_grad()
            loss_cls.backward()
            continue 
        act_rates.update(act_rate.item(), input.size(0))
        losses_act.update(loss_act_rate.item(),input.size(0))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        acc1.update(prec1[0], input.size(0))
        acc5.update(prec5[0], input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0 or i == train_batches_num:
            progress.display(i)
    return losses.avg, acc1.avg, acc5.avg

def validate(val_loader, model,criterion, args, target_rate):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    losses_cls = AverageMeter('Loss_cls', ':.4e')
    losses_act = AverageMeter('Loss_activate', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    act_rates = AverageMeter('Activation rate', ':.2e')
    acc1 = AverageMeter('Acc@1', ':6.2f')
    acc5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, act_rates,losses, losses_cls, losses_act, acc1, acc5],
        prefix='Test: ')
    
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            output, gate_activation = model(input, temperature=args.temp, inference=False)
            gate_activation = [a for a in gate_activation if a is not None]
            loss_cls = criterion(output, target)
            
            act_rate = 0.0
            loss_act_rate = 0.0
            for act in gate_activation:
                act_rate += torch.mean(act)
                loss_act_rate += torch.pow(target_rate-torch.mean(act), 2)
            act_rate = torch.mean(act_rate/len(gate_activation))
            loss_act_rate = torch.mean(loss_act_rate/len(gate_activation))
            loss_act_rate = args.lambda_act * loss_act_rate
            loss = loss_cls + loss_act_rate
            
            act_rates.update(act_rate.item(), input.size(0))
            losses_act.update(loss_act_rate.item(),input.size(0))
            losses_cls.update(loss_cls.item(), input.size(0))
            losses.update(loss.item(), input.size(0))
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            acc1.update(prec1[0], input.size(0))
            acc5.update(prec5[0], input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 100 == 0:
                progress.display(i)
    
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=acc1, top5=acc5))
    return losses_cls.avg, acc1.avg, acc5.avg

def adjust_learning_rate_iter_warmup(optimizer, epoch, step, len_epoch, args):
    if args.lr_scheduler == 'multiStep':
        factor = epoch // 30
        if epoch >= 80:
            factor = factor + 1
        lr = args.lr*(0.1**factor)
        """Warmup"""
        if epoch < 5:
            lr = lr/4 + (lr - lr/4)*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    elif args.lr_scheduler == 'cosine':
        if epoch < 5:
            lr = args.lr/4 + 3*args.lr/4 * float(1 + step + epoch*len_epoch)/(5.*len_epoch)
        else:
            lr = 0.5 * args.lr * (1 + math.cos(math.pi * float((epoch-5)*len_epoch+step) / float((args.epochs-5)*len_epoch)))
    else:
        assert('lr_scheduler wrong')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_gs_temperature(epoch, step, len_epoch, args):
    alpha = math.pow(0.01/args.t0, 1/(args.epochs*len_epoch))
    args.temp = math.pow(alpha, epoch*len_epoch+step)*args.t0

def adjust_target_rate(epoch, args):
    if not args.dynamic_rate:
        return args.target_rate
        
    if epoch < 45:
        target_rate = 0.95
    elif epoch < 75:
        target_rate = (args.target_rate - 0.95) / 30 * (epoch - 44) + 0.95
    else:
        target_rate = args.target_rate
    return target_rate

def adjust_learning_rate(optimizer, epoch, args):
    if args.lr_scheduler == 'multiStep':
        lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print('lr:',param_group['lr'])
    elif args.lr_scheduler == 'cosine':
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
            print('lr:',param_group['lr'])
    else:
        assert('scheduler not defined')
    

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state, is_best, args, epoch,filename='checkpoint.pth.tar'):
    try:
        if epoch > 0:
            mox.file.copy(args.train_url + 'checkpoint.pth.tar', args.train_url + 'checkpoint_last.pth.tar')
        torch.save(state, args.train_url + filename)
        if is_best:
            mox.file.copy(args.train_url + filename, args.train_url + 'model_best.pth.tar')
        print('save checkpoint success')
    except:
        print('save checkpoint error, continue to train')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
