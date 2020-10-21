# import moxing as mox
# mox.file.shift('os', 'mox')

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

import double_checked_models
import models
from utils import *
from optimizer import get_optimizer
from criterion import get_criterion
from scheduler import get_scheduler
from transform import get_transform
from hyperparams import get_hyperparams
from config import Config

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

parser = argparse.ArgumentParser(description='PyTorch SARNet')
parser.add_argument('--config', help='train config file path')
parser.add_argument('--data_url', type=str, metavar='DIR', default='/data/dataset/CLS-LOC/',
                    help='path to dataset')
parser.add_argument('--train_url', type=str, metavar='PATH', default='./log/test/',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet', choices=['cifar10', 'cifar100', 'imagenet'],
                    help='dataset')
parser.add_argument('-j', '--workers', default=96, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning_rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)')
parser.add_argument('--scheduler', default='multistep', type=str, metavar='T',
                    help='learning rate strategy (default: multistep)',
                    choices=['cosine', 'multistep', 'linear'])
parser.add_argument('--warmup_epoch', default=None, type=int, metavar='N',
                    help='number of epochs to warm up')
parser.add_argument('--warmup_lr', default=0.1, type=float,
                    metavar='LR', help='initial warm up learning rate (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set (default: false)')
parser.add_argument('--evaluate_from', default=None, type=str, metavar='PATH',
                    help='path to saved checkpoint (default: none)')
# hyperparameter
parser.add_argument('--hyperparams_set_index', default=1, type=int,
                    help='choose which hyperparameter set to use')
# huawei cloud
parser.add_argument('--no_train_on_cloud', dest='train_on_cloud', action='store_false', default=True,
                    help='whether to run the code on huawei cloud')
parser.add_argument('--init_method', type=str, default='',
                    help='an argument needed in huawei cloud, but i do not know its usage')
parser.add_argument('--test_code', default=0, type=int,
                    help='whether to test the code')

# multiprocess
parser.add_argument('--world_size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:29501', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--visible_gpus', type=str, default='0',
                    help='visible gpus')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--t0', default=5.0, type=float, metavar='M', help='momentum')
parser.add_argument('--t_last', default=0.01, type=float, metavar='M', help='momentum')
parser.add_argument('--target_rate', default=0.5, type=float, metavar='M', help='momentum')
parser.add_argument('--lambda_act', default=1.0, type=float, metavar='M', help='momentum')
parser.add_argument('--temp', default=0.1, type=float, metavar='M', help='momentum')
parser.add_argument('--lrfact', default=1, type=float,
                    help='learning rate factor')
parser.add_argument('--dynamic_rate', default=0, type=int)
parser.add_argument('--patch_groups', default=1, type=int)
parser.add_argument('--optimize_rate_begin_epoch', default=45, type=int)
parser.add_argument('--temp_scheduler', default='exp', type=str)

parser.add_argument('--use_amp', type=int, default=0,
                    help='apex')

args = parser.parse_args()
args.dynamic_rate = True if args.dynamic_rate > 0 else False
if args.use_amp > 0:
    try:
        from apex import amp
        from apex.parallel import DistributedDataParallel as DDP
        from apex.parallel import convert_syncbn_model
        has_apex = True
    except ImportError:
        os.system('pip --default-timeout=100 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex-master')
        from apex import amp
        from apex.parallel import DistributedDataParallel as DDP
        from apex.parallel import convert_syncbn_model
        has_apex = True
        print('successfully install apex')

best_acc1 = 0
best_acc1_corresponding_acc5 = 0
val_acc_top1 = []
val_acc_top5 = []
tr_acc_top1 = []
tr_acc_top5 = []
train_loss = []
valid_loss = []
lr_log = []
epoch_log = []


def main():

    if not args.train_on_cloud:
        if not os.path.exists(args.train_url):
            os.makedirs(args.train_url)

    assert args.dataset == 'imagenet'
    args.num_classes = 1000
    args.IMAGE_SIZE = 224

    args.multiprocessing_distributed = True
    args.use_amp = True if args.use_amp == 1 else 0

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
    global best_acc1_corresponding_acc5
    global val_acc_top1
    global val_acc_top5
    global tr_acc_top1
    global tr_acc_top5
    global train_loss
    global valid_loss
    global lr_log
    global epoch_log
    args.gpu = gpu
    args.cfg = Config.fromfile(args.config)
    print(args.cfg)
    args.hyperparams_set_index = args.cfg['train_cfg']['hyperparams_set_index']
    args = get_hyperparams(args, test_code=args.test_code)
    print('Hyper-parameters:', str(args))

    if args.train_on_cloud:
        with mox.file.File(args.train_url+'train_configs.txt', "w") as f:
            f.write(str(args))
    else:
        with open(args.train_url+'train_configs.txt', "w") as f:
            f.write(str(args))

    # assert(0==1)
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

    ### Create model
    # model = pytorchmodels.resnet50(pretrained=False)
    model_type = args.cfg['model'].pop('type')
    model = eval(model_type)(**args.cfg['model'])
    print('Model Struture:', str(model))
    if args.train_on_cloud:
        with mox.file.File(args.train_url+'model_arch.txt', "w") as f:
            f.write(str(model))
    else:
        with open(args.train_url+'model_arch.txt', "w") as f:
            f.write(str(model))
    ### Calculate FLOPs & Param
    # params = 0
    # import operator
    # from functools import reduce
    # for m in model.modules():
    #     if is_leaf(m):
    #         # print(m)
    #         params += sum([reduce(operator.mul, i.size(), 1) for i in m.parameters()])
    # print(params)
    # assert 1 == 0

    # n_flops, n_params = measure_model(model, args.cfg['test_cfg']['crop_size'],
    #                                   args.cfg['test_cfg']['crop_size'])
    # print('Params: %.2fM, FLOPs: %.2fM' % (n_params / 1e6, n_flops / 1e6))
    # assert 1==0
    # del (model)
    # model = eval(str(args.model))(args)

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
    criterion = get_criterion(args).cuda(args.gpu)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args)
    

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            if args.use_amp:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            if args.use_amp:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if args.use_amp:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    
    # optionally resume from a checkpoint
    # args.gpu = None
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
            best_acc1_corresponding_acc5 = ['best_acc1_corresponding_acc5']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                # best_acc1 = best_acc1.to(args.gpu)
                # best_acc1_corresponding_acc5 = best_acc1_corresponding_acc5.to(args.gpu)
                pass

            model.load_state_dict(checkpoint['state_dict'])
            if not args.evaluate:
                optimizer.load_state_dict(checkpoint['optimizer'])
            val_acc_top1 = checkpoint['val_acc_top1']
            val_acc_top5 = checkpoint['val_acc_top5']
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

    cudnn.benchmark = True

    ### Data loading
    print('Train data augmentaion:', get_transform(args, is_train_set=True))
    print('Valid data augmentaion:', get_transform(args, is_train_set=False))

    traindir = args.data_url + 'train/'
    valdir = args.data_url + 'val/'

    train_dataset = datasets.ImageFolder(
        traindir,
        get_transform(args, is_train_set=True))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            get_transform(args, is_train_set=False)),
        batch_size=args.batch_size * torch.cuda.device_count(), shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        # target_rate = args.target_rate
        adaptive_inferece(val_loader, model, criterion, args)
        return

    epoch_time = AverageMeter('Epoch Tiem', ':6.3f')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        ### Train for one epoch
        target_rate = adjust_target_rate(epoch, args)
        print(f'Target rate: {target_rate}')
        tr_acc1, tr_acc5, tr_loss, lr = \
            train(train_loader, model, criterion, optimizer, scheduler, epoch, args, target_rate)

        if epoch % 10 == 0 or epoch >= args.start_eval_epoch:
            ### Evaluate on validation set
            val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion, args, target_rate)
            # assert(0==1)
            ### Remember best Acc@1 and save checkpoint
            is_best = val_acc1 > best_acc1
            if is_best:
                best_acc1_corresponding_acc5 = val_acc5
            best_acc1 = max(val_acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                val_acc_top1.append(val_acc1)
                val_acc_top5.append(val_acc5)
                tr_acc_top1.append(tr_acc1)
                tr_acc_top5.append(tr_acc5)
                train_loss.append(tr_loss)
                valid_loss.append(val_loss)
                lr_log.append(lr)
                epoch_log.append(epoch)
                df = pd.DataFrame({'val_acc_top1': val_acc_top1, 'val_acc_top5': val_acc_top5, 'tr_acc_top1': tr_acc_top1,
                                   'tr_acc_top5': tr_acc_top5, 'train_loss': train_loss, 'valid_loss': valid_loss,
                                   'lr_log': lr_log, 'epoch_log': epoch_log})
                log_file = args.train_url + 'log.txt'
                if args.train_on_cloud:
                    with mox.file.File(log_file, "w") as f:
                        df.to_csv(f)
                else:
                    with open(log_file, "w") as f:
                        df.to_csv(f)
                save_checkpoint({
                    'epoch': epoch,
                    'model': model_type,
                    'hyper_set': str(args),
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'best_acc1_corresponding_acc5': best_acc1_corresponding_acc5,
                    'optimizer': optimizer.state_dict(),
                    'val_acc_top1': val_acc_top1,
                    'val_acc_top5': val_acc_top5,
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

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

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
        if args.dynamic_rate:
            loss = loss_cls + loss_act_rate
        else:
            loss = loss_cls + loss_act_rate if epoch >= args.optimize_rate_begin_epoch else loss_cls
        
        # dist.all_reduce(acc1)
        # acc1 /= args.world_size
        # dist.all_reduce(acc5)
        # acc5 /= args.world_size
        # dist.all_reduce(loss)
        # loss /= args.world_size
        # dist.all_reduce(loss_cls)
        # loss_cls /= args.world_size
        # dist.all_reduce(loss_act_rate)
        # loss_act_rate /= args.world_size
        # dist.all_reduce(act_rate)
        # act_rate /= args.world_size

        act_rates.update(act_rate.item(), input.size(0))
        losses_act.update(loss_act_rate.item(),input.size(0))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        ### Compute gradient and do SGD step
        optimizer.zero_grad()
        if args.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            train_progress.display(i)
            print('LR: %6.4f' % (lr))

    return top1.avg, top5.avg, losses.avg, lr

def adaptive_inferece(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_cls = AverageMeter('Loss_cls', ':.4e')
    losses_act = AverageMeter('Loss_activate', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    act_rates = AverageMeter('Activation rate', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    all_flops = AverageMeter('FLOPs', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, act_rates,losses, losses_cls, losses_act, top1, top5, all_flops],
        prefix='Test: ')

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            ### Compute output single crop
            # output = model(input)
            output, _masks, flops = model.module.forward_calc_flops(input, temperature=args.t_last, inference=False)
            # for i in range(len(_masks)):
            #     hhh = (_masks[i] > 0.99).float()
            #     print(hhh.shape, hhh.sum()/_masks[i].numel())
            # assert(0==1)
            flops /= 1e9
            all_flops.update(flops, input.size(0))
            loss_cls= criterion(output, target)
            act_rate = 0.0
            for act in _masks:
                act_rate += torch.mean(act)
                
            act_rate = torch.mean(act_rate/len(_masks))
            loss = loss_cls
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

            dist.all_reduce(acc1)
            acc1 /= args.world_size
            dist.all_reduce(acc5)
            acc5 /= args.world_size
            dist.all_reduce(loss)
            loss /= args.world_size
            dist.all_reduce(loss_cls)
            loss_cls /= args.world_size
            dist.all_reduce(act_rate)
            act_rate /= args.world_size

            act_rates.update(act_rate.item(), input.size(0))
            losses_cls.update(loss_cls.item(), input.size(0))
            losses.update(loss.data.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            ### Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} FLOPs {flops.avg:.4f}'
          .format(top1=top1, top5=top5, flops=all_flops))

    return top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, args, target_rate):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_cls = AverageMeter('Loss_cls', ':.4e')
    losses_act = AverageMeter('Loss_activate', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    act_rates = AverageMeter('Activation rate', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, act_rates,losses, losses_cls, losses_act, top1, top5],
        prefix='Test: ')

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            ### Compute output single crop
            # output = model(input)
            output, _masks = model(input, temperature=args.temp, inference=False)
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
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

            dist.all_reduce(acc1)
            acc1 /= args.world_size
            dist.all_reduce(acc5)
            acc5 /= args.world_size
            dist.all_reduce(loss)
            loss /= args.world_size
            dist.all_reduce(loss_cls)
            loss_cls /= args.world_size
            dist.all_reduce(loss_act_rate)
            loss_act_rate /= args.world_size
            dist.all_reduce(act_rate)
            act_rate /= args.world_size

            act_rates.update(act_rate.item(), input.size(0))
            losses_act.update(loss_act_rate.item(),input.size(0))
            losses_cls.update(loss_cls.item(), input.size(0))

            losses.update(loss.data.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            ### Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def adjust_gs_temperature(epoch, step, len_epoch, args):
    T_total = args.epochs * len_epoch
    T_cur = epoch * len_epoch + step
    if args.temp_scheduler == 'exp':
        alpha = math.pow(args.t_last/args.t0, 1/(args.epochs*len_epoch))
        args.temp = math.pow(alpha, epoch*len_epoch+step)*args.t0
    elif args.temp_scheduler == 'linear':
        args.temp = (args.t0 - args.t_last) * (1 - T_cur / T_total) + args.t_last
    else:
        args.temp = 0.5 * (args.t0-args.t_last) * (1 + math.cos(math.pi * T_cur / T_total)) + args.t_last

def adjust_target_rate(epoch, args):
    if not args.dynamic_rate:
        return args.target_rate
    # if epoch < args.epochs // 4:
    #     target_rate = 1.0
    # elif epoch < args.epochs // 2:
    #     target_rate = 0.8
    # elif epoch < args.epochs // 4 * 3:
    #     target_rate = (args.target_rate-0.8) / (args.epochs//4) * (epoch - args.epochs // 2) + 0.8
    # else:
    #     target_rate = args.target_rate
    # return target_rate

    if epoch < args.optimize_rate_begin_epoch:
        target_rate = 1.0
    else:
        target_rate = args.target_rate
    return target_rate

if __name__ == '__main__':
    main()
