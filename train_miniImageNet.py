import argparse
import time

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from utils import *
from data.miniImageNet import miniImageNet

import numpy as np
import miniimagenet_network.resnet
import miniimagenet_network.densenet_bc 
import plugins.conedensenet 
import plugins.resconenet
from tools.flops_calculate import count_net_flops, count_net_flops_scale
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch ConeConv Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')

parser.add_argument('--model', default='', type=str,
                    help='deep networks to be trained')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')

parser.add_argument('--layers', default=0, type=int,
                    help='total number of layers (have to be explicitly given!)')

parser.add_argument('--growth_rate', default=0, type=int,
                    help='growth_rate for densenet (have to be explicitly given!)')
parser.add_argument('--scale_size', default=4, type=int,
                    help='scale sizefor conenet (have to be explicitly given!)')
# model
parser.add_argument('--separable_flag', dest='separable_flag', action='store_true',
                    help='whether to use separable conv (default: false)')
parser.set_defaults(separable_flag=False)
parser.add_argument('--cond_flag', dest='cond_flag', action='store_true',
                    help='whether to use condconv (default: false)')
parser.set_defaults(cond_flag=False)

parser.add_argument('--droprate', default=0.0, type=float,
                    help='dropout probability (default: 0.0)')

parser.add_argument('--lr', default=0.1, type=float,
                    help='learning rate (default: 0.1)')

parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.set_defaults(augment=True)

parser.add_argument('--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data_path', default='../dataset', type=str,
                    help='path to data (default: none)')

parser.add_argument('--name', default='', type=str,
                    help='name of experiment')
parser.add_argument('--no', default='1', type=str,
                    help='index of the experiment (for recording convenience)')

# Multi cls 
parser.add_argument('--sep', dest='sep', action='store_true',
                    help='train multiscale separately')
parser.set_defaults(sep=False)
parser.add_argument('--cls_share', dest='cls_share', action='store_true',
                    help='multi cls share weights')
parser.set_defaults(cls_share=False)

# Autoaugment
parser.add_argument('--autoaugment', dest='autoaugment', action='store_true',
                    help='whether to use autoaugment')
parser.set_defaults(autoaugment=False)

# cutout
parser.add_argument('--cutout', dest='cutout', action='store_true',
                    help='whether to use cutout')
parser.set_defaults(cutout=False)
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')

# Cosine learning rate
parser.add_argument('--cos_lr', dest='cos_lr', action='store_true',
                    help='whether to use cosine learning rate')
parser.set_defaults(cos_lr=False)

# GPUs
parser.add_argument('--visible_gpus', type=str, default='0', help='visible gpus')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus

# Configurations adopted for training deep networks.
# (specialized for each type of models)
training_configurations = {
    'resnet': {
        'epochs': 160,
        'batch_size': 32,
        'initial_learning_rate': 0.05,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'resconenet': {
        'epochs': 160,
        'batch_size': 32,
        'initial_learning_rate': 0.05,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'densenet': {
        'epochs': 160,
        'batch_size': 32,
        'initial_learning_rate': 0.05,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
    'conedensenet': {
        'epochs': 160,
        'batch_size': 32,
        'initial_learning_rate': 0.05,
        'changing_lr': [80, 120],
        'lr_decay_rate': 0.1,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': 1e-4,
    },
}

record_path = '/home/whh/experiments/' + str(args.dataset) \
              + '_' + str(args.model) \
              + '-' + str(args.layers) \
              + '_' + str(args.name) \
              + '/' + 'no_' + str(args.no) \
              + ('_standard-Aug' if args.augment else '') \
              + ('_dropout_' + str(args.droprate) if args.droprate > 0 else '') \
              + ('_autoaugment' if args.autoaugment else '') \
              + ('_cutout' if args.cutout else '') \
              + ('_cos-lr' if args.cos_lr else '')

record_file = record_path + '/training_process.txt'
accuracy_file = record_path + '/accuracy_epoch.txt'
loss_file = record_path + '/loss_epoch.txt'
check_point = os.path.join(record_path, args.checkpoint)

def main():

    global best_prec1
    best_prec1 = 0

    global best_prec5
    best_prec5 = 0

    global val_acc
    val_acc = []

    global val_acc_top5
    val_acc_top5 = []

    global class_num
    
    class_num = 100

    assert(args.dataset == 'miniImageNet')
    train_dataset = miniImageNet(root=args.data_path, split='train')
    val_dataset = miniImageNet(root=args.data_path, split='val')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=training_configurations[args.model]['batch_size'],
                                               shuffle=True, num_workers=8, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size=training_configurations[args.model]['batch_size'],
                                               shuffle=False, num_workers=8, pin_memory=False)

    # create model
    if args.model == 'resnet':
        model = eval('miniimagenet_network.resnet.resnet' + str(args.layers))(num_classes=class_num)
    elif args.model == 'resconenet':
        model = eval('plugins.resconenet.resconenet' + str(args.layers))(num_classes=class_num, planes=args.growth_rate, separable_flag=args.separable_flag, cond_flag=args.cond_flag)
    elif args.model == 'densenet':
        model = eval('miniimagenet_network.densenet_bc.densenet_bc' +
                     str(args.layers))(num_classes=class_num,
                                       growth_rate=args.growth_rate)
    elif args.model == 'conedensenet':
        model = eval('plugins.conedensenet.conedensenet_bc' + str(args.layers))(num_classes=class_num, growth_rate=args.growth_rate, scale_size=args.scale_size, separable_flag=args.separable_flag)
    else:
        pass

    if not os.path.isdir(check_point):
        mkdir_p(check_point)

    fd = open(record_file, 'a+')

    print('Training Config:', str(training_configurations[args.model]))
    fd.write(str(training_configurations[args.model]) + '\n')

    print('Model Struture:', str(model))
    fd.write(str(model) + '\n')

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
    ))
    #for param in model.parameters():
    #    print(param.shape)
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name)
    flops = count_net_flops_scale(model, args.model, args.scale_size)
    print('Model FLOPs: {}'.format(flops))

    fd.write('Model FLOPs: {}'.format(flops) + '\n')

    fd.write('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])
    ) + '\n')
    fd.close()

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    ce_criterion = nn.CrossEntropyLoss().cuda()
    lr = training_configurations[args.model]['initial_learning_rate']
    if args.sep and not args.cls_share:
        optimizer = torch.optim.SGD([
            {'params': model.conv0.parameters()},
            {'params': model.norm0.parameters()},
            {'params': model.features.parameters()},
            {'params': model.classifier.parameters(), 'lr': args.scale_size * lr},
            ], 
                                    lr=lr,
                                    # lr=args.lr,
                                    momentum=training_configurations[args.model]['momentum'],
                                    nesterov=training_configurations[args.model]['nesterov'],
                                    weight_decay=training_configurations[args.model]['weight_decay'])
    else:
        optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                    lr=lr,
                                    momentum=training_configurations[args.model]['momentum'],
                                    nesterov=training_configurations[args.model]['nesterov'],
                                    weight_decay=training_configurations[args.model]['weight_decay'])

    model = torch.nn.DataParallel(model).cuda()

    tensorboard_writer = SummaryWriter(log_dir=record_path)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        ce_criterion = checkpoint['ce_criterion']
        #val_acc = checkpoint['val_acc']
        #val_acc_top5 = checkpoint['val_acc_top5']
        #print(val_acc)
        best_prec1 = checkpoint['best_acc']
        np.savetxt(accuracy_file, np.array(val_acc))
    else:
        start_epoch = 0

    start_time = time.time()
    for epoch in range(start_epoch, training_configurations[args.model]['epochs']):

        adjust_learning_rate(optimizer, epoch + 1, args, training_configurations)
        if not args.sep:
            # train for one epoch
            train(train_loader, model, ce_criterion, optimizer, epoch, tensorboard_writer)
            # evaluate on validation set
            prec1, prec5 = validate(val_loader, model, ce_criterion, epoch, tensorboard_writer)
        else:
            # train for one epoch
            train_scale_sep(train_loader, model, ce_criterion, optimizer, epoch, tensorboard_writer)
            # evaluate on validation set
            prec1, prec5 = validate_scale_sep(val_loader, model, ce_criterion, epoch, tensorboard_writer)

        pass_time = (time.time() - start_time) / 3600
        left_time = (pass_time * training_configurations[args.model]['epochs'] / (epoch + 1))
        print('Passed time: %f h' % (pass_time))
        print('ETS left time: %f h' %(left_time))

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_prec5 = max(prec5, best_prec5)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_prec1,
            'optimizer': optimizer.state_dict(),
            'ce_criterion': ce_criterion,
            'val_acc': val_acc,
            'val_acc_top5': val_acc_top5,

        }, is_best, checkpoint=check_point)
        print('Best accuracy@1: ', best_prec1)
        print('Best accuracy@5: ', best_prec5)
        all_val = [val_acc, val_acc_top5]
        np.savetxt(accuracy_file, np.array(all_val))

    print('Best accuracy@1: ', best_prec1)
    print('Best accuracy@5: ', best_prec5)
    print('Average accuracy@1', sum(val_acc[-10:]) / 10)
    print('Average accuracy@5', sum(val_acc_top5[-10:]) / 10)
    all_val = [val_acc, val_acc_top5]
    np.savetxt(accuracy_file, np.array(all_val))

def train(train_loader, model, criterion, optimizer, epoch, writer):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    train_batches_num = len(train_loader)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        target = target.cuda()
        x = x.cuda()
        if args.model == 'resconenet' or args.model == 'conedensenet':
            input = []
            input.append(x)
            rate = 0.5 
            for _ in range(1, args.scale_size):
                x_tmp = nn.functional.interpolate(x, scale_factor=rate, mode='bilinear', align_corners=True) 
                rate *= 0.5
                input.append(x_tmp)
        else:
            input = x

        data_time.update(time.time() - end)
        end = time.time()
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec[0].item(), x.size(0))
        top5.update(prec[1].item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            fd = open(record_file, 'a+')
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Lr {lr:.4f}\t'
                      'Batch_Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
                      'Data_Time {data_time.value:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.value:.3f} ({top5.avg:.3f})\t'.format(
                       epoch, i+1, train_batches_num, lr=optimizer.param_groups[0]['lr'], batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))

            print(string)
            fd.write(string + '\n')
            fd.close()

    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/acc@1', top1.avg, epoch)
    writer.add_scalar('train/acc@5', top5.avg, epoch)

def validate(val_loader, model, criterion, epoch, writer):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (x, target) in enumerate(val_loader):
        target = target.cuda()
        x = x.cuda()

        # compute output
        with torch.no_grad():
            if args.model == 'resconenet' or args.model == 'conedensenet':
                input = []
                input.append(x)
                rate = 0.5 
                for _  in range(1, args.scale_size):
                    x_tmp = nn.functional.interpolate(x, scale_factor=rate, mode='bilinear', align_corners=True) 
                    rate *= 0.5
                    input.append(x_tmp)
            else:
                input = x
            output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), x.size(0))
        top1.update(prec[0].item(), x.size(0))
        top5.update(prec[1].item(), x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            fd = open(record_file, 'a+')
            string = ('Test: [{0}][{1}/{2}]\t'
                      'Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.value:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.value:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.value:.3f} ({top5.avg:.3f})\t'.format(
                       epoch, (i+1), train_batches_num, batch_time=batch_time, loss=losses, top1=top1, top5=top5))
            print(string)
            fd.write(string + '\n')
            fd.close()

    fd = open(record_file, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.value:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.value:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.value:.3f} ({top5.avg:.3f})\t'.format(
              epoch, (i + 1), train_batches_num, batch_time=batch_time, loss=losses, top1=top1, top5=top5))
    print(string)
    fd.write(string + '\n')
    fd.close()
    val_acc.append(top1.avg)
    val_acc_top5.append(top5.avg)

    writer.add_scalar('val/loss', losses.avg, epoch)
    writer.add_scalar('val/acc@1', top1.avg, epoch)
    writer.add_scalar('val/acc@5', top5.avg, epoch)

    return top1.avg, top5.avg


def train_scale_sep(train_loader, model, fc, criterion, optimizer, epoch, writer):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for i in range(args.scale_size + 1)]
    top1 = [AverageMeter() for i in range(args.scale_size)]
    top5 = [AverageMeter() for i in range(args.scale_size)]

    train_batches_num = len(train_loader)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):
        target = target.cuda()  # TODO: this will slow down the experiment
        #input = x.cuda()  # TODO: this will slow down the experiment
        x = x.cuda()
        if args.model == 'resconenet' or args.model == 'conedensenet':
            input = []
            input.append(x)
            rate = 0.5 
            for _ in range(1, args.scale_size):
                x_tmp = nn.functional.interpolate(x, scale_factor=rate, mode='bilinear', align_corners=True) 
                rate *= 0.5
                input.append(x_tmp)
        else:
            input = x

        data_time.update(time.time() - end)
        end = time.time()
        output = model(input) # a list
        total_loss = 0
        for j, pred in enumerate(output):
            tmp_loss = criterion(pred, target)
            total_loss += tmp_loss

            prec = accuracy(pred.data, target, topk=(1,5))
            top1[j].update(prec[0].item(), x.size(0))
            top5[j].update(prec[1].item(), x.size(0))
            losses[j].update(tmp_loss.data.item(), x.size(0))

        losses[-1].update(total_loss.data.item(), x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            fd = open(record_file, 'a+')
            string = ('Epoch: [{0}][{1}/{2}]\t'
                      'Lr {lr:.4f}\t'
                      'Batch_Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
                      'Data_Time {data_time.value:.3f} ({data_time.avg:.3f})\t'.format(epoch, i+1, train_batches_num, lr=optimizer.param_groups[0]['lr'], batch_time=batch_time, data_time=data_time))
            for j in range(args.scale_size):
                string += 'Loss_s{0} {1.value:.4f} ({1.avg:.4f})\t'.format(j, losses[j])
            string += 'Loss {loss.value:.4f} ({loss.avg:.4f})\t'.format(loss=losses[-1])
            for j in range(args.scale_size):
                string += 'Prec@1_s{0} {1.value:.4f} ({1.avg:.4f})\t'.format(j, top1[j])
                string += 'Prec@5_s{0} {1.value:.4f} ({1.avg:.4f})\t'.format(j, top5[j])
            '''
                      'Loss_s0 {loss_s0.value:.4f} ({loss_s0.avg:.4f})\t'
                      'Loss_s1 {loss_s1.value:.4f} ({loss_s1.avg:.4f})\t'
                      'Loss_s2 {loss_s2.value:.4f} ({loss_s2.avg:.4f})\t'
                      'Loss_s3 {loss_s3.value:.4f} ({loss_s3.avg:.4f})\t'
                      'Loss {loss.value:.4f} ({loss.avg:.4f})\t'
                      'Prec@1_s0 {top1_s0.value:.3f} ({top1_s0.avg:.3f})\t'
                      'Prec@1_s1 {top1_s1.value:.3f} ({top1_s1.avg:.3f})\t'
                      'Prec@1_s2 {top1_s2.value:.3f} ({top1_s2.avg:.3f})\t'
                      'Prec@1_s3 {top1_s3.value:.3f} ({top1_s3.avg:.3f})\t'
                      'Prec@5_s0 {top5_s0.value:.3f} ({top5_s0.avg:.3f})\t'
                      'Prec@5_s1 {top5_s1.value:.3f} ({top5_s1.avg:.3f})\t'
                      'Prec@5_s2 {top5_s2.value:.3f} ({top5_s2.avg:.3f})\t'
                      'Prec@5_s3 {top5_s3.value:.3f} ({top5_s3.avg:.3f})\t'.format(
                epoch, i+1, train_batches_num, lr=optimizer.param_groups[0]['lr'], batch_time=batch_time, data_time=data_time, 
                loss_s0=losses[0], loss_s1=losses[1], loss_s2=losses[2], loss_s3=losses[3], loss=losses[4],
                top1_s0=top1[0], top1_s1=top1[1], top1_s2=top1[2], top1_s3=top1[3],
                top5_s0=top5[0], top5_s1=top5[1], top5_s2=top5[2], top5_s3=top5[3]))
            '''
            print(string)
            fd.write(string + '\n')
            fd.close()

    writer.add_scalar('train/loss', losses[-1].avg, epoch)
    for i in range(args.scale_size):
        writer.add_scalar('train/loss_s%d'%(i), losses[i].avg, epoch)
        writer.add_scalar('train/acc@1_s%d'%(i), top1[i].avg, epoch)
        writer.add_scalar('train/acc@5_s%d'%(i), top5[i].avg, epoch)
    '''
    writer.add_scalar('train/loss_s0', losses[0].avg, epoch)
    writer.add_scalar('train/loss_s1', losses[1].avg, epoch)
    writer.add_scalar('train/loss_s2', losses[2].avg, epoch)
    writer.add_scalar('train/loss_s3', losses[3].avg, epoch)
    writer.add_scalar('train/loss', losses[4].avg, epoch)
    writer.add_scalar('train/acc@1_s0', top1[0].avg, epo,ch)
    writer.add_scalar('train/acc@1_s1', top1[1].avg, epoch)
    writer.add_scalar('train/acc@1_s2', top1[2].avg, epoch)
    writer.add_scalar('train/acc@1_s3', top1[3].avg, epoch)
    writer.add_scalar('train/acc@5_s0', top5[0].avg, epo,ch)
    writer.add_scalar('train/acc@5_s1', top5[1].avg, epoch)
    writer.add_scalar('train/acc@5_s2', top5[2].avg, epoch)
    writer.add_scalar('train/acc@5_s3', top5[3].avg, epoch)
    '''

def validate_scale_sep(val_loader, model, criterion, epoch, writer):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = [AverageMeter() for i in range(args.scale_size + 1)]
    top1 = [AverageMeter() for i in range(args.scale_size)]
    top5 = [AverageMeter() for i in range(args.scale_size)]

    train_batches_num = len(val_loader)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (x, target) in enumerate(val_loader):
        target = target.cuda()
        x = x.cuda()

        # compute output
        with torch.no_grad():
            if args.model == 'resconenet' or args.model == 'conedensenet':
                input = []
                input.append(x)
                rate = 0.5 
                for _  in range(1, args.scale_size):
                    x_tmp = nn.functional.interpolate(x, scale_factor=rate, mode='bilinear', align_corners=True) 
                    rate *= 0.5
                    input.append(x_tmp)
            else:
                input = x

            output = model(input) # a list
            total_loss = 0
            for j, pred in enumerate(output):
                tmp_loss = criterion(pred, target)
                total_loss += tmp_loss

                prec = accuracy(pred.data, target, topk=(1,5))
                top1[j].update(prec[0].item(), x.size(0))
                top5[j].update(prec[1].item(), x.size(0))
                losses[j].update(tmp_loss.data.item(), x.size(0))

            losses[-1].update(total_loss.data.item(), x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            fd = open(record_file, 'a+')
            string = ('Test: [{0}][{1}/{2}]\t'
                      'Batch_Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'.format(epoch, i+1, train_batches_num, batch_time=batch_time))
            for j in range(args.scale_size):
                string += 'Loss_s{0} {1.value:.4f} ({1.avg:.4f})\t'.format(j, losses[j])
            string += 'Loss {loss.value:.4f} ({loss.avg:.4f})\t'.format(loss=losses[-1])
            for j in range(args.scale_size):
                string += 'Prec@1_s{0} {1.value:.4f} ({1.avg:.4f})\t'.format(j, top1[j])
                string += 'Prec@5_s{0} {1.value:.4f} ({1.avg:.4f})\t'.format(j, top5[j])
            print(string)
            fd.write(string + '\n')
            fd.close()

    fd = open(record_file, 'a+')
    string = ('Test: [{0}][{1}/{2}]\t'
              'Batch_Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'.format(epoch, i+1, train_batches_num, batch_time=batch_time))
    for j in range(args.scale_size):
        string += 'Loss_s{0} {1.value:.4f} ({1.avg:.4f})\t'.format(j, losses[j])
    string += 'Loss {loss.value:.4f} ({loss.avg:.4f})\t'.format(loss=losses[-1])
    for j in range(args.scale_size):
        string += 'Prec@1_s{0} {1.value:.4f} ({1.avg:.4f})\t'.format(j, top1[j])
        string += 'Prec@5_s{0} {1.value:.4f} ({1.avg:.4f})\t'.format(j, top5[j])
    print(string)
    fd.write(string + '\n')
    fd.close()
    val_acc.append(max([top1[i].avg for i in range(args.scale_size)]))
    val_acc_top5.append(max([top5[i].avg for i in range(args.scale_size)]))

    writer.add_scalar('val/loss', losses[-1].avg, epoch)
    for i in range(args.scale_size):
        writer.add_scalar('val/loss_s%d'%(i), losses[i].avg, epoch)
        writer.add_scalar('val/acc@1_s%d'%(i), top1[i].avg, epoch)
        writer.add_scalar('val/acc@5_s%d'%(i), top5[i].avg, epoch)

    return val_acc[-1], val_acc_top5[-1]
if __name__ == '__main__':
    main()
