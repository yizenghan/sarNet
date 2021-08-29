import argparse
from collections import OrderedDict
import numpy as np
import torch
import mmcv
from mmcv.runner import get_dist_info, load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel, MMDistributedDataParallel

from mmcls.datasets import get_dataset
from mmcls.models import build_classifier
from mmcls.datasets import build_dataloader
from mmcls.utils import is_host, Sparsity
from flops import CalculationCalculator

def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def run_test(model, data_set, cfg):
    model.eval()
    data_loader = build_dataloader(
            data_set,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus = len(model.device_ids),
            dist=False)
    prog_bar = mmcv.ProgressBar(len(data_loader))

    top_1_acc = []
    top_5_acc = []
    for i, data_batch in enumerate(data_loader):
        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False)

            top_1_acc.append(outputs['log_vars']['acc_top1'])
            top_5_acc.append(outputs['log_vars']['acc_top5'])
        prog_bar.update()

    print()
    print("top_1_acc: {}".format(np.array(top_1_acc).mean()))
    print("top_5_acc: {}".format(np.array(top_5_acc).mean()))
    # for item in Sparsity:
    #     print('{}:'.format(item))
    #     print(Sparsity[item])
    #     #print('{}: {}', format(item, Sparsity[item]))
    #
    # print('Approx overall: {}'.format(np.array([list(x.values()) for x in list(Sparsity.values())]).flatten().mean()))
    print('Block Index | Conv1 | Conv2')
    Conv1 = 0.0
    Conv2 = 0.0
    for key, value in Sparsity.items():
        print(f'{key} | {value["Conv1"]:.4f} | {value["Conv2"]:.4f}')
        Conv1 += value['Conv1']
        Conv2 += value['Conv2']
    print(f'Overall | {Conv1 / len(Sparsity):.4f} | {Conv2 / len(Sparsity):.4f}')
    Weighted = Conv1 * 0.5 + Conv2 * 0.5
    print(f'Weighted | {Weighted / len(Sparsity):.4f}')

    CalculationCalculator(model=f'res{cfg.model.backbone.depth}', sparsity_dict=Sparsity).get_cal()


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=8, type=int, help='GPU number used for testing')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    val_dataset = get_dataset(cfg.data.val)

    model = build_classifier(
        cfg.model, test_cfg=cfg.test_cfg)

    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=range(args.gpus)).cuda()
    run_test(model, val_dataset, cfg)

if __name__ == '__main__':
    main()
