import moxing as mox
mox.file.shift('os', 'mox')

import os
import argparse
import threading
import warnings

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'


parser = argparse.ArgumentParser(description='PyTorch SARNet')
parser.add_argument('--data_url', type=str, metavar='DIR', default='/data/dataset/CLS-LOC/',
                    help='path to dataset')
parser.add_argument('--train_url', type=str, metavar='PATH', default='./log/test/',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--init_method', type=str, default='',
                    help='an argument needed in huawei cloud, but i do not know its usage')

parser.add_argument('--patch_groups1', type=int, default=1)
parser.add_argument('--patch_groups2', type=int, default=1)

parser.add_argument('--use_ls1', type=int, default=0)
parser.add_argument('--use_ls2', type=int, default=0)
parser.add_argument('--use_amp1', type=int, default=0)
parser.add_argument('--use_amp2', type=int, default=0)
parser.add_argument('--warmup1', type=int, default=0)
parser.add_argument('--warmup2', type=int, default=0)

parser.add_argument('--width1', type=float, default=1.0)
parser.add_argument('--width2', type=float, default=1.0)

parser.add_argument('--alpha1', type=int, default=1)
parser.add_argument('--alpha2', type=int, default=1)

parser.add_argument('--beta1', type=int, default=1)
parser.add_argument('--beta2', type=int, default=1)

args = parser.parse_args()

if args.use_amp1 > 0 or args.use_amp2 > 0:
    try:
        from apex import amp
        from apex.parallel import DistributedDataParallel as DDP
        from apex.parallel import convert_syncbn_model
        has_apex = True
    except ImportError:
        mox.file.copy_parallel('sarNet/apex-master/', '/cache/apex-master')
        os.system('pip --default-timeout=100 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" /cache/apex-master')
        from apex import amp
        from apex.parallel import DistributedDataParallel as DDP
        from apex.parallel import convert_syncbn_model
        has_apex = True
        print('successfully install apex')

args.use_ls1 = True if args.use_ls1 > 0 else False
args.use_ls2 = True if args.use_ls2 > 0 else False
args.warmup1 = True if args.warmup1 > 0 else False
args.warmup2 = True if args.warmup2 > 0 else False

def multi_thread_run(config='_sarResNet50_g1_blConfig', 
                    train_url_base='',
                    patch_groups=1, 
                    data_url='',
                    dist_url='127.0.0.1:10001',
                    lambda_act=1.0,
                    t0=0.5,
                    target_rate=0.5,
                    optimize_rate_begin_epoch=55,
                    use_amp=1,
                    use_ls=1,
                    width=1.0,
                    warmup=True,
                    test_code=0,
                    gpu='0,1,2,3'):
    ls = 1 if use_ls else 0
    wmup = 1 if warmup else 0
    train_url = f'{train_url_base}mask1_g{patch_groups}_alpha2_beta2_ls{ls}_amp{use_amp}_warmup{wmup}/'
    cmd = f'CUDA_VISIBLE_DEVICES={gpu} python sarNet/main_sar_nomask.py   \
            --train_url {train_url} \
            --data_url {data_url} \
            --config obs://d-cheap-net-shanghai/hanyz/sarNet/configs/{config}.py \
            --dist_url {dist_url} \
            --lambda_act {lambda_act} \
            --t0 {t0} \
            --target_rate {target_rate} \
            --optimize_rate_begin_epoch {optimize_rate_begin_epoch} \
            --use_amp {use_amp} \
            --test_code {test_code}'
    # cmd = 'python /cache/tmp.py'
    print(cmd)
    return cmd

class myThread(threading.Thread):
    def __init__(self, threadID=1, 
                config='_sarResNet50_g1_blConfig', 
                train_url_base='', 
                patch_groups=1,
                data_url='',
                dist_url='127.0.0.1:10001',
                lambda_act=1.0,
                t0=0.5,
                target_rate=0.5,
                optimize_rate_begin_epoch=55,
                use_amp=1,
                use_ls=1,
                width=1.0,
                warmup=True,
                test_code=0,
                gpu='0,1,2,3'):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.config = config
        self.patch_groups = patch_groups
        self.train_url_base = train_url_base
        self.data_url = data_url
        self.dist_url = dist_url
        self.lambda_act = lambda_act
        self.t0 = t0
        self.target_rate = target_rate
        self.optimize_rate_begin_epoch = optimize_rate_begin_epoch
        self.use_amp = use_amp
        self.width = width
        self.test_code = test_code
        self.gpu = gpu
        self.use_ls = use_ls
        self.warmup = warmup

    def run(self):
        print ("start" + str(self.threadID))
        cmd = multi_thread_run(config=self.config, 
                train_url_base=self.train_url_base, 
                patch_groups=self.patch_groups,
                data_url=self.data_url,
                dist_url=self.dist_url,
                lambda_act=self.lambda_act,
                t0=self.t0,
                target_rate=self.target_rate,
                optimize_rate_begin_epoch=self.optimize_rate_begin_epoch,
                use_amp=self.use_amp,
                use_ls=self.use_ls,
                width = self.width,
                warmup=self.warmup,
                test_code=self.test_code,
                gpu=self.gpu)
        os.system(cmd)


config1 = f'_sarResNet50_mask1_g1_alpha{args.alpha1}_beta{args.beta1}_blConfig'
config2 = f'_sarResNet50_mask1_g1_alpha{args.alpha2}_beta{args.beta2}_blConfig'

if args.use_ls1:
    config1 += '_ls'
if args.use_ls2:
    config2 += '_ls'

t1 = myThread(threadID=1,
                config=config1, 
                patch_groups=args.patch_groups1,
                train_url_base=args.train_url, 
                data_url=args.data_url,
                dist_url=f'tcp://127.0.0.1:30076',
                lambda_act=0.0,
                t0=0.5,
                target_rate=1.0,
                optimize_rate_begin_epoch=55,
                use_amp=args.use_amp1,
                use_ls = args.use_ls1,
                warmup=args.warmup1,
                width = args.width1,
                test_code=0,
                gpu='0,1,2,3')
t1.start()

t2 = myThread(threadID=2,
                config=config2, 
                train_url_base=args.train_url, 
                patch_groups=args.patch_groups2,
                data_url=args.data_url,
                dist_url=f'tcp://127.0.0.1:30075',
                lambda_act=0.0,
                t0=0.5,
                target_rate=1.0,
                optimize_rate_begin_epoch=55,
                use_amp=args.use_amp2,
                use_ls = args.use_ls2,
                warmup=args.warmup2,
                width = args.width2,
                test_code=0,
                gpu='4,5,6,7')
t2.start()