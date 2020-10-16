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
parser.add_argument('--target_rate1', type=float, default=0.5)
parser.add_argument('--target_rate2', type=float, default=0.5)
parser.add_argument('--lambda_act1', type=float, default=1.0)
parser.add_argument('--lambda_act2', type=float, default=1.0)

parser.add_argument('--use_ls1', type=int, default=0)
parser.add_argument('--use_ls2', type=int, default=0)
parser.add_argument('--use_amp1', type=int, default=0)
parser.add_argument('--use_amp2', type=int, default=0)
parser.add_argument('--warmup1', type=int, default=0)
parser.add_argument('--warmup2', type=int, default=0)

parser.add_argument('--width1', type=float, default=1.0)
parser.add_argument('--width2', type=float, default=1.0)

parser.add_argument('--t0_1', type=float, default=0.5)
parser.add_argument('--t0_2', type=float, default=0.5)

parser.add_argument('--alpha1', type=int, default=1)
parser.add_argument('--alpha2', type=int, default=1)

parser.add_argument('--self_mask1', type=int, default=0)
parser.add_argument('--self_mask2', type=int, default=0)

parser.add_argument('--dynamic_rate1', type=int, default=0)
parser.add_argument('--dynamic_rate2', type=int, default=0)

parser.add_argument('--stage2', type=int, default=0)

parser.add_argument('--temp_scheduler1', default='exp', type=str)
parser.add_argument('--temp_scheduler2', default='exp', type=str)

parser.add_argument('--base_scale1', type=int, default=2)
parser.add_argument('--base_scale2', type=int, default=2)

parser.add_argument('--optimize_rate_begin_epoch1', type=int, default=55)
parser.add_argument('--optimize_rate_begin_epoch2', type=int, default=55)

args = parser.parse_args()

args.self_mask1 = True if args.self_mask1 > 0 else False
args.self_mask2 = True if args.self_mask2 > 0 else False

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
                    alpha=1,
                    data_url='',
                    dist_url='127.0.0.1:10001',
                    lambda_act=1.0,
                    t0=0.5,
                    target_rate=0.5,
                    optimize_rate_begin_epoch=55,
                    use_amp=1,
                    use_ls=1,
                    dynamic_rate=0,
                    width=1.0,
                    warmup=True,
                    test_code=0,
                    temp_scheduler = 'exp',
                    gpu='0,1,2,3'):
    ta_str = str(target_rate)
    ls = 1 if use_ls else 0
    wmup = 1 if warmup else 0
    if width == 0.5:
        wd = '05'
    elif width == 0.75:
        wd = '075'
    else:
        wd = '1'
    str_t0 = str(t0).replace('.','_')
    str_lambda = str(lambda_act).replace('.','_')
    train_url = f'{train_url_base}width{wd}_g{patch_groups}_alpha{alpha}_t0_{str_t0}_target{ta_str[-1]}_lambda_{str_lambda}_ls{ls}_amp{use_amp}_warmup{wmup}_dynamicRate{dynamic_rate}/'    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu} python sarNet/main_sar.py   \
            --train_url {train_url} \
            --data_url {data_url} \
            --config obs://d-cheap-net/hyz/sarNet/configs/{config}.py \
            --dist_url {dist_url} \
            --dynamic_rate {dynamic_rate} \
            --lambda_act {lambda_act} \
            --t0 {t0} --temp_scheduler {temp_scheduler} \
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
                alpha=1,
                dynamic_rate=0,
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
                temp_scheduler = 'exp',
                gpu='0,1,2,3'):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.config = config
        self.patch_groups = patch_groups
        self.alpha = alpha
        self.train_url_base = train_url_base
        self.data_url = data_url
        self.dist_url = dist_url
        self.lambda_act = lambda_act
        self.dynamic_rate=dynamic_rate
        self.t0 = t0
        self.target_rate = target_rate
        self.optimize_rate_begin_epoch = optimize_rate_begin_epoch
        self.use_amp = use_amp
        self.width = width
        self.test_code = test_code
        self.gpu = gpu
        self.use_ls = use_ls
        self.warmup = warmup
        self.temp_scheduler = temp_scheduler

    def run(self):
        print ("start" + str(self.threadID))
        cmd = multi_thread_run(config=self.config, 
                train_url_base=self.train_url_base, 
                patch_groups=self.patch_groups,
                alpha=self.alpha,
                data_url=self.data_url,
                dist_url=self.dist_url,
                lambda_act=self.lambda_act,
                dynamic_rate=self.dynamic_rate,
                t0=self.t0,
                target_rate=self.target_rate,
                optimize_rate_begin_epoch=self.optimize_rate_begin_epoch,
                use_amp=self.use_amp,
                use_ls=self.use_ls,
                width = self.width,
                warmup=self.warmup,
                test_code=self.test_code,
                temp_scheduler=self.temp_scheduler,
                gpu=self.gpu)
        os.system(cmd)


config1 = f'_sarResNet50_g{args.patch_groups1}a{args.alpha1}s{args.base_scale1}_ls_warmup'
config2 = f'_sarResNet50_g{args.patch_groups2}a{args.alpha2}s{args.base_scale2}_ls_warmup'

# if args.stage2 > 0:
#     config1 += '_s2'
#     config2 += '_s2'

# if args.self_mask1:
#     config1 += f'_selfmask_a{args.alpha1}b1_blConfig'
# else:
#     config1 += f'_a{args.alpha1}b1_blConfig'

# if args.self_mask2:
#     config2 += f'_selfmask_a{args.alpha2}b1_blConfig'
# else:
#     config2 += f'_a{args.alpha2}b1_blConfig'

# if args.use_ls1:
#     config1 += '_ls'
# if args.use_ls2:
#     config2 += '_ls'

# if args.warmup1:
#     config1 += '_warmup'
# if args.warmup2:
#     config2 += '_warmup'

t1 = myThread(threadID=1,
                config=config1, 
                patch_groups=args.patch_groups1,
                alpha=args.alpha1,
                train_url_base=args.train_url, 
                data_url=args.data_url,
                dist_url=f'tcp://127.0.0.1:30076',
                lambda_act=args.lambda_act1,
                dynamic_rate=args.dynamic_rate1,
                t0=args.t0_1,
                target_rate=args.target_rate1,
                optimize_rate_begin_epoch=args.optimize_rate_begin_epoch1,
                use_amp=args.use_amp1,
                use_ls = args.use_ls1,
                warmup=args.warmup1,
                width = args.width1,
                temp_scheduler=args.temp_scheduler1,
                test_code=0,
                gpu='0,1,2,3')
t1.start()

t2 = myThread(threadID=2,
                config=config2, 
                train_url_base=args.train_url, 
                patch_groups=args.patch_groups2,
                alpha=args.alpha2,
                data_url=args.data_url,
                dist_url=f'tcp://127.0.0.1:30075',
                lambda_act=args.lambda_act2,
                dynamic_rate=args.dynamic_rate2,
                t0=args.t0_2,
                target_rate=args.target_rate2,
                optimize_rate_begin_epoch=args.optimize_rate_begin_epoch2,
                use_amp=args.use_amp2,
                use_ls = args.use_ls2,
                warmup=args.warmup2,
                width = args.width2,
                temp_scheduler=args.temp_scheduler2,
                test_code=0,
                gpu='4,5,6,7')
t2.start()