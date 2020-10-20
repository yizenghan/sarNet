# import moxing as mox
# mox.file.shift('os', 'mox')

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
parser.add_argument('--patch_groups3', type=int, default=1)
parser.add_argument('--patch_groups4', type=int, default=1)

parser.add_argument('--target_rate1', type=float, default=0.5)
parser.add_argument('--target_rate2', type=float, default=0.5)
parser.add_argument('--target_rate3', type=float, default=0.5)
parser.add_argument('--target_rate4', type=float, default=0.5)

parser.add_argument('--lambda_act1', type=float, default=1.0)
parser.add_argument('--lambda_act2', type=float, default=1.0)
parser.add_argument('--lambda_act3', type=float, default=1.0)
parser.add_argument('--lambda_act4', type=float, default=1.0)

parser.add_argument('--use_ls1', type=int, default=0)
parser.add_argument('--use_ls2', type=int, default=0)
parser.add_argument('--use_ls3', type=int, default=0)
parser.add_argument('--use_ls4', type=int, default=0)

parser.add_argument('--use_amp1', type=int, default=0)
parser.add_argument('--use_amp2', type=int, default=0)
parser.add_argument('--use_amp3', type=int, default=0)
parser.add_argument('--use_amp4', type=int, default=0)

parser.add_argument('--warmup1', type=int, default=0)
parser.add_argument('--warmup2', type=int, default=0)
parser.add_argument('--warmup3', type=int, default=0)
parser.add_argument('--warmup4', type=int, default=0)


parser.add_argument('--t0_1', type=float, default=0.5)
parser.add_argument('--t0_2', type=float, default=0.5)
parser.add_argument('--t0_3', type=float, default=0.5)
parser.add_argument('--t0_4', type=float, default=0.5)

parser.add_argument('--alpha1', type=int, default=2)
parser.add_argument('--alpha2', type=int, default=2)
parser.add_argument('--alpha3', type=int, default=2)
parser.add_argument('--alpha4', type=int, default=2)

parser.add_argument('--dynamic_rate1', type=int, default=0)
parser.add_argument('--dynamic_rate2', type=int, default=0)
parser.add_argument('--dynamic_rate3', type=int, default=0)
parser.add_argument('--dynamic_rate4', type=int, default=0)


parser.add_argument('--temp_scheduler1', default='exp', type=str)
parser.add_argument('--temp_scheduler2', default='exp', type=str)
parser.add_argument('--temp_scheduler3', default='exp', type=str)
parser.add_argument('--temp_scheduler4', default='exp', type=str)

parser.add_argument('--base_scale1', type=int, default=2)
parser.add_argument('--base_scale2', type=int, default=2)
parser.add_argument('--base_scale3', type=int, default=2)
parser.add_argument('--base_scale4', type=int, default=2)

parser.add_argument('--optimize_rate_begin_epoch1', type=int, default=55)
parser.add_argument('--optimize_rate_begin_epoch2', type=int, default=55)
parser.add_argument('--optimize_rate_begin_epoch3', type=int, default=55)
parser.add_argument('--optimize_rate_begin_epoch4', type=int, default=55)

args = parser.parse_args()

if args.use_amp1 + args.use_amp2 + args.use_amp3 + args.use_amp4 > 0 :
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
                    warmup=1,
                    test_code=0,
                    temp_scheduler = 'exp',
                    gpu='0,1,2,3'):
    ta_str = str(target_rate)
    str_t0 = str(t0).replace('.','_')
    str_lambda = str(lambda_act).replace('.','_')
    train_url = f'{train_url_base}_baseAlpha_preact_g{patch_groups}_alpha{alpha}_t0_{str_t0}_target{ta_str[-1]}_optimizeFromEpoch{optimize_rate_begin_epoch}_lambda_{str_lambda}_ls{use_ls}_amp{use_amp}_warmup{warmup}_dynamicRate{dynamic_rate}/'    
    cmd = f'CUDA_VISIBLE_DEVICES={gpu} python main_sar.py  --no_train_on_cloud \
            --train_url {train_url} \
            --data_url {data_url} \
            --config configs/{config}.py \
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
                warmup=self.warmup,
                test_code=self.test_code,
                temp_scheduler=self.temp_scheduler,
                gpu=self.gpu)
        os.system(cmd)


config1 = f'_baseAlpha_pre50_g{args.patch_groups1}a{args.alpha1}s{args.base_scale1}_ls_warmup'
config2 = f'_baseAlpha_pre50_g{args.patch_groups2}a{args.alpha2}s{args.base_scale2}_ls_warmup'
config3 = f'_baseAlpha_pre50_g{args.patch_groups3}a{args.alpha3}s{args.base_scale3}_ls_warmup'
config4 = f'_baseAlpha_pre50_g{args.patch_groups4}a{args.alpha4}s{args.base_scale4}_ls_warmup'

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
                temp_scheduler=args.temp_scheduler1,
                test_code=0,
                gpu='0,1')
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
                temp_scheduler=args.temp_scheduler2,
                test_code=0,
                gpu='2,3')
t2.start()

t3 = myThread(threadID=3,
                config=config3, 
                patch_groups=args.patch_groups3,
                alpha=args.alpha3,
                train_url_base=args.train_url, 
                data_url=args.data_url,
                dist_url=f'tcp://127.0.0.1:30077',
                lambda_act=args.lambda_act3,
                dynamic_rate=args.dynamic_rate3,
                t0=args.t0_3,
                target_rate=args.target_rate3,
                optimize_rate_begin_epoch=args.optimize_rate_begin_epoch3,
                use_amp=args.use_amp3,
                use_ls = args.use_ls3,
                warmup=args.warmup3,
                temp_scheduler=args.temp_scheduler3,
                test_code=0,
                gpu='4,5')
t3.start()

t4 = myThread(threadID=4,
            config=config4, 
            train_url_base=args.train_url, 
            patch_groups=args.patch_groups4,
            alpha=args.alpha4,
            data_url=args.data_url,
            dist_url=f'tcp://127.0.0.1:30078',
            lambda_act=args.lambda_act4,
            dynamic_rate=args.dynamic_rate4,
            t0=args.t0_4,
            target_rate=args.target_rate4,
            optimize_rate_begin_epoch=args.optimize_rate_begin_epoch4,
            use_amp=args.use_amp4,
            use_ls = args.use_ls4,
            warmup=args.warmup4,
            temp_scheduler=args.temp_scheduler4,
            test_code=0,
            gpu='6,7')
t4.start()