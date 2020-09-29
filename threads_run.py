import moxing as mox
mox.file.shift('os', 'mox')

import os
import argparse
import threading
import warnings

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

# import tmp
# import main_sar

# mox.file.copy('sarNet/main_sar.py', '/cache/main_sar.py')

# mox.file.copy_parallel('sarNet/double_checked_models/', '/cache/double_checked_models')
# mox.file.copy_parallel('sarNet/models/', '/cache/models')
# mox.file.copy('sarNet/optimizer.py', '/cache/optimizer.py')
# mox.file.copy('sarNet/criterion.py', '/cache/criterion.py')
# mox.file.copy('sarNet/transform.py', '/cache/transform.py')
# mox.file.copy('sarNet/scheduler.py', '/cache/scheduler.py')
# mox.file.copy('sarNet/hyperparams.py', '/cache/hyperparams.py')
# mox.file.copy('sarNet/config.py', '/cache/config.py')
# mox.file.copy('sarNet/utils.py', '/cache/utils.py')

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

parser = argparse.ArgumentParser(description='PyTorch SARNet')
parser.add_argument('--data_url', type=str, metavar='DIR', default='/data/dataset/CLS-LOC/',
                    help='path to dataset')
parser.add_argument('--train_url', type=str, metavar='PATH', default='./log/test/',
                    help='path to save result and checkpoint (default: results/savedir)')
parser.add_argument('--init_method', type=str, default='',
                    help='an argument needed in huawei cloud, but i do not know its usage')
args = parser.parse_args()

def multi_thread_run(config='_sarResNet50_g1_blConfig', 
                    train_url_base='', 
                    data_url='',
                    dist_url='127.0.0.1:10001',
                    lambda_act=1.0,
                    t0=0.5,
                    target_rate=0.5,
                    optimize_rate_begin_epoch=55,
                    use_amp=1,
                    test_code=0,
                    gpu='0,1,2,3'):
    ta_str = str(target_rate)
    train_url = f'{train_url_base}target{ta_str[-1]}/'
    cmd = f'CUDA_VISIBLE_DEVICES={gpu} python sarNet/main_sar.py   \
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
    def __init__(self, threadID=1, config='_sarResNet50_g1_blConfig', 
                train_url_base='', 
                data_url='',
                dist_url='127.0.0.1:10001',
                lambda_act=1.0,
                t0=0.5,
                target_rate=0.5,
                optimize_rate_begin_epoch=55,
                use_amp=1,
                test_code=0,
                gpu='0,1,2,3'):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.config = config
        self.train_url_base = train_url_base
        self.data_url = data_url
        self.dist_url = dist_url
        self.lambda_act = lambda_act
        self.t0 = t0
        self.target_rate = target_rate
        self.optimize_rate_begin_epoch = optimize_rate_begin_epoch
        self.use_amp = use_amp
        self.test_code = test_code
        self.gpu = gpu

    def run(self):
        print ("start" + str(self.threadID))
        cmd = multi_thread_run(config=self.config, 
                train_url_base=self.train_url_base, 
                data_url=self.data_url,
                dist_url=self.dist_url,
                lambda_act=self.lambda_act,
                t0=self.t0,
                target_rate=self.target_rate,
                optimize_rate_begin_epoch=self.optimize_rate_begin_epoch,
                use_amp=self.use_amp,
                test_code=self.test_code,
                gpu=self.gpu)
        os.system(cmd)

# cmdddd = multi_thread_run(config='_sarResNet50_g1_blConfig', 
#                 train_url_base=args.train_url, 
#                 data_url=args.data_url,
#                 dist_url=f'tcp://127.0.0.1:30077',
#                 lambda_act=1.0,
#                 t0=0.5,
#                 target_rate=0.5,
#                 optimize_rate_begin_epoch=55,
#                 use_amp=0,
#                 test_code=0,
#                 gpu='0,1,2,3')
# os.system(cmdddd)


# gpu = ['0,1,2,3', '4,5,6,7']

t1 = myThread(threadID=1,
                config='_sarResNet50_g1_blConfig', 
                train_url_base=args.train_url, 
                data_url=args.data_url,
                dist_url=f'tcp://127.0.0.1:30077',
                lambda_act=1.0,
                t0=0.5,
                target_rate=0.5,
                optimize_rate_begin_epoch=55,
                use_amp=1,
                test_code=0,
                gpu='0,1,2,3')
t1.start()

t2 = myThread(threadID=2,
                config='_sarResNet50_g1_blConfig', 
                train_url_base=args.train_url, 
                data_url=args.data_url,
                dist_url=f'tcp://127.0.0.1:30078',
                lambda_act=1.0,
                t0=0.5,
                target_rate=0.7,
                optimize_rate_begin_epoch=55,
                use_amp=1,
                test_code=0,
                gpu='4,5,6,7')
t2.start()