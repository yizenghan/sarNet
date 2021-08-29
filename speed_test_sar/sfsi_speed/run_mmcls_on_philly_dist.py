import argparse
import os
import time

def ompi_rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_RANK') or 0)


def ompi_size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)


print('rank: {}'.format(ompi_rank()))
print('env:')
print(os.environ)
os.system("ls")
parser = argparse.ArgumentParser(description='Helper run')
# general
parser.add_argument('--data', help='data input', type=str, default='.')
parser.add_argument('--auth', help='auth', required=True, type=str)
parser.add_argument('--path', help='path', required=True, type=str)
parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
parser.add_argument('--branch', help="branch of code", type=str, default='master')
args, rest = parser.parse_known_args()
print(args)
extra_args = ' '.join(rest)
print('extra_args: {}'.format(extra_args))

is_master = ompi_rank() == 0 or ompi_size() == 1
clone_dir = os.path.join(os.environ['HOME'], 'mmclassification')
done_signal_path = os.path.join(os.environ['HOME'], 'done.txt')
master_init_done = os.path.exists(clone_dir) and os.path.exists(done_signal_path)
torch_model_zoo_path = '/hdfs/public/v-yuca/torch_model_zoo'
os.environ['TORCH_MODEL_ZOO'] = torch_model_zoo_path
if not is_master:
    while not master_init_done:
        print('rank {} waiting for git clone'.format(ompi_rank()))
        master_init_done = os.path.exists(clone_dir) and os.path.exists(done_signal_path)
        time.sleep(10.0)
elif master_init_done:
    print('mmclassification master already fully inited')
else:
    os.system("git clone --recursive https://{0}@github.com/stupidZZ/msra_mmclassification -b {1} $HOME/mmclassification".format(args.auth, args.branch))
    # only master need to install package
    os.chdir(os.path.expanduser('~/mmclassification'))
    os.system('./init_philly.sh')
    os.system('echo done > $HOME/done.txt')
    os.system('ls $HOME')
os.chdir(os.path.expanduser('~/mmclassification'))
if is_master:
    os.system("ls")
os.system("python {0} {1} --launcher {2} {3} --validate".format(args.path, args.cfg, 'mpi', extra_args))
