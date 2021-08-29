import json
import requests
from requests_ntlm import HttpNtlmAuth
import copy
import argparse
import os
from tempfile import NamedTemporaryFile
import pprint
import urllib3
import time
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='hack philly vc')
    parser.add_argument('--name', type=str, help='Job Name', default='test_mmcls')
    parser.add_argument('--user', type=str, help='hack user name', default='{alias}')
    parser.add_argument('--VcId', type=str, help='VcId', default='nextmsra')
    parser.add_argument('--ClusterId', type=str, help='ClusterId', default='eu1')
    parser.add_argument('--gpus', type=int, help='Gpu numbers', default=8)
    parser.add_argument('--path', type=str, help='python scrip path', default='tools/train.py')
    parser.add_argument('--branch', type=str, help='git branch to pull', default='master')

    return parser.parse_known_args()


args, rest = parse_args()
extra_args = ' '.join(rest)

submitted_jobId = []
cfgs= [
    'configs/mask_rcnn_r50_fpn_1x_philly.py',
]
pprint.pprint(cfgs)

gpus8_clusters = ['sc3', 'philly-prod-cy4']

submit_url = 'https://philly/api/v2/submit'
submit_headers = {'Content-Type': 'application/json'}
pwd = {'{alias}': '{passwd}'}

ClusterId = args.ClusterId
VcId = args.VcId
path = args.path
name = os.path.splitext(os.path.basename(path))[0] + '_' + args.name
user = args.user
branch = args.branch
auth = '{git cridential}'
gpus = args.gpus
philly_auth = HttpNtlmAuth(user, pwd[user])
submit_data = {}
submit_data["ClusterId"] = ClusterId
submit_data["VcId"] = VcId
submit_data["JobName"] = name
submit_data["UserName"] = user
submit_data["BuildId"] = 0
submit_data["ToolType"] = None
submit_data["Inputs"] = [{ "Name": "dataDir", "Path": "/hdfs/nextmsra/{alias}/" },]
submit_data["Outputs"] =[]
submit_data["IsDebug"] = False
submit_data["RackId"] = "anyConnected"
submit_data["MinGPUs"] = gpus
submit_data["PrevModelPath"] = None
submit_data["ExtraParams"] = "--cfg {0} --path {1} --branch {2} --auth {3} {4}"
submit_data["SubmitCode"] = "p"
submit_data["IsMemCheck"] = False
submit_data["IsCrossRack"] = False
submit_data["Registry"] = "phillyregistry.azurecr.io"
submit_data["Repository"] = "philly/jobs/custom/pytorch"
submit_data["Tag"] = "pytorch1.0.0-py36-cuda9-mpi-nccl-hvd-apex-video"
submit_data["OneProcessPerContainer"] = False
submit_data["NumOfContainers"] = str(gpus//8) if ClusterId in gpus8_clusters else str(gpus//4)
submit_data["dynamicContainerSize"] = False
if 'resrchprojvc' not in VcId:
    submit_data["Queue"] = "bonus"
for cfg in cfgs:

    data = submit_data.copy()
    data["ConfigFile"] = "/hdfs/nextmsra/{alias}/run_mmcls_on_philly_dist.py"
    if ClusterId in ['gcr', 'rr1', 'rr2', 'cam', 'philly-prod-cy4']:
        data["CustomMPIArgs"] = "env CUDA_CACHE_DISABLE=1 NCCL_SOCKET_IFNAME=ib0 NCCL_DEBUG=INFO OMP_NUM_THREADS=2"
    else:
        data["CustomMPIArgs"] = "env CUDA_CACHE_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_DEBUG=INFO OMP_NUM_THREADS=2"

    data['ExtraParams'] = data['ExtraParams'].format(cfg, path, branch, auth, extra_args)

    r = requests.post(json=data, url=submit_url, auth=philly_auth, headers=submit_headers, verify=False)
    if r.status_code == 200:
        res = r.json()
        print("submit {0} to {1} successfully".format(data['JobName'], ClusterId))
        print(res)
        res_dict = {}
        res_dict["JobName"] = data["JobName"]
        res_dict["AppId"] = res['jobId']
        res_dict["cfg"] = cfg
        res_dict["Link"] = "https://philly/#/job/{}/{}/{}".format(ClusterId, VcId, res['jobId'][12:])
        submitted_jobId.append(res_dict)
    else:
        print(r)
        print('submit failed with status_code {}'.format(r.status_code))

pprint.pprint(submitted_jobId)
