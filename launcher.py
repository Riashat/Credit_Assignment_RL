import os
import itertools
import numpy as np
import subprocess
import argparse
parser = argparse.ArgumentParser()

def grid_search(args_vals):
    """ arg_vals: a list of lists, each one of format (argument, list of possible values) """
    lists = []
    for arg_vals in args_vals:
        arg, vals = arg_vals
        ll = []
        for val in vals:
            ll.append("-" + arg + " " + str(val) + " ")
        lists.append(ll)
    return ["".join(item) for item in itertools.product(*lists)]


parser = argparse.ArgumentParser()

parser.add_argument('--experiments', type=int, default=5)
parser.add_argument('--policy', type=str, default="a2c")
parser.add_argument('--lr', type=float, default=7e-4, help="learning rate")

parser.add_argument('--actor_lr', type=float, default=1e-4, help='actor learning rate (default: 1e-4)')
parser.add_argument('--critic_lr', type=float, default=1e-3, help='critic learning rate (default: 1e-3)')

parser.add_argument('--batch_size', type=int, default=64, help='number of batches for ppo (default: 32)')

parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
parser.add_argument('--tau', type=float, default=0.95, help='gae parameter (default: 0.95)')
parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=16, help='how many training CPU processes to use (default: 16)')
parser.add_argument('--num-steps', type=int, default=5, help='number of forward steps in A2C (default: 5)')
parser.add_argument('--env_name', default='PongNoFrameskip-v4', help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('-g',  type=str, default='0', help=['specify GPU'])
parser.add_argument('-f', type=str, default="./results/")          # Folder to save results in

locals().update(parser.parse_args().__dict__)    


job_prefix = "python "
exp_script = './riashat_main.py ' 
job_prefix += exp_script

args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.g

experiments = args.experiments
env_name = args.env_name
seed = args.seed
learning_rate = args.lr

entropy_coef = args.entropy_coef


folder = args.f


learning_rate_search = [0.4]
entropy_coef = [0.01, 0.03, 0.05]

"""
Specify other hyperparameters to search over here- 
"""


grid = [] 
grid += [['-env_name', [env_name]]]
grid += [['-seed', [seed]]]
grid += [['-lr', learning_rate_search]]
grid += [['-entropy_coef', entropy_coef]]
grid += [['-actor_lr', [actor_lr]]]
grid += [['-critic_lr', [critic_lr]]]
grid += [['-batch_size', [batch_size]]]


grid += [['-f', [folder]]]
#grid += [['-gpu', [gpu]]]

job_strs = []
for settings in grid_search(grid):
    for e in range(experiments):    
        job_str = job_prefix + settings
        job_strs.append(job_str)
print("njobs", len(job_strs))

for job_str in job_strs:
    os.system(job_str)