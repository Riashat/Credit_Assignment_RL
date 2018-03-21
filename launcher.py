
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
parser.add_argument('--policy', type=str, default="Trust_DDPG")          # Policy name
parser.add_argument('--env', type=str, default="HalfCheetah-v1")         # OpenAI gym environment name
parser.add_argument('--seed', default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument('--start_timesteps', default=10000, type=int)     # How many time steps purely random policy is run for
parser.add_argument('--eval_freq', default=5e3, type=float)         # How often (time steps) we evaluate
parser.add_argument('--max_timesteps', default=1e6, type=float)     # Max time steps to run environment for
parser.add_argument('--save_models', default=True,type=bool)           # Whether or not models are saved
parser.add_argument('--expl_noise', default=0.1, type=float)        # Std of Gaussian exploration noise
parser.add_argument('--batch_size', default=100, type=int)          # Batch size for both actor and critic
parser.add_argument('--discount', default=0.99, type=float)         # Discount factor
parser.add_argument('--tau', default=0.005, type=float)             # Target network update rate
parser.add_argument('--policy_noise', default=0.2, type=float)      # Noise added to target policy during critic update
parser.add_argument('--noise_clip', default=0.5, type=float)        # Range to clip target policy noise
parser.add_argument('--policy_freq', default=2, type=int)           # Frequency of delayed policy updates
parser.add_argument('--lambda_critic', default = 0.1, type=float)   # Lambda trade-off for critic regularizer
parser.add_argument('--lambda_actor', default = 0.1, type=float)    # Lambda trade-off for actor regularizer
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
policy_name = args.policy
env_name = args.env
seed = args.seed
#import pdb;pdb.set_trace()
start_timesteps = args.start_timesteps
eval_freq = args.eval_freq
max_timesteps = args.max_timesteps
expl_noise = args.expl_noise
batch_size = args.batch_size
discount = args.discount
tau = args.tau
policy_noise = args.policy_noise
noise_clip = args.noise_clip
policy_freq = args.policy_freq
lambda_critic = args.lambda_critic
lambda_actor = args.lambda_actor
folder = args.f
#save_models=args.save_models


grid = [] 
grid += [['-policy_name', [[policy_name]]]]
grid += [['-env_name', [env_name]]]
grid += [['-seed', [seed]]]
grid += [['-start_timesteps', [start_timesteps]]]
grid += [['-eval_freq', [eval_freq]]]
grid += [['-max_timesteps', [max_timesteps]]]
grid += [['-save_models',[args.save_models]]]
grid += [['-expl_noise', [expl_noise]]]
grid += [['-batch_size', [batch_size]]]
grid += [['-batch_size', [batch_size]]]
grid += [['-tau', [tau]]]
grid += [['-policy_noise', [policy_noise]]]
grid += [['-noise_clip', [noise_clip]]]
grid += [['-policy_freq', [policy_freq]]]
grid += [['-lambda_critic', [lambda_critic]]]
grid += [['-lambda_actor', [lambda_actor]]]
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