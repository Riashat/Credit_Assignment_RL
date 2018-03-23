import copy
import glob
import os
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from kfac import KFACOptimizer
from model import Critic, Actor, Baseline_Critic
#from model import CNNPolicy, MLPPolicy, Critic, Actor
from storage import RolloutStorage
from visualize import visdom_plot
from replay_buffer import ReplayBuffer

from utils import Logger

args = get_args()
assert args.algo in ["a2c"]
criterion = nn.MSELoss()

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

# soft update after the actor and critic updates
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau 
        )

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)


def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    os.environ['OMP_NUM_THREADS'] = '1'

    logger = Logger(environment_name = args.env_name, entropy_coff= 'entropy_coeff_' + str(args.entropy_coef), folder = args.folder)
    logger.save_args(args)

    print ("---------------------------------------")
    print ('Saving to', logger.save_folder)
    print ("---------------------------------------")    

    if args.vis:
        from visdom import Visdom
        viz = Visdom()
        win = None

    envs = [make_env(args.env_name, args.seed, i, args.log_dir)
                for i in range(args.num_processes)]

    ### for the number of processes to use
    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    ## ALE Environments : mostly has Discrete action_space type
    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]

    ### shape==3 for ALE Environments : States are 3D (Image Pi)
    if len(envs.observation_space.shape) == 3:
        actor = Actor(obs_shape[0], envs.action_space, args.recurrent_policy,  envs.action_space.n)
        target_actor = Actor(obs_shape[0], envs.action_space, args.recurrent_policy, envs.action_space.n)
        critic = Critic(in_channels=4, num_actions=envs.action_space.n)
        critic_target = Critic(in_channels=4, num_actions=envs.action_space.n)
        baseline_target = Baseline_Critic(in_channels=4, num_actions=envs.action_space.n)


    if args.cuda:
        actor.cuda()
        critic.cuda()
        critic_target.cuda()
        target_actor.cuda()
        baseline_target.cuda()

    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)    
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr)
    baseline_optim = optim.Adam(actor.parameters(), lr=1e-4)
    tau_soft_update = 0.001

    mem_buffer = ReplayBuffer()
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor.state_size, envs.action_space.n)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):

        ## num_steps = 5 as in A2C
        for step in range(args.num_steps):

            # Sample actions
            action, action_log_prob, states, dist_entropy = actor.act(Variable(rollouts.observations[step], volatile=True),
                                                                      Variable(rollouts.states[step], volatile=True),
                                                                      Variable(rollouts.masks[step], volatile=True))

            value = critic.forward(Variable(rollouts.observations[step], volatile=True), action_log_prob)

            cpu_actions = action.data.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            pre_state= rollouts.observations[step].cpu().numpy()
            update_current_obs(obs)
           
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, dist_entropy.data,  value.data, reward, masks)


        nth_step_return = rollouts.returns[0].cpu().numpy()
        current_state = rollouts.observations[0].cpu().numpy()
        nth_state = rollouts.observations[-1].cpu().numpy()
        current_action = rollouts.action_log_probs[0].cpu().numpy()
        current_action_dist_entropy = rollouts.dist_entropy[0].cpu().numpy()
        
        mem_buffer.add( (current_state, nth_state, current_action, nth_step_return, done, current_action_dist_entropy) )
        action, action_log_prob, states, dist_entropy = actor.act(Variable(rollouts.observations[-1], volatile=True),
                                            Variable(rollouts.states[-1], volatile=True),
                                            Variable(rollouts.masks[-1], volatile=True))#[0].data

        next_value = critic.forward(Variable(rollouts.observations[-1], volatile=True), action_log_prob).data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)


        bs_size = args.batch_size
        if len(mem_buffer.storage) >= bs_size :
        #if True:
            ##samples from the replay buffer
            state, next_state, action, returns, done, entropy_log_prob = mem_buffer.sample(bs_size)

            next_state = next_state.reshape([-1, *obs_shape])
            state = state.reshape([-1, *obs_shape])
            action = action.reshape([-1, envs.action_space.n])

            #current Q estimate
            q_batch = critic(to_tensor(state), to_tensor(action))

            # target Q estimate
            next_q_values = critic_target(to_tensor(next_state, volatile=True),target_actor(to_tensor(next_state, volatile=True), to_tensor(next_state, volatile=True), to_tensor(next_state, volatile=True))[0])
            next_q_values.volatile=False
            target_q_batch = to_tensor(returns) + args.gamma * to_tensor(done.astype(np.float))*next_q_values

            #Critic loss estimate and update
            critic.zero_grad()
            value_loss = criterion(q_batch, target_q_batch)
            value_loss.backward()
            critic_optim.step()

            #evaluating actor_loss : - Q(s, \mu(s))
            actor.zero_grad()
            policy_loss = -critic(to_tensor(state),actor(to_tensor(state), to_tensor(state), to_tensor(state))[0])

            ## Actor update with entropy penalty
            policy_loss = policy_loss.mean() - args.entropy_coef * Variable(torch.from_numpy(np.expand_dims(entropy_log_prob.mean(), axis=0))).cuda()

            #gradient wrt to actor loss 
            if j == 0:
                grad_params = torch.autograd.grad(policy_loss, actor.parameters(), retain_graph=True)
            else:
                grad_params[:] = torch.autograd.grad(policy_loss, actor.parameters(), retain_graph=True)
                
#             grad_params = torch.autograd.grad(policy_loss, actor.parameters(), retain_graph=True)

            policy_loss.backward()
            ### TODO : Do we need gradient clipping?
            #nn.utils.clip_grad_norm(actor.parameters(), args.max_grad_norm)
            actor_optim.step()

            """
            Training the Baseline critic (f(s, \mu(s)))
            """
            # baseline_target.zero_grad()
            # #trade-off between two constraints when training baseline
            # lambda_baseline = 1

            # ## f(s, \mu(s))
            # current_baseline = baseline_target(to_tensor(state),actor(to_tensor(state), to_tensor(state), to_tensor(state))[0])
            # #current_baseline.volatile=False
             
            # ## \grad f(s,a)
            # grad_baseline_params = torch.autograd.grad(current_baseline.mean(), actor.parameters(), retain_graph=True, create_graph=True)

            # ## MSE : (Q - f)^{2}
            # baseline_loss = (q_batch.detach() - current_baseline).pow(2).mean()
            # # baseline_loss.volatile=True

            # actor.zero_grad()
            # baseline_target.zero_grad()
            # grad_norm = 0
            # for grad_1, grad_2 in zip(grad_params, grad_baseline_params):
            #     grad_norm += grad_1.data.pow(2).sum() - grad_2.pow(2).sum()
            # grad_norm = grad_norm.sqrt()
            
            # ##Loss for the Baseline approximator (f)  
            # overall_loss = baseline_loss + lambda_baseline * grad_norm

            # overall_loss.backward()

            # baseline_optim.step()

            soft_update(target_actor, actor, tau_soft_update)
            soft_update(critic_target, critic, tau_soft_update)


        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "" and len(mem_buffer.storage) >= bs_size:
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor
            if args.cuda:
                save_model = copy.deepcopy(actor).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(mem_buffer.storage) >= bs_size:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, value loss {:.5f}, policy loss {:.5f}, Entropy {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(),
                       value_loss.data.cpu().numpy()[0], policy_loss.data.cpu().numpy()[0], entropy_log_prob.mean()))

            final_rewards_mean = [final_rewards.mean()]
            final_rewards_median = [final_rewards.median()]
            final_rewards_min = [final_rewards.min()]
            final_rewards_max = [final_rewards.max()]

            all_value_loss = [value_loss.data.cpu().numpy()[0]]
            all_policy_loss = [policy_loss.data.cpu().numpy()[0]]

            logger.record_data(final_rewards_mean, final_rewards_median, final_rewards_min, final_rewards_max, all_value_loss, all_policy_loss)
            logger.save()


        
        if args.vis and j % args.vis_interval == 0:

            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo)
            except IOError:
                pass

if __name__ == "__main__":
    main()
