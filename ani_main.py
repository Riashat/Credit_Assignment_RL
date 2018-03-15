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
from model import CNNPolicy, MLPPolicy, Critic, Actor
from storage import RolloutStorage
from visualize import visdom_plot
from replay_buffer import ReplayBuffer

args = get_args()
criterion = nn.MSELoss()


assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes
USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

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

    if args.vis:
        from visdom import Visdom
        viz = Visdom()
        win = None

    envs = [make_env(args.env_name, args.seed, i, args.log_dir)
                for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    if envs.action_space.__class__.__name__ == "Discrete":
        action_shape = 1
    else:
        action_shape = envs.action_space.shape[0]


    if len(envs.observation_space.shape) == 3:
        actor_critic = Actor(obs_shape[0], envs.action_space, args.recurrent_policy,  envs.action_space.n)
        target_actor = Actor(obs_shape[0], envs.action_space, args.recurrent_policy, envs.action_space.n)
        critic = Critic(in_channels=4, num_actions=envs.action_space.n)
        critic_target = Critic(in_channels=4, num_actions=envs.action_space.n)
    else:
        assert not args.recurrent_policy, \
            "Recurrent policy is not implemented for the MLP controller"
        actor_critic = MLPPolicy(obs_shape[0], envs.action_space)


    if args.cuda:
        actor_critic.cuda()
        critic.cuda()
        critic_target.cuda()
        target_actor.cuda()

    if args.algo == 'a2c':
        optimizer = optim.RMSprop(actor_critic.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
        critic_optim = optim.Adam(critic.parameters(), lr=1e-4)
        gamma = 0.99
        tau = 0.001


    #memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
    mem_buffer = ReplayBuffer()

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size, envs.action_space.n)
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
        for step in range(args.num_steps):
            # Sample actions
            action, action_log_prob, states = actor_critic.act(Variable(rollouts.observations[step], volatile=True),
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
            mem_buffer.add((pre_state, current_obs, action_log_prob.data.cpu().numpy(), reward, done))
            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value.data, reward, masks)

        action, action_log_prob, states  = actor_critic.act(Variable(rollouts.observations[-1], volatile=True),
                                            Variable(rollouts.states[-1], volatile=True),
                                            Variable(rollouts.masks[-1], volatile=True))#[0].data

        next_value = critic.forward(Variable(rollouts.observations[-1], volatile=True), action_log_prob).data


        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        if True:
            state, next_state, action, reward, done = mem_buffer.sample(5)
            next_state = next_state.reshape([-1, *obs_shape])
            state = state.reshape([-1, *obs_shape])
            action = action.reshape([-1, 6])
            next_q_values = critic_target(to_tensor(next_state, volatile=True),target_actor(to_tensor(next_state, volatile=True), to_tensor(next_state, volatile=True), to_tensor(next_state, volatile=True))[0])
            next_q_values.volatile=False
            target_q_batch = to_tensor(reward) + args.gamma*to_tensor(done.astype(np.float))*next_q_values
            critic.zero_grad()
            q_batch = critic(to_tensor(state), to_tensor(action))
            value_loss = criterion(q_batch, target_q_batch)
            value_loss.backward()
            critic_optim.step()
            actor_critic.zero_grad()
            policy_loss = -critic(to_tensor(state),actor_critic(to_tensor(state), to_tensor(state), to_tensor(state))[0])
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)
            optimizer.step()
            soft_update(target_actor, actor_critic, tau)
            soft_update(critic_target, critic, tau)

        '''
        if args.algo in ['a2c', 'acktr']:
            action_log_probs, probs, dist_entropy, states = actor_critic.evaluate_actions(Variable(rollouts.observations[:-1].view(-1, *obs_shape)),
                                                                                           Variable(rollouts.states[0].view(-1, actor_critic.state_size)),
                                                                                           Variable(rollouts.masks[:-1].view(-1, 1)),
                                                                                           Variable(rollouts.actions.view(-1, action_shape)))
            values = critic.forward(Variable(rollouts.observations[:-1].view(-1, *obs_shape)), probs).data

            values = values.view(args.num_steps, args.num_processes, 1)
            action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

            #advantages = Variable(rollouts.returns[:-1]) - values
            advantages = rollouts.returns[:-1] - values
            value_loss = advantages.pow(2).mean()

            action_loss = -(Variable(advantages) * action_log_probs).mean()
            #action_loss = -(Variable(advantages.data) * action_log_probs).mean()


            optimizer.zero_grad()
            critic_optim.zero_grad()
            (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

            if args.algo == 'a2c':
                nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

            optimizer.step()
            critic_optim.step()
        '''
        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(),
                       value_loss.data.cpu().numpy()[0], policy_loss.data.cpu().numpy()[0]))
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo)
            except IOError:
                pass

if __name__ == "__main__":
    main()
