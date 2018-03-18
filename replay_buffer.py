import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
    # def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size, action_dim):
    #     self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
    #     self.states = torch.zeros(num_steps + 1, num_processes, state_size)
    #     self.rewards = torch.zeros(num_steps, num_processes, 1)
    #     self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
    #     self.returns = torch.zeros(num_steps + 1, num_processes, 1)
    #     self.action_log_probs = torch.zeros(num_steps, num_processes, action_dim)
    #     if action_space.__class__.__name__ == 'Discrete':
    #         action_shape = 1
    #     else:
    #         action_shape = action_space.shape[0]
    #     self.actions = torch.zeros(num_steps, num_processes, action_shape)
    #     if action_space.__class__.__name__ == 'Discrete':
    #         self.actions = self.actions.long()
    #     self.masks = torch.ones(num_steps + 1, num_processes, 1)

    # def cuda(self):
    #     self.observations = self.observations.cuda()
    #     self.states = self.states.cuda()
    #     self.rewards = self.rewards.cuda()
    #     self.value_preds = self.value_preds.cuda()
    #     self.returns = self.returns.cuda()
    #     self.action_log_probs = self.action_log_probs.cuda()
    #     self.actions = self.actions.cuda()
    #     self.masks = self.masks.cuda()

    def __init__(self):
        self.storage = []

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, data):
        self.storage.append(data)

    # Store tuples of N transitions where N is the trajectory length
    # def insert(self, step, current_obs, state, action, action_log_prob, value_pred, reward, mask):
    #     self.observations[step + 1].copy_(current_obs)
    #     self.states[step + 1].copy_(state)
    #     self.actions[step].copy_(action)
    #     self.action_log_probs[step].copy_(action_log_prob)
    #     self.value_preds[step].copy_(value_pred)
    #     self.rewards[step].copy_(reward)
    #     self.masks[step + 1].copy_(mask)


    # def sample(self, batch_size=100):
    #     import pdb; pdb.set_trace()
    #     ind = np.random.randint(0, len(self.observations), size=batch_size)
    #     x, y, u, r, d = [], [], [], [], []

    #     for i in ind:
    #         X, Y, U, R, D = self.storage[i]
    #         x.append(np.array(X, copy=False))
    #         y.append(np.array(Y, copy=False))
    #         u.append(np.array(U, copy=False))
    #         r.append(np.array(R, copy=False))
    #         d.append(np.array(D, copy=False))
    #     return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)



    def sample(self, batch_size=100):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d, e = [], [], [], [], [], []

        for i in ind:
            X, Y, U, R, D, E = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            e.append(np.array(E, copy=False))
        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), np.array(e)
