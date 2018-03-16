import torch
import torch.nn as nn

import numpy as np
import argparse
import numpy as np
import random
import os
import time
import json


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

# A temporary solution from the master branch.
# https://github.com/pytorch/pytorch/blob/7752fe5d4e50052b3b0bbc9109e599f8157febc0/torch/nn/init.py#L312
# Remove after the next version of PyTorch gets release.
def orthogonal(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor





create_folder = lambda f: [ os.makedirs(f) if not os.path.exists(f) else False ]
class Logger(object):
      def __init__(self, environment_name='', folder='./results' ):
            """
            Saves experimental metrics for use later.
            :param experiment_name: name of the experiment
            :param folder: location to save data
            : param environment_name: name of the environment
            """
            self.final_rewards_mean = []
            self.final_rewards_median =[]
            self.final_rewards_min = []
            self.final_rewards_max = []

            self.all_value_loss = []
            self.all_policy_loss = []

            self.save_folder = os.path.join(folder, environment_name, time.strftime('%y-%m-%d-%H-%M-%s'))
            create_folder(self.save_folder)


      def record_reward(self, reward_return):
            self.returns_eval = reward_return

      def record_data(self, final_rewards_mean, final_rewards_median, final_rewards_min, final_rewards_max, all_value_loss, all_policy_loss):
            self.final_rewards_mean.append(final_rewards_mean)
            self.final_rewards_median.append(final_rewards_median)
            self.final_rewards_min.append(final_rewards_min)
            self.final_rewards_max.append(final_rewards_max)
            self.all_value_loss.append(all_value_loss)
            self.all_policy_loss.append(all_value_loss)



      def save(self):
            np.save(os.path.join(self.save_folder, "final_rewards_mean.npy"), self.final_rewards_mean)
            np.save(os.path.join(self.save_folder, "final_rewards_median.npy"), self.final_rewards_median)
            np.save(os.path.join(self.save_folder, "final_rewards_min.npy"), self.final_rewards_min)
            np.save(os.path.join(self.save_folder, "final_rewards_max.npy"), self.final_rewards_max)
            np.save(os.path.join(self.save_folder, "all_value_loss.npy"), self.all_value_loss)
            np.save(os.path.join(self.save_folder, "all_policy_loss.npy"), self.all_policy_loss)



      def save_args(self, args):
            """
            Save the command line arguments
            """
            with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
                  json.dump(dict(args._get_kwargs()), f)



