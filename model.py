import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import orthogonal


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


# class FFPolicy(nn.Module):
#     def __init__(self):
#         super(FFPolicy, self).__init__()

#     def forward(self, inputs, states, masks):
#         raise NotImplementedError

#     def act(self, inputs, states, masks, deterministic=False):
#         value, x, states = self(inputs, states, masks)
#         action = self.dist.sample(x, deterministic=deterministic)
#         action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
#         return action, action_log_probs, states

#     def evaluate_actions(self, inputs, states, masks, actions):
#         value, x, states = self(inputs, states, masks)
#         action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
#         return value, action_log_probs, dist_entropy, states


class FFPolicy_discrete(nn.Module):
    def __init__(self):
        super(FFPolicy_discrete, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        x, pre_softmax = self(inputs, states, masks)
        probs = F.softmax(pre_softmax)
        if deterministic is False:
            action = probs.multinomial()
        else:
            action = probs.max(1, keepdim=True)[1]
        
        # log_probs = F.log_softmax(pre_softmax)
        # dist_entropy = - (log_probs * probs).sum(-1).mean()

        log_probs = F.log_softmax(x)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
    

        dist_entropy.cuda()        

        return action, probs, states, dist_entropy


    # def evaluate_actions(self, inputs, states, masks, actions):
    #     x, pre_softmax = self(inputs, states, masks)
    #     #action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
    #     log_probs = F.log_softmax(pre_softmax)
    #     probs = F.softmax(pre_softmax)
    #     action_log_probs = log_probs.gather(1, actions)
    #     dist_entropy = -(log_probs * probs).sum(-1).mean()
    #     return value, action_log_probs, dist_entropy, states


class Critic(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc_action = nn.Linear(num_actions, 256)
        self.fc6 = nn.Linear(512 + 256 , 256)
        self.fc7 = nn.Linear(256 , 1)

    def forward(self, x, action):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        action_emb =  F.relu(self.fc_action(action))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = F.relu(self.fc6( torch.cat((x, action_emb), dim=1)))
        return self.fc7(x)


class Baseline_Critic(nn.Module):
    def __init__(self, in_channels=4, num_actions=18):
        """
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(Baseline_Critic, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc_action = nn.Linear(num_actions, 256)
        self.fc6 = nn.Linear(512 + 256 , 256)
        self.fc7 = nn.Linear(256 , 1)

    def forward(self, x, action):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        action_emb =  F.relu(self.fc_action(action))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        x = F.relu(self.fc6( torch.cat((x, action_emb), dim=1)))
        return self.fc7(x)


class Actor(FFPolicy_discrete):
    def __init__(self, num_inputs, action_space, use_gru, action_dim):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        self.linear1 = nn.Linear(32 * 7 * 7, 512)
        self.linear2 = nn.Linear(512, action_dim)
        self.reset_parameters()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.linear2.weight.data.mul_(relu_gain)

        if hasattr(self, 'gru'):
            orthogonal(self.gru.weight_ih.data)
            orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    def forward(self, inputs, states, masks):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        pre_softmax=x
        x = F.softmax(x)

        return x, pre_softmax

