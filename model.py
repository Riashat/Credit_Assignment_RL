import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import orthogonal
import numpy as np
from torch.autograd import Variable
from spectral_normalization import SpectralNorm

USE_CUDA = torch.cuda.is_available()
FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class LayerNorm(torch.nn.Module):
     def __init__(self, input_dim):
         super(LayerNorm, self).__init__()
         self.gamma = torch.nn.Parameter(torch.ones(input_dim))
         self.beta = torch.nn.Parameter(torch.zeros(input_dim))
         self.eps = 1e-6


     def forward(self, x): 
         mean = x.mean(-1, keepdim=True)
         std = torch.sqrt(x.var(dim=1, keepdim=True) + self.eps)
         output = self.gamma * (x - mean) / (std + self.eps) + self.beta
         return output #* mask.unsqueeze(1)



class FFPolicy_discrete(nn.Module):
    def __init__(self):
        super(FFPolicy_discrete, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, temperature, action_space, num_processes, deterministic=True):
        x, pre_softmax = self(inputs, states, masks)
        probs = F.softmax(pre_softmax)
        probs = probs ** (1 / temperature)
        ## add Gaussian noise to probs
        OU_Noise = np.random.normal(0, 0.5, size=(num_processes,action_space))
        OU_Noise = Variable(torch.from_numpy(OU_Noise), volatile=False, requires_grad=False).type(FLOAT)
        probs = probs + OU_Noise

        probs = F.softmax(probs)

        if deterministic is False:
            # action = np.random.multinomial(1, probs).argmax()
            action = probs.multinomial()

        else:
            action = probs.max(1, keepdim=True)[1]
        
        log_probs = F.log_softmax(x)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        # dist_entropy.cuda()        

        return action, pre_softmax, states, dist_entropy

    def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
        return Variable(
            torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
        ).type(dtype)



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

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

        ### With Batch Norm added to the Critic
        # self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(32)

        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc_action = nn.Linear(num_actions, 256)
        self.fc6 = nn.Linear(512 + 256 , 256)
        
        self.fc7 = nn.Linear(256 , 1)

        ### With Spectral Normalization addded to the critic
        # self.conv1 = SpectralNorm(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        # self.conv2 = SpectralNorm(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        # self.conv3 = SpectralNorm(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        # self.fc4 = SpectralNorm(nn.Linear(7 * 7 * 64, 512))
        # self.fc_action = SpectralNorm(nn.Linear(num_actions, 256))
        # self.fc6 = SpectralNorm(nn.Linear(512 + 256 , 256))
        # self.fc7 = SpectralNorm(nn.Linear(256 , 1))


    def forward(self, x, action):
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))

        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv3(x)))

        action_emb =  F.relu(self.fc_action(action))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        #x = F.relu(self.fc6( torch.cat((x, action_emb), dim=1)))
        x = self.fc6(torch.cat((x, action_emb), dim=1))
        x = F.relu(x)

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

        """
        When using layer norm and Batch Norm
        """
        # self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 32, 3, stride=1)
        # self.bn3 = nn.BatchNorm2d(32)
        # self.linear1 = nn.Linear(32 * 7 * 7, 512)
        # #self.layernorm_i = LayerNorm(input_dim=512)
        # self.linear2 = nn.Linear(512, action_dim)
        # self.reset_parameters()


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
        #   x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        #x = self.bn3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        pre_softmax=x
        x = F.softmax(x, dim=1)

        return x, pre_softmax

