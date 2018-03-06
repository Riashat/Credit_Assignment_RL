import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from numpy import genfromtxt
import pdb
from scipy import stats


eps = np.arange(0, 230000, 5e3)

plt.rcParams['text.usetex'] = True


cum_rwd = np.load('./results/DDPG_LunarLander-v2_0.npy')



def comparison_plot(stats1, stats2, stats3,  smoothing_window=5, noshow=False):

    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'32'}


    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_2 = pd.Series(stats2).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed_3 = pd.Series(stats3).rolling(smoothing_window, min_periods=smoothing_window).mean()

    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "#1f77b4", linewidth=2.5, linestyle='solid', label="Critic Network Activation = ReLU")    
    plt.fill_between( eps, rewards_smoothed_1 + std_hs_relu,   rewards_smoothed_1 - std_hs_relu, alpha=0.2, edgecolor='#1f77b4', facecolor='#1f77b4')

    cum_rwd_2, = plt.plot(eps, rewards_smoothed_2, color = "#ff7f0e", linewidth=2.5, linestyle='dashed', label="Critic Network Activation = TanH" )  
    plt.fill_between( eps, rewards_smoothed_2 + std_hs_tanh,   rewards_smoothed_2 - std_hs_tanh, alpha=0.2, edgecolor='#ff7f0e', facecolor='#ff7f0e')

    cum_rwd_3, = plt.plot(eps, rewards_smoothed_3, color = "#d62728", linewidth=2.5, linestyle='dashdot', label="Critic Network Activation = Leaky ReLU" )  
    plt.fill_between( eps, rewards_smoothed_3 + std_hs_leaky_relu,   rewards_smoothed_3 - std_hs_leaky_relu, alpha=0.2, edgecolor='#d62728', facecolor='#d62728')

    plt.legend(handles=[cum_rwd_1, cum_rwd_2, cum_rwd_3],  loc='lower right', prop={'size' : 26})
    plt.xlabel("Timesteps",**axis_font)
    plt.ylabel("Average Returns", **axis_font)
    plt.title("DDPG with HalfCheetah Environment - Critic Network Activations", **axis_font)
  
    plt.show()

    fig.savefig('ddpg_halfcheetah_value_activations.png')
    
    return fig



def plot_results(stats1, smoothing_window=5, noshow=False):

    fig = plt.figure(figsize=(16, 8))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'32'}


    rewards_smoothed_1 = pd.Series(stats1).rolling(smoothing_window, min_periods=smoothing_window).mean()


    cum_rwd_1, = plt.plot(eps, rewards_smoothed_1, color = "#1f77b4", linewidth=2.5, linestyle='solid', label="Discrete Action DDPG")    
    #plt.fill_between( eps, rewards_smoothed_1 + std_hs_relu,   rewards_smoothed_1 - std_hs_relu, alpha=0.2, edgecolor='#1f77b4', facecolor='#1f77b4')

    plt.legend(handles=[cum_rwd_1],  loc='lower right', prop={'size' : 26})
    plt.xlabel("Timesteps",**axis_font)
    plt.ylabel("Average Returns", **axis_font)
    plt.title("DDPG with Discrete Actions - Lunar Lander Environment", **axis_font)
  
    plt.show()

    fig.savefig('ddpg_halfcheetah_value_activations.png')
    
    return fig

def main():
   plot_results(cum_rwd)


if __name__ == '__main__':
    main()