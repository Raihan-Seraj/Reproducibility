import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from glob import glob

sns.set_style('white')

COLORS = lambda n: list(reversed(sns.color_palette("hls", n)))
ax=plt.gca()
mean_rewards=np.load("mean_rewards.npy")
variance_rewards=np.load("variance_rewards.npy")
mean_rewards=mean_rewards[0:990]
variance_rewards=variance_rewards[0:990]
x=np.arange(990)
std=np.sqrt(variance_rewards)
plt.plot(x,mean_rewards,'b-')
plt.fill_between(x,mean_rewards-std,mean_rewards+std,alpha=0.5)
plt.ylabel("Average Reward/Episode")
plt.xlabel("Episodes")
plt.title("Cartpole Environment")


# ax.plot(x, mean_rewards, color='black')
# ax.fill_between(x, mean_rewards+std, mean_rewards-std, facecolor='blue', alpha=0.1)

# ax.set_xticks(x)
# ax.set_ylabel('Average Cumulative Reward')
# ax.set_xlabel('Number of Episodes')
# sns.despine()
# ax.set_xlim(xmin=0)
plt.show()