import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from glob import glob

sns.set_style('white')
smoothing_window=10
COLORS = lambda n: list(reversed(sns.color_palette("hls", n)))
ax=plt.gca()
mean_rewards=np.load("normal_q_grid_mean_reward.npy")
variance_rewards=np.load("normal_q_grid_variance_reward.npy")
print(mean_rewards.shape)
mean_rewards=mean_rewards[0:1000]
variance_rewards=variance_rewards[0:1000]
x=np.arange(1000)
variance_rewards=pd.Series(variance_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
std=np.sqrt(variance_rewards)



mean_rewards=pd.Series(mean_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

plt.plot(x,mean_rewards,'b-')
plt.fill_between(x,mean_rewards-std,mean_rewards+std,alpha=0.5)
plt.ylabel("Average Reward/Episode")
plt.xlabel("Episodes")
#plt.ylim(-200,10)
plt.title("Grid World Environment")


plt.show()