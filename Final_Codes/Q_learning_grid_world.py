'''Created on Fri Mar  3 11:59:45 2017
@author: Raihan
'''

import gym
import numpy as np

import pandas as pd
from lib.envs.cliff_walking import CliffWalkingEnv
import matplotlib.pyplot as plt
import itertools
import gridworld10x10 as gd

#env = CliffWalkingEnv()
#Initialize table with all zeros
env=gd.GridworldEnv()
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .85
gamma = .99
episodes = 2000
#create lists to contain total rewards and steps per episode

rList = np.zeros(1000)
def Q_learning(episodes=1000):
        for i in range(episodes):
            #Reset environment and get first new observation
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            #The Q-Table learning algorithm
            for j in itertools.count():
                j+=1
                #Choose an action by greedily (with noise) picking from Q table
                a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
                #Get new state and reward from environment
                s1,r,d,_ = env.step(a)
                #Update Q-Table with new knowledge
                Q[s,a] = Q[s,a] + lr*(r + gamma*np.max(Q[s1,:]) - Q[s,a])
                rList[i] += r
                s = s1
                if d == True:
                    
                    break
            #jList.append(j)
            #rList.append(rAll)
        return (rList,Q)
    
def main():
   
    num_experiments=10
    avg=[]
    for n in range(num_experiments):
        print("Experiment",n)
        episodes=1000 #defining the number of episodes
        r,Q=Q_learning(episodes)


        avg.append(r)

    mean_rewards=np.mean(np.array(avg),axis=0)
    variance_rewards=np.var(np.array(avg),axis=0)    
    np.save("normal_q_grid_mean_reward",mean_rewards)
    np.save("normal_q_grid_variance_reward",variance_rewards)

    # plt.plot(np.arange(episodes), r)
    # plt.ylabel('Average reward per episode')
    # plt.xlabel('number of episodes')
    # plt.title('Plot of convergence of Q_learning')
    # plt.show()
main()    
    
            