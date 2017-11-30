# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:59:45 2017

@author: Raihan
"""

import gym
import numpy as np
#mport cliff_walking
import pandas as pd
import matplotlib.pyplot as plt
from lib.envs.cliff_walking import CliffWalkingEnv

env = CliffWalkingEnv()
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .85
gamma = .99
episodes = 2000
#create lists to contain total rewards and steps per episode

rList = []
def Q_learning(episodes=2000):
        for i in range(episodes):
            #Reset environment and get first new observation
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            #The Q-Table learning algorithm
            while j < 100:
                j+=1
                #Choose an action by greedily (with noise) picking from Q table
                a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
                #Get new state and reward from environment
                s1,r,d,_ = env.step(a)
                #Update Q-Table with new knowledge
                Q[s,a] = Q[s,a] + lr*(r + gamma*np.max(Q[s1,:]) - Q[s,a])
                rAll += r
                s = s1
                if d == True:
                    
                    break
            #jList.append(j)
            rList.append(rAll)
        return (rList,Q)
    
def main():
    episodes=3000 #defining the number of episodes
    r,Q=Q_learning(episodes)
    
    print(r)
    r= pd.Series(r).rolling(150,150).mean()#smoothing the plot for r
    
    plt.plot(np.arange(episodes), r)
    plt.ylabel('Average reward per episode')
    plt.xlabel('number of episodes')
    plt.title('Plot of convergence of Q_learning')
    plt.show()
main()    
    
            
        
   
    
    