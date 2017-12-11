import gym
import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
import pdb

import pandas as pd
import sys
import random

from lib import plotting
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from collections import namedtuple
from lib.envs.windy_gridworld import WindyGridworldEnv

import matplotlib.pyplot as plt

"""
Environment
"""

from collections import defaultdict
from lib.envs.windy_gridworld import WindyGridworldEnv
#from lib import plotting

#env = gym.envs.make('MountainCar-v0')
env=WindyGridworldEnv()




"""
Feature Extactor
"""
observation_examples = np.array([env.observation_space.sample() for x in range(1)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)


#convert states to a feature representation:
#used an RBF sampler here for the feature map
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))


# featurizer = sklearn.pipeline.FeatureUnion([
#         ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
#         ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
#         ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
#         ("rbf4", RBFSampler(gamma=0.5, n_components=100))
#         ])
# featurizer.fit(scaler.transform(observation_examples))



def featurize_state( state):
	state = np.array([state])
	state=state.reshape(-1,1)
	scaled = scaler.transform(state)
	featurized = featurizer.transform(scaled)
	return featurized[0]



	

def policy(env,epsilon,observation,theta):
	nA=env.action_space.n
	A = np.ones(nA, dtype=float) * epsilon / nA
	features_state=featurize_state(observation)
	q_values =np.dot(theta.T,features_state)
	best_action = np.argmax(q_values)
	A[best_action] += (1.0 - epsilon)
	return A
	



alpha=0.01

"""
Main Baselines
"""
def q_learning(env, num_episodes,theta,discount_factor=0.99, epsilon=0.1, epsilon_decay=0.999):
    
	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	
	cum_rewards=[]  

	for i_episode in range(num_episodes):
		print ("Episode Number, Q Learning:", i_episode)
		
		state = env.reset()
		sum_rewards=0
		

		#for each one step in the environment
		for t in itertools.count():
			action_probs=policy(env,epsilon,state,theta)
			
			action=np.random.choice(np.arange(len(action_probs)),p=action_probs)
			
			features_state=featurize_state(state)
			
			q_values_state=np.dot(theta.T,features_state)
			
			next_state, reward, done, _ = env.step(action)

			sum_rewards+=reward
			#stats.episode_lengths[i_episode]=t
			
			
			#update Q-values for the next state
			features_next=featurize_state(next_state)
			
			q_values_next = np.dot(theta.T,features_next)

			#Q-value TD Target
			td_target = reward + discount_factor * np.max(q_values_next)
			td_error=td_target-q_values_state[action]
			#pdb.set_trace()

			#update the Q values
			#not this anymore
			#Q[state][action] += alpha * td_delta
			theta[:,action]+=alpha*td_error*features_state
			if done:
				break
			state = next_state
		cum_rewards.append(sum_rewards)
		#print('Episode reward is',stats.episode_rewards[i_episode])
	return cum_rewards



def main():
    theta = np.zeros(shape=(400,env.action_space.n))
    num_episodes = 1000
    smoothing_window = 5
    cum_reward = q_learning(env, num_episodes,theta)
    #avg.append(cum_reward)

    plt.plot(cum_reward)
    plt.show()
    env.close()


	
	
if __name__ == '__main__':
	main()
