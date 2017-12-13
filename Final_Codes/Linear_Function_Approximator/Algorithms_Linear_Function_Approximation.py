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
from lib import plotting

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
# featurizer = sklearn.pipeline.FeatureUnion([
#         ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
#         ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
#         ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
#         ("rbf4", RBFSampler(gamma=0.5, n_components=100))
#         ])
# featurizer.fit(scaler.transform(observation_examples))



##mountain car env
featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
featurizer.fit(scaler.transform(observation_examples))



def featurize_state( state):
	state = np.array([state])
	state=state.reshape(-1,1)
	scaled = scaler.transform(state)
	featurized = featurizer.transform(scaled)
	return featurized[0]





##gridworld
# def featurize_state( state):
# 	state = np.array([state])
# 	state=state.reshape(-1,1)
# 	scaled = scaler.transform(state)
# 	featurized = featurizer.transform(scaled)
# 	return featurized[0]



	

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
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))
	cum_reward=np.zeros(num_episodes)

	for i_episode in range(num_episodes):
		print ("Episode Number, Q Learning:", i_episode)
		#agent policy based on the greedy maximisation of Q
		#policy = make_epsilon_greedy_policy( epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()
		sum_rewards=0
		

		#for each one step in the environment
		for t in itertools.count():
			action_probs=policy(env,epsilon,state,theta)
			
			action=np.random.choice(np.arange(len(action_probs)),p=action_probs)
			
			features_state=featurize_state(state)
			
			q_values_state=np.dot(theta.T,features_state)
			
			next_state, reward, done, _ = env.step(action)

			cum_reward[i_episode]+=reward
			
			
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
		
		#print('Episode reward is',stats.episode_rewards[i_episode])
	return cum_reward

def constrained_q_learning(env, num_episodes,theta, discount_factor=0.99, epsilon=0.1, epsilon_decay=0.999):

	#q-learning algorithm with linear function approximation here

	#estimator : Estimator of Q^w(s,a)	- function approximator
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print ("Episode Number, Q Learning:", i_episode)
		 
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()

		#next_action = None

		#for each one step in the environment
		done=False
		for t in itertools.count():
    		
			action_probs=policy(env,epsilon,state,theta)

			action=np.random.choice(np.arange(len(action_probs)),p=action_probs)
			
			features_state=featurize_state(state)
			q_values_state=np.dot(theta.T,features_state) #estimator.predict(state)
			next_state, reward, done, _ = env.step(action)

			if done:
				break
			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			features_next_state=featurize_state(next_state)
			q_values_next_state=np.dot(theta.T,features_next_state)

			g_td=(-reward-discount_factor*(np.max(q_values_next_state))+q_values_state[action])*features_state

			
			g_td_next=(-reward-discount_factor*(np.max(q_values_next_state))+q_values_state[action])*features_next_state

			g_v_next_hat=features_next_state/(np.linalg.norm(g_td_next))
			pi_gtd=np.dot(g_td.T,g_v_next_hat)*g_v_next_hat
			g_update=g_td-pi_gtd

			theta[:,action]=theta[:,action]-alpha*g_update



			if done:
				break
			state = next_state
			#action=next_action
		print("Episode reward is ",stats.episode_rewards[i_episode])
	return stats

def sarsa(env, num_episodes, theta,discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
	#estimator : Estimator of Q^w(s,a)	- function approximator
	#stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes)) 
	cum_reward=np.zeros(num_episodes)

	for i_episode in range(num_episodes):
		print ("Episode Number, SARSA:", i_episode)
		#policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()
		next_action = None

		action_probs = policy(env,epsilon,state,theta)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)		
		sum_rewards=0

		for t in itertools.count():
			features_state=featurize_state(state)
			q_state=np.dot(theta.T,features_state)
			q_state_action=q_state[action]

			next_state, reward, done, _ = env.step(action)

			cum_reward[i_episode]+=reward
			

			next_action_probs = policy(env,epsilon,next_state,theta)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			#update Q-values for the next state, next action
			features_next=featurize_state(next_state)

			q_values_next = np.dot(theta.T,features_next)#estimator.predict(next_state)

			q_next_state_next_action = q_values_next[next_action] 

			td_target = reward + discount_factor * q_next_state_next_action
			td_error=td_target-q_state_action

			theta[:,action]+=alpha*td_error*features_state

			#estimator.update(state, action, td_target)

			if done:
				break

			state = next_state
			action = next_action
		
	return cum_reward


def constrained_sarsa(env, num_episodes, theta,discount_factor=1.0, epsilon=0.1, epsilon_decay=0.8):
	#estimator : Estimator of Q^w(s,a)	- function approximator
	#stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))  

	for i_episode in range(num_episodes):
		print ("Episode Number, SARSA:", i_episode)
		#policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)
		last_reward = stats.episode_rewards[i_episode - 1]
		state = env.reset()
		next_action = None

		action_probs = policy(env,epsilon,state,theta)
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)		

		done=False
		done_next=False

		for t in itertools.count():

			features_state=featurize_state(state)
			q_state=np.dot(theta.T,features_state)
			q_state_action=q_state[action]

			next_state, reward, done, _ = env.step(action)

			stats.episode_rewards[i_episode] += reward
			stats.episode_lengths[i_episode] = t

			next_action_probs = policy(env,epsilon,next_state,theta)
			next_action = np.random.choice(np.arange(len(next_action_probs)), p = next_action_probs)

			# next_next_state,next_reward,done,_=env.step(next_action)

			# next_next_action_probs = policy(env,epsilon,next_next_state,theta)
			# next_next_action = np.random.choice(np.arange(len(next_next_action_probs)), p = next_next_action_probs)


			#update Q-values for the next state, next action
			features_next=featurize_state(next_state)

			#features_next_next=featurize_state(next_next_state)

			q_values_next = np.dot(theta.T,features_next)#estimator.predict(next_state)
			#q_values_next_next=np.dot(theta.T,features_next_next)

			q_next_state_next_action = q_values_next[next_action]

			#q_next_next_state_next_next_action=q_values_next_next[next_next_action] 

			td_target = reward + discount_factor * q_next_state_next_action
			td_error=td_target-q_state_action

			#td_target_next=next_reward+discount_factor*q_next_next_state_next_next_action
			#td_error_next=td_target-q_next_state_next_action

			
			g_td=td_error*features_state
			g_td_next=td_error*features_next*discount_factor*-1

			g_hat_v=g_td_next/(np.linalg.norm(g_td_next))

			pi_gtd=(np.dot(g_td.T,g_hat_v))*g_hat_v

			
			g_update=g_td-pi_gtd
			#print(np.linalg.norm(g_update))
			theta[:,action]=theta[:,action]+alpha*g_update



			#theta[:,action]+=alpha*td_error*features_state

			#estimator.update(state, action, td_target)

			if done:
				break

			state = next_state
			action = next_action

	return stats







def main():

	#print ("Q Learning")
	#estimator = Estimator()
	
	theta = np.zeros(shape=(400,env.action_space.n))
	num_episodes = 1000
	smoothing_window = 25
	cum_reward = constrained_sarsa(env, num_episodes,theta)
	plotting.plot_episode_stats(cum_reward)
	env.close()


	
	
if __name__ == '__main__':
	main()




