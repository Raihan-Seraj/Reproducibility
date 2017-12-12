main#######################################################################
# Copyright (C)                                                       #
# 2016 - 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#############################
#used,modified by ALAN,RAIHAN

#after modification, this correspond to the original baird's counter
#-example

###########################

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
# all states: state 0-5 are upper states
STATES = np.arange(0, 6)
# state 6 is lower state
LOWER_STATE = 5
# discount factor
DISCOUNT = 0.99
# each state is represented by a vector of length 6
FEATURE_SIZE = 7
FEATURES = np.zeros((len(STATES), FEATURE_SIZE))
# FEATURES[i,6] is w0 for state i
#FEATURES[i,0] is w1 for state i
for i in range(LOWER_STATE):
    FEATURES[i, i] = 2
    FEATURES[i, 6] = 1
FEATURES[LOWER_STATE, 5] = 1
FEATURES[LOWER_STATE, 6] = 2

# all possible actions
RED = 0
BLACK = 1
ACTIONS = [RED, BLACK]

# reward is always zero
REWARD = 0

# take @action at @state, return the new state
def takeAction(state, action):
    if action == BLACK:
        return LOWER_STATE
    return np.random.choice(STATES[: LOWER_STATE])

# target policy
def targetPolicy(state):
    return BLACK

# state distribution for the behavior policy
stateDistribution = np.ones(len(STATES)) / 6
stateDistributionMat = np.matrix(np.diag(stateDistribution))
# projection matrix for minimize MSVE
projectionMatrix = np.matrix(FEATURES) * \
                   np.linalg.pinv(np.matrix(FEATURES.T) * stateDistributionMat * np.matrix(FEATURES)) * \
                   np.matrix(FEATURES.T) * \
                   stateDistributionMat

# behavior policy
BEHAVIOR_BLACK_PROBABILITY = 1.0 / 6
def behaviorPolicy(state):
    if np.random.binomial(1, BEHAVIOR_BLACK_PROBABILITY) == 1:
        return BLACK
    return RED

""" this is my implementation, I think this is the Q learning
    note this only differ from the semiGradientOffPolicyTD()
    on the rho part
    this is quite converging down,which is good
    however, if this corresponds to the report,
    this should be going up
    like the semiGradientOffPolicyTD( so does that mean
     the author actually implemented  semiGradientOffPolicyTD() ?
     )
  """
def TDlearning(state,theta,alpha):
    action = behaviorPolicy(state)
    nextState = takeAction(state,action)

    if action == RED:
        rho = 0.0
    else:
        rho = 1.0 / BEHAVIOR_BLACK_PROBABILITY
    # rho=1
    delta = REWARD + DISCOUNT * np.dot(FEATURES[nextState, :], theta) - \
            np.dot(FEATURES[state, :], theta)
    delta *= rho * alpha
    # derivatives happen to be the same matrix due to the linearity
    theta += FEATURES[state, :] * delta

    return nextState
""" the ploting """
def TDlearning_plot():
    # Initialize the theta

    temp=np.zeros(2000)
    for _ in range(10):
        theta = np.ones(FEATURE_SIZE)
        theta[6] = 10

        alpha = 0.01

        steps = 2000
        thetas = np.zeros((FEATURE_SIZE, steps))
        state = np.random.choice(STATES)
        states=np.zeros(2000)
        values=np.zeros(2000)
        for step in range(steps):
            state = TDlearning(state, theta, alpha)
            thetas[:, step] = theta
            states[step]=state
            values[step]=np.dot(FEATURES[state,:],theta)
        temp+=values
    avg_val=temp/10.0
    GLOBAL_TD_VAL=avg_val
    global GLOBAL_TD_VAL
    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    # for i in range(FEATURE_SIZE):
    #     plt.plot(thetas[i, :], label='theta' + str(i + 1))
    plt.plot(avg_val,label="values of state")
    plt.xlabel('Steps')
    plt.ylabel('average values of states ')
    plt.title('TD learning')
    plt.legend()
def constraintTD(state,theta,alpha):
    action = behaviorPolicy(state)
    nextState = takeAction(state,action)
    action_next=behaviorPolicy(nextState)
    nextState_next=takeAction(nextState,action_next)
    delta=REWARD+DISCOUNT*np.dot(FEATURES[nextState, :], theta) -\
    np.dot(FEATURES[state,:],theta)
    theta +=alpha*delta*FEATURES[state,:]
    if action == RED:
        rho = 0.0
    else:
        rho = 1.0 / BEHAVIOR_BLACK_PROBABILITY
    if action_next==RED:
        rho_next=0.0
    else:
        rho_next=1.0 / BEHAVIOR_BLACK_PROBABILITY

    # rho=1
    # rho_next=1
    # #gradient_TD

    delta = REWARD + DISCOUNT * np.dot(FEATURES[nextState, :], theta) - \
            np.dot(FEATURES[state, :], theta)
    delta *= FEATURES[state, :] *rho

    delta_next=REWARD+DISCOUNT *np.dot(FEATURES[nextState_next,:],theta) - \
        np.dot(FEATURES[nextState, :],theta)
    if(rho_next!=0):
        delta_next*=FEATURES[nextState, :]
        delta_next/=LA.norm(delta_next)
    else:
        delta_next=0.0
    delta-=np.dot(delta,delta_next)*delta_next
    # derivatives happen to be the same matrix due to the linearity
    theta +=  delta* alpha

    return nextState
""" the ploting """
def c_TDlearning_plot():
    # Initialize the theta

    temp=np.zeros(2000)
    for _ in range(10):
        theta = np.ones(FEATURE_SIZE)
        theta[6] = 10

        alpha = 0.01

        steps = 2000
        thetas = np.zeros((FEATURE_SIZE, steps))
        state = np.random.choice(STATES)
        states=np.zeros(2000)
        values=np.zeros(2000)
        for step in range(steps):
            state = constraintTD(state, theta, alpha)
            thetas[:, step] = theta
            states[step]=state
            values[step]=np.dot(FEATURES[state,:],theta)
        temp+=values
    avg_val=temp/10.0
    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    # for i in range(FEATURE_SIZE):
    #     plt.plot(thetas[i, :], label='theta' + str(i + 1))
    plt.plot(avg_val,label="constrained TD")
    print(GLOBAL_TD_VAL)
    plt.plot(GLOBAL_TD_VAL,label="TD")
    plt.xlabel('Steps')
    plt.ylim(0,50)
    plt.ylabel('average values of states ')
    plt.title('constrained TD learning ')
    plt.legend()

# Semi-gradient off-policy temporal difference
# @state: current state
# @theta: weight for each component of the feature vector
# @alpha: step size
# # @return: next state
# def semiGradientOffPolicyTD(state, theta, alpha):
#     action = behaviorPolicy(state)
#     nextState = takeAction(state, action)
#     # get the importance ratio
#     if action == RED:
#         rho = 0.0
#     else:
#         rho = 1.0 / BEHAVIOR_BLACK_PROBABILITY
#     delta = REWARD + DISCOUNT * np.dot(FEATURES[nextState, :], theta) - \
#             np.dot(FEATURES[state, :], theta)
#     delta *= rho * alpha
#     # derivatives happen to be the same matrix due to the linearity
#     theta += FEATURES[state, :] * delta
#     return nextState
#
#
#
# # Figure 11.2(left), semi-gradient off-policy TD
# def figure11_2_a():
#     # Initialize the theta
#     theta = np.ones(FEATURE_SIZE)
#     theta[6] = 10
#
#     alpha = 0.01
#
#     steps = 1000
#     thetas = np.zeros((FEATURE_SIZE, steps))
#     state = np.random.choice(STATES)
#     for step in range(steps):
#         state = semiGradientOffPolicyTD(state, theta, alpha)
#         thetas[:, step] = theta
#
#     global figureIndex
#     plt.figure(figureIndex)
#     figureIndex += 1
#     for i in range(FEATURE_SIZE):
#         plt.plot(thetas[i, :], label='theta' + str(i + 1))
#     plt.xlabel('Steps')
#     plt.ylabel('Theta value')
#     plt.title('semi-gradient off-policy TD')
#     plt.legend()
#
# # Semi-gradient DP
# # @theta: weight for each component of the feature vector
# # @alpha: step size
# def semiGradientDP(theta, alpha):
#     delta = 0.0
#     # go through all the states
#     for currentState in STATES:
#         expectedReturn = 0.0
#         # compute bellman error for each state
#         for nextState in STATES:
#             if nextState == LOWER_STATE:
#                 expectedReturn += REWARD + DISCOUNT * np.dot(theta, FEATURES[nextState, :])
#         bellmanError = expectedReturn - np.dot(theta, FEATURES[currentState, :])
#         # accumulate gradients
#         delta += bellmanError * FEATURES[currentState, :]
#     # derivatives happen to be the same matrix due to the linearity
#     theta += alpha / len(STATES) * delta
#
# # temporal difference with gradient correction
# # @state: current state
# # @theta: weight of each component of the feature vector
# # @weight: auxiliary trace for gradient correction
# # @alpha: step size of @theta
# # @beta: step size of @weight
# def TDC(state, theta, weight, alpha, beta):
#     action = behaviorPolicy(state)
#     nextState = takeAction(state, action)
#     # get the importance ratio
#     if action == RED:
#         rho = 0.0
#     else:
#         rho = 1.0 / BEHAVIOR_BLACK_PROBABILITY
#     delta = REWARD + DISCOUNT * np.dot(FEATURES[nextState, :], theta) - \
#             np.dot(FEATURES[state, :], theta)
#     theta += alpha * rho * (delta * FEATURES[state, :] - DISCOUNT * FEATURES[nextState, :] * np.dot(FEATURES[state, :], weight))
#     weight += beta * rho * (delta - np.dot(FEATURES[state, :], weight)) * FEATURES[state, :]
#     return nextState
#
# # expected temporal difference with gradient correction
# # @theta: weight of each component of the feature vector
# # @weight: auxiliary trace for gradient correction
# # @alpha: step size of @theta
# # @beta: step size of @weight
# def expectedTDC(theta, weight, alpha, beta):
#     for currentState in STATES:
#         # When computing expected update target, if next state is not lower state, importance ratio will be 0,
#         # so we can safely ignore this case and assume next state is always lower state
#         delta = REWARD + DISCOUNT * np.dot(FEATURES[LOWER_STATE, :], theta) - np.dot(FEATURES[currentState, :], theta)
#         rho = 1 / BEHAVIOR_BLACK_PROBABILITY
#         # Under behavior policy, state distribution is uniform, so the probability for each state is 1.0 / len(STATES)
#         expectedUpdateTheta = 1.0 / len(STATES) * BEHAVIOR_BLACK_PROBABILITY * rho * (
#             delta * FEATURES[currentState, :] - DISCOUNT * FEATURES[LOWER_STATE, :] * np.dot(weight, FEATURES[currentState, :]))
#         theta += alpha * expectedUpdateTheta
#         expectedUpdateWeight = 1.0 / len(STATES) * BEHAVIOR_BLACK_PROBABILITY * rho * (
#             delta - np.dot(weight, FEATURES[currentState, :])) * FEATURES[currentState, :]
#         weight += beta * expectedUpdateWeight
#
#     # if *accumulate* expected update and actually apply update here, then it's synchronous
#     # theta += alpha * expectedUpdateTheta
#     # weight += beta * expectedUpdateWeight
#
# # interest is 1 for every state
# INTEREST = 1
#
# # expected update of ETD
# # @theta: weight of each component of the feature vector
# # @emphasis: current emphasis
# # @alpha: step size of @theta
# # @return: expected next emphasis
# def expectedEmphaticTD(theta, emphasis, alpha):
#     # we perform synchronous update for both theta and emphasis
#     expectedUpdate = 0
#     expectedNextEmphasis = 0.0
#     # go through all the states
#     for state in STATES:
#         # compute rho(t-1)
#         if state == LOWER_STATE:
#             rho = 1.0 / BEHAVIOR_BLACK_PROBABILITY
#         else:
#             rho = 0
#         # update emphasis
#         nextEmphasis = DISCOUNT * rho * emphasis + INTEREST
#         expectedNextEmphasis += nextEmphasis
#         # When computing expected update target, if next state is not lower state, importance ratio will be 0,
#         # so we can safely ignore this case and assume next state is always lower state
#         nextState = LOWER_STATE
#         delta = REWARD + DISCOUNT * np.dot(FEATURES[nextState, :], theta) - np.dot(FEATURES[state, :], theta)
#         expectedUpdate += 1.0 / len(STATES) * BEHAVIOR_BLACK_PROBABILITY * nextEmphasis * 1 / BEHAVIOR_BLACK_PROBABILITY * delta * FEATURES[state, :]
#     theta += alpha * expectedUpdate
#     return expectedNextEmphasis / len(STATES)
#
# # compute RMSVE for a value function parameterized by @theta
# # true value function is always 0 in this example
# def computeRMSVE(theta):
#     return np.sqrt(np.dot(np.power(np.dot(FEATURES, theta), 2), stateDistribution))
#
# # compute RMSPBE for a value function parameterized by @theta
# # true value function is always 0 in this example
# def computeRMSPBE(theta):
#     bellmanError = np.zeros(len(STATES))
#     for state in STATES:
#         for nextState in STATES:
#             if nextState == LOWER_STATE:
#                 bellmanError[state] += REWARD + DISCOUNT * np.dot(theta, FEATURES[nextState, :]) - np.dot(theta, FEATURES[state, :])
#     bellmanError = np.dot(np.asarray(projectionMatrix), bellmanError)
#     return np.sqrt(np.dot(np.power(bellmanError, 2), stateDistribution))
#
# figureIndex = 0
#
# # Figure 11.2(right), semi-gradient DP
# def figure11_2_b():
#     # Initialize the theta
#     theta = np.ones(FEATURE_SIZE)
#     theta[6] = 10
#
#     alpha = 0.01
#
#     sweeps = 1000
#     thetas = np.zeros((FEATURE_SIZE, sweeps))
#     for sweep in range(sweeps):
#         semiGradientDP(theta, alpha)
#         thetas[:, sweep] = theta
#
#     global figureIndex
#     plt.figure(figureIndex)
#     figureIndex += 1
#     for i in range(FEATURE_SIZE):
#         plt.plot(thetas[i, :], label='theta' + str(i + 1))
#     plt.xlabel('Sweeps')
#     plt.ylabel('Theta value')
#     plt.title('semi-gradient DP')
#     plt.legend()
#
# # Figure 11.6(left), temporal difference with gradient correction
# def figure11_6_a():
#     # Initialize the theta
#     theta = np.ones(FEATURE_SIZE)
#     theta[6] = 10
#     weight = np.zeros(FEATURE_SIZE)
#
#     alpha = 0.005
#     beta = 0.05
#
#     steps = 1000
#     thetas = np.zeros((FEATURE_SIZE, steps))
#     RMSVE = np.zeros(steps)
#     RMSPBE = np.zeros(steps)
#     state = np.random.choice(STATES)
#     for step in range(steps):
#         state = TDC(state, theta, weight, alpha, beta)
#         thetas[:, step] = theta
#         RMSVE[step] = computeRMSVE(theta)
#         RMSPBE[step] = computeRMSPBE(theta)
#
#     global figureIndex
#     plt.figure(figureIndex)
#     figureIndex += 1
#     for i in range(FEATURE_SIZE):
#         plt.plot(thetas[i, :], label='theta' + str(i + 1))
#     plt.plot(RMSVE, label='RMSVE')
#     plt.plot(RMSPBE, label='RMSPBE')
#     plt.xlabel('Steps')
#     plt.title('TDC')
#     plt.legend()
#
# # Figure 11.6(right), expected temporal difference with gradient correction
# def figure11_6_b():
#     # Initialize the theta
#     theta = np.ones(FEATURE_SIZE)
#     theta[6] = 10
#     weight = np.zeros(FEATURE_SIZE)
#
#     alpha = 0.005
#     beta = 0.05
#
#     sweeps = 1000
#     thetas = np.zeros((FEATURE_SIZE, sweeps))
#     RMSVE = np.zeros(sweeps)
#     RMSPBE = np.zeros(sweeps)
#     for sweep in range(sweeps):
#         expectedTDC(theta, weight, alpha, beta)
#         thetas[:, sweep] = theta
#         RMSVE[sweep] = computeRMSVE(theta)
#         RMSPBE[sweep] = computeRMSPBE(theta)
#
#     global figureIndex
#     plt.figure(figureIndex)
#     figureIndex += 1
#     for i in range(FEATURE_SIZE):
#         plt.plot(thetas[i, :], label='theta' + str(i + 1))
#     plt.plot(RMSVE, label='RMSVE')
#     plt.plot(RMSPBE, label='RMSPBE')
#     plt.xlabel('Sweeps')
#     plt.title('Expected TDC')
#     plt.legend()
#
# # Figure 11.7, expected ETD
# def figure11_7():
#     # Initialize the theta
#     theta = np.ones(FEATURE_SIZE)
#     theta[6] = 10
#
#     alpha = 0.03
#
#     sweeps = 1000
#     thetas = np.zeros((FEATURE_SIZE, sweeps))
#     RMSVE = np.zeros(sweeps)
#     emphasis = 0.0
#     for sweep in range(sweeps):
#         emphasis = expectedEmphaticTD(theta, emphasis, alpha)
#         thetas[:, sweep] = theta
#         RMSVE[sweep] = computeRMSVE(theta)
#
#     global figureIndex
#     plt.figure(figureIndex)
#     figureIndex += 1
#     for i in range(FEATURE_SIZE):
#         plt.plot(thetas[i, :], label='theta' + str(i + 1))
#     plt.plot(RMSVE, label='RMSVE')
#     plt.xlabel('Sweeps')
#     plt.title('emphatic TD')
#     plt.legend()

if __name__ == '__main__':

    TDlearning_plot()
    c_TDlearning_plot()
    plt.show()
