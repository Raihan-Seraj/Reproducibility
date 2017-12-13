import gym
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import pdb
import baselines.common.tf_util as U

from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule
import matplotlib.pyplot as plt
import gridworld10x10 as gd

def model(inpt, num_actions, scope, reuse=False):
    """This model takes as input an observation and returns values of all actions."""
    with tf.variable_scope(scope, reuse=reuse):
        
        out = inpt
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.tanh)
        out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)
        return out


if __name__ == '__main__':
    with U.make_session(8):
        # Create the environment
        env = gd.GridworldEnv()
        # Create all the functions necessary to train the model
        act, train, update_target, debug1 = deepq.build_train(
            make_obs_ph=lambda name: U.BatchInput(np.array([1,1]).shape, name=name),
            q_func=model,
            num_actions=env.action_space.n,
            optimizer=tf.train.RMSPropOptimizer(learning_rate=10e-4),
        )
        # Create the replay buffer
        replay_buffer = ReplayBuffer(50000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)
        num_experiments=10
        average_rewards=[]
        for avg in range(num_experiments):
            # Initialize the parameters and copy them to the target network.
            U.initialize()
            update_target()

            episode_rewards = [0.0]
            obs = env.reset()
            obs=gd.decodeState(obs)
            for t in itertools.count():
                # Take action and update exploration to the newest value
                #pdb.set_trace()
                action = act(obs[None], update_eps=exploration.value(t))[0]
                new_obs, rew, done, _ = env.step(action)
                temp_obs=obs
                new_obs=gd.decodeState(new_obs)
                # Store transition in the replay buffer.
                replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs
                
                episode_rewards[-1] += rew
                if done:
                    obs = env.reset()
                    obs=gd.decodeState(obs)
                    episode_rewards.append(0)

                is_solved = len(episode_rewards)==1000#t > 100 and np.mean(episode_rewards[-101:-1]) >= 200 and len(episode_rewards)>=100
                if is_solved:
                    # Show off the result
                    #env.render()
                    break
                else:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                    if t > 1000:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                        train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                    # Update target network periodically.
                    if t % 1000 == 0:
                        update_target()

                if done and len(episode_rewards) % 10 == 0:
                    logger.record_tabular("steps", t)
                    logger.record_tabular("episodes", len(episode_rewards))
                    logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                    logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                    logger.record_tabular("Number of runs",avg)
                    logger.record_tabular("Qvalue",np.max(debug1['q_values'](temp_obs[None])))
                    logger.dump_tabular()
                #pdb.set_trace()
            average_rewards.append(episode_rewards)
        episode_rewards_mean=np.mean(np.array(average_rewards),axis=0)
        episode_rewards_var=np.var(np.array(average_rewards),axis=0)

        np.save("mean_rewards_gd.npy",episode_rewards_mean)
        np.save("variance_rewards_gd.npy",episode_rewards_var)
        plt.plot(episode_rewards_mean.tolist())
        plt.show()
