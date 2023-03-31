from collections import OrderedDict
import time
import gym
from gym import wrappers
from drl.agents.dqn_agent import DQNAgent
from drl.infrastructure.atari_wrappers import ReturnWrapper

import os

import numpy as np
import haiku as hk
import jax
import jax.numpy as jnp
from jax import random
import optax

from drl.infrastructure import utils

from drl.infrastructure.logger import Logger
from drl.infrastructure import pytorch_util as ptu



class RL_Trainer(object):
    def __init__(self, params):
        self.params = params
        self.agent_class = params['agent_class']
        self.agent_params = params['agent_params']
        self.env_name = params['env_name']
        self.logger = Logger(self.params['logdir'])

        # remove later:
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        # Create environment

        self.render_mode = 'rgb_array'
        if self.params['video_log_freq'] > 0:
            self.episode_trigger = lambda episode: episode % self.params['video_log_freq'] == 0
        else:
            self.episode_trigger = lambda episode: False

        self.env = gym.make(self.env_name, max_episode_steps=self.params['episode_length'], render_mode = self.render_mode)
        self.env.seed = self.params['seed']

        self.env = wrappers.RecordEpisodeStatistics(self.env, deque_size=1000)
        self.env = ReturnWrapper(self.env)
        if 'env_wrappers' in self.params:
            self.env = params['env_wrappers'](self.env)

        if self.params['video_log_freq'] > 0:
            self.env = wrappers.RecordVideo(self.env, os.path.join(self.params['logdir'], "gym"), episode_trigger=self.episode_trigger)

        self.max_steps = params['max_steps']
        self.num_episodes = params['num_episodes']

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params['agent_params']['discrete'] = discrete

        # Observation and action sizes

        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim
        self.params['agent_params']['img'] = img

        # simulation timestep, will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30 # This is not actually used when using the Monitor wrapper
        elif 'video.frames_per_second' in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10

        # Create agent
        self.agent = self.agent_class(self.env, self.agent_params)

    def run_dqn_training_loop(self):
        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        print_period = 1000 if isinstance(self.agent, DQNAgent) else 1

        for itr in range(self.num_episodes):
            if itr % print_period == 0:
                print("\n\n********** Iteration %i ************"%itr)
            
            # decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False
            
            self.rollout_agent()
            train_logs = self.train_agent()

            if self.logmetrics or self.logvideo:
                print("Evaluating agent...")
                self.perform_logging(itr, train_logs)
    
    def rollout_agent(self, render=False):
        if isinstance(self.agent, DQNAgent):
            self.agent.step_env()
            self.total_envsteps += 1
        else:
            paths, ens_steps = self.collect_trajectory(number_of_trajectories=self.params['num_trajectory_train'])
            self.total_envsteps += ens_steps
            self.agent.add_to_replay_buffer(paths)
            
    def train_agent(self):
        all_logs = []
        for train_step in range(self.params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['batch_size'])
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
        return all_logs
    

    def collect_trajectory(self, number_of_trajectories, evaluate=False):
        timesteps_this_batch = 0
        paths = []
        for i in range(number_of_trajectories):
            path = utils.sample_trajectory(env=self.env,
                                    policy=self.agent.actor,
                                    max_path_length=self.params['episode_length'],
                                    replay_buffer=self.agent.replay_buffer,
                                    render=evaluate)
            timesteps_this_batch += utils.get_pathlength(path)
            paths.append(path)
            

        return paths, timesteps_this_batch

    def perform_logging(self, itr, all_logs):
        last_log = all_logs[-1]

        paths, _ = self.collect_trajectory(self.params['num_trajectory_eval'] ,evaluate=True)
        if self.logvideo != None:
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(paths, itr, fps=self.fps, max_videos_to_save=2,
                                            video_title='eval_rollouts')

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            eval_returns = [path["reward"].sum() for path in paths]

            # episode lengths, for logging
            eval_ep_lens = [len(path["reward"]) for path in paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)


            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(eval_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
