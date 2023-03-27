import random
from collections import namedtuple

import gym
import numpy as np
from torch import nn
import torch.optim as optim

from drl.infrastructure.atari_wrappers import wrap_deepmind
from gym.envs.registration import register

import torch



def get_env_params(env_name):
    if env_name in ['MsPacman-v0', 'PongNoFrameskip-v4']:
        kwargs = {
            'learning_starts': 50000,
            'target_update_freq': 10000,
            'replay_buffer_size': int(1e6),
            'num_timesteps': int(2e8),
            'q_func': create_atari_q_network,
            'learning_freq': 4,
            'grad_norm_clipping': 10,
            'input_shape': (84, 84, 4),
            'env_wrappers': wrap_deepmind,
            'frame_history_len': 4,
            'gamma': 0.99,
        }
        kwargs['optimizer_spec'] = atari_optimizer(kwargs['num_timesteps'])
        kwargs['exploration_schedule'] = atari_exploration_schedule(kwargs['num_timesteps'])

    elif env_name == 'LunarLander-v3':
        def lunar_empty_wrapper(env):
            return env
        kwargs = {
            'optimizer_spec': lander_optimizer(),
            'q_func': create_lander_q_network,
            'replay_buffer_size': 50000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_starts': 1000,
            'learning_freq': 1,
            'frame_history_len': 1,
            'target_update_freq': 3000,
            'grad_norm_clipping': 10,
            'lander': True,
            'num_timesteps': 500000,
            'env_wrappers': lunar_empty_wrapper
        }
        kwargs['exploration_schedule'] = lander_exploration_schedule(kwargs['num_timesteps'])

    else:
        raise NotImplementedError

    return kwargs

def register_custom_envs():
    from gym.envs.registration import registry
    if 'LunarLander-v3' not in registry.env_specs:
        register(
            id='LunarLander-v3',
            entry_point='cs285.envs.box2d.lunar_lander:LunarLander',
            max_episode_steps=1000,
            reward_threshold=200,
        )