import os
import time
import argparse
import gym

from drl.infrastructure.dqn_utils import get_env_kwargs
from drl.infrastructure.rl_trainer import RL_Trainer
from drl.agents.dqn_agent import DQNAgent

class Trainer(object):
    def __init__(self, params):
        self.params = params


        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['batch_size'],
            'double_q': params['double_q'],
        }

        env_args = get_env_kwargs(params['env_name'])
        
        self.agent_params = {**train_args, **env_args, **params}

        self.params = params
        self.params['agent_class'] = DQNAgent
        self.params['agent_params'] = self.agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        self.rl_trainer = RL_Trainer(self.params)

    def run(self):
        # Run training
        self.rl_trainer.run_dqn_training_loop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--env_name',
        '--env',
        type=str, 
        default='CartPole-v1',
        choices=['CartPole-v0', 'CartPole-v1', 'MountainCar-v0', 
                 'Acrobot-v1', 'LunarLander-v2', 'MsPacman-v0', 'PongNoFrameskip-v4', 'BreakoutNoFrameskip-v4'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='dqn_test')

    parser.add_argument('--episode_length', '--ep_len', type=int, default=200)
    parser.add_argument('--num_episodes', type=int, default=1000000)
    parser.add_argument('--max_steps', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--replay_buffer_size', type=int, default=int(1e6))

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--num_trajectory_eval', type=int, default=10)
    parser.add_argument('--num_agent_rollout_steps_per_iter', type=int, default=1)
    parser.add_argument('--double_q', action='store_true')

    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=int(1e4))
    parser.add_argument('--video_log_freq', type=int, default=-1)

    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--log_wandb', action='store_true')

    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-4)
    parser.add_argument('--discount', '--gamma', type=float, default=0.99)

    args = parser.parse_args()
    params = vars(args)

    # Create directory for saving experiment data
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    # Save params
    if args.save_params:
        with open(os.path.join(logdir, 'params.txt'), 'w') as f:
            for key, value in params.items():
                f.write('{}: {}\n'.format(key, value))
    
    # Log to wandb
    if args.log_wandb:
        import wandb
        wandb.init(project="drl-bench", entity="soheilzi", config=params)
        wandb.save(os.path.join(logdir, 'params_wandb.txt'))

    # Set GPU
    if not args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.which_gpu)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    trainer = Trainer(params)
    trainer.run()