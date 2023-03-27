import os
import time
import argparse
import gym

from drl.envs.env_utils import get_env_params

class Trainer(object):
    def __init__(self, params):
        self.params = params


        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['batch_size'],
            'double_q': params['double_q'],
        }

        env_args = get_env_params(params['env_name'])

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            }

        self.env_name = params['env_name']
        self.env = gym.make(self.env_name)
        self.env.seed(params['seed'])
        self.env.action_space.seed(params['seed'])
        self.env.observation_space.seed(params['seed'])
        self.env._max_episode_steps = params['episode_length']
        self.max_steps = params['max_steps']
        self.num_episodes = params['num_episodes']
        self.batch_size = params['batch_size']
        self.replay_buffer_size = params['replay_buffer_size']
        self.num_agent_train_steps_per_iter = params['num_agent_train_steps_per_iter']
        self.num_critic_updates_per_agent_update = params['num_critic_updates_per_agent_update']
        self.double_q = params['double_q']
        self.learning_rate = params['learning_rate']
        self.discount = params['discount']
        self.logdir = params['logdir']
        self.scalar_log_freq = params['scalar_log_freq']
        self.video_log_freq = params['video_log_freq']
        self.save_params = params['save_params']
        self.log_wandb = params['log_wandb']

        self.params = params
        self.params['agent_class'] = DQNAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']


    def run(self):
        # Run training
        self.agent.train(self.env, self.max_steps, self.num_episodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--exp_name', type=str, default='dqn')
    parser.add_argument(
        '--env',
        type=str, 
        default='CartPole-v0',
        choices=['CartPole-v0', 'CartPole-v1', 'MountainCar-v0', 
                 'Acrobot-v1', 'LunarLander-v2', 'LunarLanderContinuous-v2', 
                 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'CarRacing-v0', 
                 'Pendulum-v0', 'MountainCarContinuous-v0', 'InvertedPendulum-v2', 
                 'InvertedDoublePendulum-v2', 'HalfCheetah-v2', 'Hopper-v2', 
                 'Walker2d-v2', 'Ant-v2', 'Humanoid-v2', 'HumanoidStandup-v2', 
                 'Swimmer-v2', 'Reacher-v2', 'Pusher-v2', 'Thrower-v2', 'Striker-v2', 
                 'FetchSlide-v1', 'FetchPickAndPlace-v1', 'FetchReach-v1', 'FetchPush-v1', 
                 'HandManipulateBlockRotateZ-v0', 'HandManipulateBlockRotateZTouchSensors-v0', 
                 'HandManipulateBlockRotateZTouchSensors-v1', 'HandManipulateBlockRotateZTouchSensors-v2', 
                 'HandManipulateBlockRotateZTouchSensors-v3', 'HandManipulateBlockRotateZTouchSensors-v4', 
                 'HandManipulateBlockRotateZTouchSensors-v5', 'HandManipulateBlockRotateZTouchSensors-v6', 
                 'HandManipulateBlockRotateZTouchSensors-v7', 'HandManipulateBlockRotateZTouchSensors-v8', 
                 'HandManipulateBlockRotateZTouchSensors-v9', 'HandManipulateBlockRotateZTouchSensors-v10', 
                 'HandManipulateBlockRotateZTouchSensors-v11', 'HandManipulateBlockRotateZTouchSensors-v12', 
                 'HandManipulateBlockRotateZTouchSensors-v13', 'HandManipulateBlockRotateZTouchSensors-v14', 
                 'HandManipulateBlockRotateZTouchSensors-v15', 'HandManipulateBlockRotateZTouchSensors-v16', 
                 'HandManipulateBlockRotateZTouchSensors-v17', 'HandManipulateBlockRotateZTouchSensors-v18', 
                 'HandManipulateBlockRotateZTouchSensors-v19', 'HandManipulateBlockRotateZTouchSensors-v20', 
                 'HandManipulateBlockRotateZTouchSensors-v21', 'HandManipulateBlockRotateZ'])
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--episode_length', '--ep_len', type=int, default=200)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--max_steps', type=int, default=1000)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--replay_buffer_size', type=int, default=int(1e6))

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--double_q', action='store_true')

    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=int(1e4))
    parser.add_argument('--video_log_freq', type=int, default=-1)

    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--log_wandb', action='store_true')

    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-3)
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
        wandb.init(project=args.exp_name, config=params)
        wandb.save(os.path.join(logdir, 'params_wandb.txt'))

    # Set GPU
    if not args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.which_gpu)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    trainer = Trainer(params)
    trainer.run()