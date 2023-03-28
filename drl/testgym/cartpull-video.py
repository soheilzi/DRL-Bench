import gym
import numpy as np
from gym import wrappers

env = gym.make('MountainCarContinuous-v0')
env.seed = 0

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

env = wrappers.RecordVideo(env, './cartpull-video', video_length=100)

# Run single episode
observation = env.reset()
for t in range(100):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

# Close environment
env.close()