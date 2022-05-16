import gym
from atariari.benchmark.wrapper import AtariARIWrapper

env = AtariARIWrapper(gym.make('MsPacmanNoFrameskip-v4'))
env.render()
obs = env.reset()
obs, reward, done, info = env.step(1)
input()