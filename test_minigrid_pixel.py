import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
import random

print('minigrid')
env = gym.make("MiniGrid-Unlock-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)
obs, info = env.reset()
action_space = env.action_space
obs_space = env.observation_space
print(action_space)
print(obs_space)

done = False
step_num = 0
while not done:
    action = action_space.sample()
    obs, reward, done, _, info = env.step(action)
    step_num += 1
    print(step_num, reward, done, action)
    if step_num > 2:
      break