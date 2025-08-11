import gymnasium as gym
import ale_py
import cv2
import numpy as np
print('atari')
env = gym.make("ALE/Assault-v5", render_mode="rgb_array")
# env = gym.make("ALE/Breakout-v5")
print(env)
action_space = env.action_space
obs_space = env.observation_space

print(action_space)
print(obs_space)

ob, info = env.reset()
done = False
step_num = 0
while not done:
    action = action_space.sample()
    obs, reward, done, _, info = env.step(action)
    step_num += 1
    print(step_num, reward, done, action)
    if step_num > 2:
      break

class PreprocessAtari(gym.ObservationWrapper):
    def __init__(self, env):
        super(PreprocessAtari, self).__init__(env)
        self.img_size = (84, 84)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)
        return obs

env = PreprocessAtari(env)
action_space = env.action_space
obs_space = env.observation_space

print(action_space)
print(obs_space)

ob, info = env.reset()
done = False
step_num = 0
while not done:
    action = action_space.sample()
    obs, reward, done, _, info = env.step(action)
    step_num += 1
    print(step_num, reward, done, action)
    if step_num > 2:
      break 