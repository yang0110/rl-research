import gymnasium as gym
import ale_py

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

print('mujoco')
import gymnasium as gym
env = gym.make('Reacher-v5')
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

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from dm_control import suite
import numpy as np
import shimmy

domain_list = ['cartpole', 'acrobot', 'reacher', 'walker', 'cheetah', 'humanoid']

dm_control_domains = {
    "acrobot": [
      "swingup"
    ],
    "ball_in_cup": [
      "catch"
    ],
    "cartpole": [
      "balance",
      "swingup",
      "swingup_sparse",
      "two_poles",
      "three_poles"
    ],
    "cheetah": [
      "run"
    ],
    "finger": [
      "spin",
      "turn_easy",
      "turn_hard"
    ],
    "fish": [
      "upright",
      "swim"
    ],
    "hopper": [
      "stand",
      "hop"
    ],
    "humanoid": [
      "stand",
      "walk",
      "run",
      "run_pure_state",
    ],
    "manipulator": [
      "bring_ball",
      "bring_peg",
      "insert_ball",
      "insert_peg",
    ],
    "pendulum": [
      "swingup"
    ],
    "point_mass": [
      "easy",
      "hard"
    ],
    "reacher": [
      "easy",
      "hard"
    ],
    "swimmer": [
      "swimmer6",
      "swimmer15"
    ],
    "walker": [
      "stand",
      "walk",
      "run"
    ],
    "dog": [
      "walk",
      "run",
      "trot",
      "fetch",
      "stand"
    ]
  }

for domain_name, task_name_list in dm_control_domains.items():
    print(f'Loading domain: {domain_name}')
    print(f'Tasks: {task_name_list}')
    for task_name in task_name_list:
        print(f'Loading task: {task_name}')
        env = suite.load(domain_name=domain_name, task_name=task_name)
        # env = shimmy.DmControlCompatibilityV0(env)
        # print(env.observation_space)
        env = FlattenObservation(shimmy.DmControlCompatibilityV0(env))
        print(env)
        action_space = env.action_space
        obs_space = env.observation_space
        # print('action_space', action_space)
        print('obs_space', obs_space)

        ob, info = env.reset()
        done = False
        step_num = 0
        while not done and step_num<3:
            action = action_space.sample()
            obs, reward, done, _, info = env.step(action)
            step_num += 1
            # print(step_num, reward, done, action)
            if step_num > 2:
              break

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
    
# import procgen
# from procgen import ProcgenEnv
# env = ProcgenEnv(num_envs=1, env_name="starpilot")
# # env = gym.make("procgen:procgen-coinrun-v0", render_mode="human")
# # num_levels=0 - The number of unique levels that can be generated. Set to 0 to use unlimited levels.
# # start_level=0 - The lowest seed that will be used to generated levels. 'start_level' and 'num_levels' fully specify the set of possible levels.
# obs = env.reset()
# print(obs)  # (64, 64, 3)