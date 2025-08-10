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

