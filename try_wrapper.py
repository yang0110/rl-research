import gymnasium as gym
from gymnasium.wrappers import RescaleAction, TimeLimit
from gymnasium.wrappers import FrameStackObservation, AddRenderObservation, ResizeObservation
base_env = gym.make("Hopper-v4")
print(base_env.action_space)
wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)

print(wrapped_env.action_space)
time_env = TimeLimit(wrapped_env, max_episode_steps=100)
print('time_env._max_episode_steps', time_env._max_episode_steps)

import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

env = gym.make("LunarLander-v3", render_mode="rgb_array")
time_env = TimeLimit(env, max_episode_steps=100)
print('time_env._max_episode_steps', time_env._max_episode_steps)

trigger = lambda t: t % 10 == 0
env = RecordVideo(time_env, video_folder="./save_videos1", episode_trigger=trigger, disable_logger=True)

for i in range(50):

    termination, truncation = False, False

    _ = time_env.reset(seed=123)

    while not (termination or truncation):

        obs, rew, termination, truncation, info = time_env.step(env.action_space.sample())


env.close()

len(os.listdir("./save_videos1"))



from gymnasium.wrappers import AtariPreprocessing
import ale_py
env = gym.make("ALE/Assault-v5", render_mode="rgb_array", frameskip=1) # frameskip=1 ensures no internal frameskip
print('env', env)
wrapped_env = AtariPreprocessing(
    env,
    frame_skip=4,         # This performs the frame skipping and max-pooling
    noop_max=30,          # Applies random no-op actions at reset
    screen_size=84,       # Resizes observations to 84x84
    grayscale_obs=True,   # Converts to grayscale
)

wrapped_env = FrameStackObservation(wrapped_env, stack_size=4) 
print(f"Original observation space: {env.observation_space.shape}")
print(f"Wrapped observation space: {wrapped_env.observation_space.shape}")

obs, info = wrapped_env.reset()
print(f"Observation shape after reset: {obs.shape}") # Should be (84, 84) if grayscale_obs=True

for _ in range(2): # Take 10 steps in the wrapped environment
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    print(f"Step {_ + 1}: Reward={reward}, Terminated={terminated}, Truncated={truncated}")
    if terminated or truncated:
        obs, info = wrapped_env.reset()
        print("Episode reset.")

