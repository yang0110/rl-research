
# export MUJOCO_GL=egl
import gymnasium as gym
from dm_control import suite
from dm_control.suite.wrappers import pixels
from shimmy import DmControlCompatibilityV0
import numpy as np

domain_name = "cheetah"
task_name = "run"

# 1. Load the native dm_control environment first.
dm_env = suite.load(domain_name, task_name)
dm_env_pixels = pixels.Wrapper(dm_env, pixels_only=True)
env = DmControlCompatibilityV0(
    dm_env_pixels,
    render_mode="rgb_array",
    render_kwargs={'width': 256, 'height': 256}
)

observation, info = env.reset()

print("\n--- Environment Information ---")
print(f"Initial observation keys: {observation.keys()}")
print(f"\nObservation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

for i in range(3):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}:")
    print(f"  - Reward: {reward}")
    print(f"  - Terminated: {terminated}, Truncated: {truncated}")
    print("-" * 20)


env.close()
print("\nEnvironment closed successfully.")
