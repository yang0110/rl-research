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
