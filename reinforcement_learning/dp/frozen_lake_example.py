import numpy as np
import torch
import gymnasium as gym
import time

# 初始化环境
env = gym.make("FrozenLake-v1", render_mode="human", is_slippery=True)
env = env.unwrapped
state, info = env.reset()  # 重置环境，获取初始状态

done = False
while not done:
    action = env.action_space.sample()  # 随机选择一个动作
    next_state, reward, done, truncated, info = env.step(action)  # 执行动作
    print(f"State: {state}, Action: {action}, Reward: {reward}, info: {info}, Next State: {next_state}")
    state = next_state
    time.sleep(0.5)

env.close()
