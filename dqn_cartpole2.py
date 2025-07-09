import gymnasium as gym
import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make("CartPole-v1")
env = Monitor(env, log_dir)

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

monitor_file = os.path.join(log_dir, [f for f in os.listdir(log_dir) if f.endswith('.csv')][0])
data=pd.read_csv(monitor_file, skiprows=1)
x = np.cumsum(data["l"])
y = data["r"]
plt.figure(figsize=(10, 5))
plt.plot(x, y, label="Episode Reward")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("奖励随训练轮数变化曲线")
plt.grid(True)
plt.legend()
plt.show()

