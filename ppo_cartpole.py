log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)
from stable_baselines3 import PPO
model=PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
import pandas as pd
y_smooth=y.rolling(window=10).mean()
plt.plot(x, y, alpha=0.3, label="Raw Reward")
plt.plot(x, y_smooth, color="red",label="Moving Avg(10)")
plt.legend()
plt.show()
