import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os

class DualTaskEnv(gym.Env):
    def __init__(self):
        super().__init__()
        #balance pole and maintain target velocity
        self.cart_target_velocity = 1.0
        self.gravity = 9.8
        self.pole_length = 0.5
        self.max_force = 10.0
        
        # State space: [x, x_dot, theta, theta_dot]
        self.observation_space = spaces.Box(
            low=np.array([-4.8, -10, -0.418, -10]),
            high=np.array([4.8, 10, 0.418, 10]),
            dtype=np.float32
        )
        
        # Action space: [-max_force, max_force]
        self.action_space = spaces.Box(
            low=-self.max_force,
            high=self.max_force,
            shape=(1,),
            dtype=np.float32
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state, {}

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = float(action)
        

        temp = (force + self.pole_length * theta_dot**2 * np.sin(theta)) / (1.0 + self.pole_length * np.sin(theta)**2)
        theta_acc = (self.gravity * np.sin(theta) - np.cos(theta) * temp) / (self.pole_length * (4.0/3.0 - np.cos(theta)**2))
        x_acc = temp - self.pole_length * theta_acc * np.cos(theta)
        
        x = x + x_dot * 0.02
        x_dot = x_dot + x_acc * 0.02
        theta = theta + theta_dot * 0.02
        theta_dot = theta_dot + theta_acc * 0.02
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        # Dual reward calculation
        balance_reward = 1.0 - abs(theta) - 0.1 * abs(theta_dot)
        velocity_reward = -abs(x_dot - self.cart_target_velocity)
        reward = balance_reward + 0.5 * velocity_reward
        
        terminated = bool(
            x < -4.8 or x > 4.8 or
            theta < -0.418 or theta > 0.418
        )
        
        return self.state, reward, terminated, False, {}

env = DummyVecEnv([lambda: DualTaskEnv()])

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    n_steps=1024,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)
model.learn(total_timesteps=500000)
model.save("dual_task_ppo")

import zipfile
with zipfile.ZipFile('dual_task_model.zip', 'w') as zipf:
    zipf.write('dual_task_ppo.zip')

# Print verification
print("Model zip file exists:", os.path.exists("dual_task_model.zip"))
print("Original model file exists:", os.path.exists("dual_task_ppo.zip"))

test_env = DualTaskEnv()
obs, _ = test_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)
    if done:
        obs, _ = test_env.reset()c