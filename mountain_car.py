import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

# Custom reward wrapper to encourage the car to reach the flag
class CustomMountainCarEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = gym.make("MountainCar-v0")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Custom reward: Encourage progress towards the flag
        position, velocity = observation
        reward = position  # Reward proportional to the car's position

        if terminated:
            reward += 10  # Bonus reward for reaching the flag

        return observation, reward, terminated, truncated, info

# Create the customized environment
env = DummyVecEnv([lambda: CustomMountainCarEnv()])

# Define the DQN model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.001,
    buffer_size=100000,  # Larger buffer for more transitions
    learning_starts=1000,
    batch_size=64,
    tau=0.8,  # Target network update coefficient
    gamma=0.99,  # Discount factor for future rewards
    train_freq=4,  # Train every 4 steps
    target_update_interval=500,  # Update target network every 500 steps
    policy_kwargs=dict(net_arch=[256, 256]),  # Two hidden layers with 256 neurons each
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

# Set up evaluation callback
eval_callback = EvalCallback(
    env,
    best_model_save_path="./logs/",
    log_path="./logs/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

# Train the model
model.learn(total_timesteps=100000, callback=eval_callback)

# Save the trained model
model.save("dqn_mountain_car")
print("Model saved as dqn_mountain_car.zip")

# Test the trained model
test_env = gym.make("MountainCar-v0", render_mode="human")
observation, _ = test_env.reset()

for _ in range(1000):
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = test_env.step(action)

    if terminated:
        print("Flag reached!")
        break

test_env.close()
