import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Create and wrap the environment
def make_env():
    return gym.make("CartPole-v1")

env = DummyVecEnv([make_env])  # Required for stable-baselines3 compatibility

# Create a PPO model
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=1024)

# Train the PPO model
model.learn(total_timesteps=200000)  # Train for 200,000 steps

# Test the trained model in a human-rendered environment
test_env = gym.make("CartPole-v1", render_mode="human")
observation, info = test_env.reset()

for _ in range(1000):
    # Use the trained model to predict actions
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = test_env.step(action)

    # Reset if the episode ends
    if terminated or truncated:
        observation, info = test_env.reset()

test_env.close()