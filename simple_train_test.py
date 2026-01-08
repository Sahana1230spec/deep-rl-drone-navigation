import gym
import gym_env  # This is the correct import!
from configparser import ConfigParser

# Load config
cfg = ConfigParser()
cfg.read('configs/config_NH_center_SimpleMultirotor_3D.ini')

print("Creating environment...")
print(f"Environment: {cfg.get('options', 'env_name')}")
print(f"Dynamics: {cfg.get('options', 'dynamic_name')}")

# Create environment
env = gym.make('airsim-env-v0')
env.set_config(cfg)

print(f"\n✅ Environment created successfully!")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# Test a few steps
print("\n🚁 Running simulation test...")
obs = env.reset()
print(f"Initial observation shape: {obs.shape}")

for step in range(10):
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    print(f"Step {step+1}: reward={reward:.2f}, done={done}")
    
    if done:
        print("Episode finished!")
        break

print("\n✅ Simulation test complete! Your environment works!")
