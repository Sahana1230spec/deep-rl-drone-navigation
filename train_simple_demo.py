import sys
sys.path.insert(0, 'gym_env')
import gym
import gym_env
from gym.envs.registration import register
from configparser import ConfigParser
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

# Register environment
register(
    id='airsim-env-v0',
    entry_point='gym_env.envs:AirsimGymEnv',
    max_episode_steps=1000,
)

# Custom callback to track progress
class ProgressCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer) > len(self.episode_rewards):
            info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(info['r'])
            self.episode_lengths.append(info['l'])
            self.timesteps.append(self.num_timesteps)
            
            print(f"Episode {len(self.episode_rewards)}: Reward={info['r']:.2f}, Length={info['l']}, Timestep={self.num_timesteps}")
        return True
    
    def _on_training_end(self):
        self.plot_results()
    
    def plot_results(self):
        if len(self.episode_rewards) == 0:
            print("No episodes completed yet")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(range(len(self.episode_rewards)), self.episode_rewards, 'b-', alpha=0.6)
        if len(self.episode_rewards) >= 10:
            ax1.plot(range(len(self.episode_rewards)), 
                     np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid'), 
                     'r-', linewidth=2, label='Moving Average')
        ax1.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Episode Reward', fontsize=12, fontweight='bold')
        ax1.set_title('Training Progress - Episode Rewards', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(range(len(self.episode_lengths)), self.episode_lengths, 'g-', alpha=0.6)
        if len(self.episode_lengths) >= 10:
            ax2.plot(range(len(self.episode_lengths)), 
                     np.convolve(self.episode_lengths, np.ones(10)/10, mode='valid'), 
                     'orange', linewidth=2, label='Moving Average')
        ax2.set_xlabel('Episode', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Episode Length (steps)', fontsize=12, fontweight='bold')
        ax2.set_title('Training Progress - Episode Lengths', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
        print("\n✅ Training graph saved as 'training_progress.png'")

print("="*70)
print("   DEEP RL DRONE NAVIGATION - TRAINING DEMO")
print("="*70)

cfg = ConfigParser()
cfg.read('configs/config_NH_center_SimpleMultirotor_3D.ini')

print("\n[1/5] Creating environment...")
print(f"  Environment: {cfg.get('options', 'env_name')}")
print(f"  Dynamics: {cfg.get('options', 'dynamic_name')}")

env = gym.make('airsim-env-v0')
env.set_config(cfg)

print("✅ Environment created!")
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: {env.action_space.shape}")

print("\n[2/5] Creating PPO agent...")
model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, verbose=1)
print("✅ PPO agent created!")

callback = ProgressCallback()

print("\n[3/5] Starting training (10,000 timesteps)...")
print("-"*70)

start_time = time.time()
model.learn(total_timesteps=10000, callback=callback, progress_bar=True)
training_time = time.time() - start_time

print("-"*70)
print(f"\n✅ Training completed in {training_time:.1f} seconds!")

print("\n[4/5] Saving model...")
model.save("ppo_drone_demo")
print("✅ Model saved as 'ppo_drone_demo.zip'")

print("\n[5/5] Generating graphs...")
callback.plot_results()

if len(callback.episode_rewards) > 0:
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"  Total Episodes: {len(callback.episode_rewards)}")
    print(f"  Average Reward: {np.mean(callback.episode_rewards):.2f}")
    print(f"  Best Reward: {np.max(callback.episode_rewards):.2f}")
    print("="*70)

print("\n🎉 TRAINING COMPLETE! Files: ppo_drone_demo.zip, training_progress.png")
