import sys
sys.path.insert(0, 'gym_env')
import gym
import gym_env
from configparser import ConfigParser
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

print("="*70)
print("   LIVE 3D DRONE NAVIGATION SIMULATION")
print("="*70)

# Register environment
from gym.envs.registration import register
register(
    id='airsim-env-v0',
    entry_point='gym_env.envs:AirsimGymEnv',
    max_episode_steps=1000,
)

# Load config
cfg = ConfigParser()
cfg.read('configs/config_NH_center_SimpleMultirotor_3D.ini')

print("\nInitializing environment...")
env = gym.make('airsim-env-v0')
env.set_config(cfg)
print("✅ Environment ready!")

# Initialize
obs = env.reset()
trajectory = []
goal_pos = env.dynamic_model.goal_pose[:3]

print(f"Goal position: {goal_pos}")
print("\nStarting live 3D visualization...")
print("Close the window to stop.\n")

# Setup plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    global obs, trajectory
    
    # Take action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    
    # Get current position
    current_pos = env.dynamic_model.pose[:3]
    trajectory.append(current_pos)
    
    # Clear and redraw
    ax.clear()
    
    # Plot trajectory
    if len(trajectory) > 1:
        traj_array = np.array(trajectory)
        ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 
                'b-', linewidth=2, alpha=0.7, label='Flight Path')
    
    # Plot current drone position
    ax.scatter(*current_pos, c='red', s=200, marker='o', 
               edgecolors='black', linewidths=2, label='Drone', zorder=5)
    
    # Plot goal
    ax.scatter(*goal_pos, c='green', s=300, marker='*', 
               edgecolors='black', linewidths=2, label='Goal', zorder=5)
    
    # Plot start
    if len(trajectory) > 0:
        ax.scatter(*trajectory[0], c='orange', s=150, marker='^', 
                   label='Start', zorder=5)
    
    # Labels
    ax.set_xlabel('X (meters)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z (meters)', fontsize=11, fontweight='bold')
    ax.set_title(f'Live Drone Navigation - Step {frame+1}\nReward: {reward:.2f}', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set consistent view
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([0, 15])
    
    if done:
        print(f"Episode finished at step {frame+1}!")
        return
    
    return ax,

# Run animation
ani = FuncAnimation(fig, update, frames=200, interval=100, repeat=False, blit=False)
plt.show()

print("\n✅ Simulation complete!")
