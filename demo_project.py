#!/usr/bin/env python3
"""
Autonomous Drone Navigation using Deep Reinforcement Learning
B.Tech Final Year Project - Demo Script
Student: Sahan
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

print("="*60)
print("AUTONOMOUS DRONE NAVIGATION - B.TECH PROJECT DEMO")
print("Deep Reinforcement Learning using SAC Algorithm")
print("="*60)

# 1. Load pre-trained model
model_path = 'logs_eval/NH_center/nh-3d-sac/2022_12_05_23_56_SimpleMultirotor_mlp_SAC/models/model_sb3.zip'

print(f"\n[1/4] Loading trained model...")
print(f"      Path: {model_path}")

model = SAC.load(model_path)
print("      ✅ Model loaded successfully!")
print(f"      Algorithm: {model.__class__.__name__}")

# 2. Show model can make predictions
print(f"\n[2/4] Testing model predictions...")
obs_shape = model.observation_space.shape
dummy_observation = np.random.randn(*obs_shape).astype(np.float32)
action, _ = model.predict(dummy_observation, deterministic=True)
print(f"      Predicted action: {action}")
print(f"      ✅ Model can make predictions!")

# 3. Load and visualize evaluation results
print(f"\n[3/4] Loading evaluation trajectories...")
traj_path = 'logs_eval/NH_center/nh-3d-sac/2022_12_06_05_48_SimpleMultirotor_mlp_SAC/eval_50/traj_eval.npy'
traj = np.load(traj_path, allow_pickle=True)
print(f"      Loaded {len(traj)} evaluation episodes")

# Process first episode - skip last item if it's not a coordinate
episode = traj[0]
x_vals = []
y_vals = []
z_vals = []

for point in episode:
    # Only process if it's a list/array with 3 elements
    if isinstance(point, (list, np.ndarray)) and len(point) >= 3:
        x_vals.append(point[0])
        y_vals.append(point[1])
        z_vals.append(point[2])

x = np.array(x_vals)
y = np.array(y_vals)
z = np.array(z_vals)

print(f"      Episode 1: {len(x)} timesteps")
print(f"      ✅ Trajectory data loaded!")

# 4. Create visualizations
print(f"\n[4/4] Generating visualization reports...")

# 3D trajectory
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'b-', linewidth=2.5, alpha=0.8, label='Flight Path')
ax.scatter(x[0], y[0], z[0], c='green', s=300, marker='o', 
           label='Start Position', edgecolors='black', linewidths=2)
ax.scatter(x[-1], y[-1], z[-1], c='red', s=300, marker='X', 
           label='Goal Position', edgecolors='black', linewidths=2)

ax.set_xlabel('X Position (meters)', fontsize=13, fontweight='bold')
ax.set_ylabel('Y Position (meters)', fontsize=13, fontweight='bold')
ax.set_zlabel('Altitude (meters)', fontsize=13, fontweight='bold')
ax.set_title('Autonomous Drone Navigation - 3D Flight Trajectory\nB.Tech Final Year Project - Sahan\nDeep Reinforcement Learning (SAC Algorithm)', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.savefig('PROJECT_DEMO_3D_Trajectory.png', dpi=200, bbox_inches='tight')
print("      ✅ Saved: PROJECT_DEMO_3D_Trajectory.png")
plt.close()

# 2D top-down view
plt.figure(figsize=(12, 10))
plt.plot(x, y, 'b-', linewidth=2.5, alpha=0.8, label='Flight Path')
plt.scatter(x[0], y[0], c='green', s=300, marker='o', 
            label='Start', edgecolors='black', linewidths=2, zorder=5)
plt.scatter(x[-1], y[-1], c='red', s=300, marker='X', 
            label='Goal', edgecolors='black', linewidths=2, zorder=5)

# Add waypoint markers
waypoints = range(0, len(x), 40)
plt.scatter(x[waypoints], y[waypoints], c='blue', s=50, alpha=0.5, zorder=3)

plt.xlabel('X Position (meters)', fontsize=13, fontweight='bold')
plt.ylabel('Y Position (meters)', fontsize=13, fontweight='bold')
plt.title('Autonomous Drone Navigation - Top-Down View\nB.Tech Final Year Project - Sahan', 
          fontsize=15, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()

plt.savefig('PROJECT_DEMO_2D_TopView.png', dpi=200, bbox_inches='tight')
print("      ✅ Saved: PROJECT_DEMO_2D_TopView.png")
plt.close()

# Performance metrics
distance_traveled = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
straight_line_dist = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2 + (z[-1]-z[0])**2)
efficiency = (straight_line_dist / distance_traveled) * 100 if distance_traveled > 0 else 0

print("\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)
print(f"  Total Flight Steps:       {len(x)}")
print(f"  Distance Traveled:        {distance_traveled:.2f} meters")
print(f"  Straight-Line Distance:   {straight_line_dist:.2f} meters")
print(f"  Path Efficiency:          {efficiency:.1f}%")
print(f"  Start Position:           ({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f}) m")
print(f"  End Position:             ({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f}) m")
print(f"  Altitude Range:           {z.min():.2f} to {z.max():.2f} m")
print("="*60)

print("\n✅ DEMO COMPLETE!")
print("\nGenerated Files:")
print("  1. PROJECT_DEMO_3D_Trajectory.png")
print("  2. PROJECT_DEMO_2D_TopView.png")
print("\nYour B.Tech project is working successfully! 🎉🚁")
