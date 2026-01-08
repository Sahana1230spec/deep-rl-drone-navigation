import numpy as np
import matplotlib.pyplot as plt

# Load trajectory
traj = np.load('logs_eval/NH_center/nh-3d-sac/2022_12_05_23_56_SimpleMultirotor_mlp_SAC/eval_50/traj_eval.npy', allow_pickle=True)

# Get first episode
episode = traj[0]

# Extract x, y, z
x = np.array([point[0] for point in episode])
y = np.array([point[1] for point in episode])
z = np.array([point[2] for point in episode])

print(f"Extracted {len(x)} waypoints")

# 3D Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.8)
ax.scatter(x[0], y[0], z[0], c='green', s=250, marker='o', label='Start')
ax.scatter(x[-1], y[-1], z[-1], c='red', s=250, marker='X', label='End')
ax.set_xlabel('X Position (m)', fontsize=12)
ax.set_ylabel('Y Position (m)', fontsize=12)
ax.set_zlabel('Z Position (m)', fontsize=12)
ax.set_title('Autonomous Drone Navigation - 3D Trajectory', fontsize=14)
ax.legend()
ax.grid(True)
plt.savefig('drone_trajectory_3d.png', dpi=150)
print("Saved: drone_trajectory_3d.png")
plt.close()

# 2D Top View
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b-', linewidth=2)
plt.scatter(x[0], y[0], c='green', s=250, marker='o', label='Start')
plt.scatter(x[-1], y[-1], c='red', s=250, marker='X', label='End')
plt.xlabel('X Position (m)', fontsize=12)
plt.ylabel('Y Position (m)', fontsize=12)
plt.title('Drone Navigation Path - Top View', fontsize=14)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('drone_trajectory_2d.png', dpi=150)
print("Saved: drone_trajectory_2d.png")
plt.close()

# Stats
distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
print(f"\nFlight Statistics:")
print(f"  Steps: {len(x)}")
print(f"  Distance: {distance:.2f} m")
print(f"  Start: ({x[0]:.2f}, {y[0]:.2f}, {z[0]:.2f})")
print(f"  End: ({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f})")
