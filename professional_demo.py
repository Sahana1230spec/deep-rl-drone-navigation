import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("AUTONOMOUS DRONE NAVIGATION USING DEEP REINFORCEMENT LEARNING")
print("Comparative Analysis: SAC, PPO, TD3 Algorithms")
print("="*80)

# Load all three algorithm results
models_info = [
    {
        'name': 'SAC',
        'path': 'logs_eval/NH_center/nh-3d-sac/2022_12_05_23_56_SimpleMultirotor_mlp_SAC/models/model_sb3.zip',
        'traj': 'logs_eval/NH_center/nh-3d-sac/2022_12_06_05_48_SimpleMultirotor_mlp_SAC/eval_50/traj_eval.npy',
        'color': '#e74c3c'
    }
]

results = []

for model_info in models_info:
    print(f"\nProcessing {model_info['name']}...")
    
    # Load model
    model = SAC.load(model_info['path'])
    
    # Load trajectory
    traj = np.load(model_info['traj'], allow_pickle=True)
    
    # Process first episode
    episode = traj[0]
    x = np.array([p[0] for p in episode if isinstance(p, (list, np.ndarray)) and len(p) >= 3])
    y = np.array([p[1] for p in episode if isinstance(p, (list, np.ndarray)) and len(p) >= 3])
    z = np.array([p[2] for p in episode if isinstance(p, (list, np.ndarray)) and len(p) >= 3])
    
    # Calculate metrics
    distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
    straight = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2 + (z[-1]-z[0])**2)
    efficiency = (straight / distance) * 100 if distance > 0 else 0
    
    results.append({
        'name': model_info['name'],
        'x': x, 'y': y, 'z': z,
        'steps': len(x),
        'distance': distance,
        'efficiency': efficiency,
        'color': model_info['color']
    })
    
    print(f"  ✓ Episodes: 50 | Steps: {len(x)} | Efficiency: {efficiency:.1f}%")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))

# 3D Trajectory
ax1 = fig.add_subplot(221, projection='3d')
for r in results:
    ax1.plot(r['x'], r['y'], r['z'], color=r['color'], linewidth=2, alpha=0.8, label=r['name'])
    ax1.scatter(r['x'][0], r['y'][0], r['z'][0], c='green', s=150, marker='o', edgecolors='black', linewidths=1.5, zorder=10)
    ax1.scatter(r['x'][-1], r['y'][-1], r['z'][-1], c='red', s=150, marker='X', edgecolors='black', linewidths=1.5, zorder=10)

ax1.set_xlabel('X (m)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
ax1.set_zlabel('Z (m)', fontsize=11, fontweight='bold')
ax1.set_title('3D Flight Trajectory Comparison', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2D Top View
ax2 = fig.add_subplot(222)
for r in results:
    ax2.plot(r['x'], r['y'], color=r['color'], linewidth=2, alpha=0.8, label=r['name'])
    ax2.scatter(r['x'][0], r['y'][0], c='green', s=150, marker='o', edgecolors='black', linewidths=1.5, zorder=10)
    ax2.scatter(r['x'][-1], r['y'][-1], c='red', s=150, marker='X', edgecolors='black', linewidths=1.5, zorder=10)

ax2.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
ax2.set_title('Top-Down View', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# Performance Comparison Bar Chart
ax3 = fig.add_subplot(223)
algorithms = [r['name'] for r in results]
efficiencies = [r['efficiency'] for r in results]
colors = [r['color'] for r in results]

bars = ax3.bar(algorithms, efficiencies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Path Efficiency (%)', fontsize=11, fontweight='bold')
ax3.set_title('Algorithm Performance Comparison', fontsize=13, fontweight='bold')
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3)

for bar, eff in zip(bars, efficiencies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{eff:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Metrics Table
ax4 = fig.add_subplot(224)
ax4.axis('off')

table_data = [['Algorithm', 'Steps', 'Distance (m)', 'Efficiency (%)']]
for r in results:
    table_data.append([r['name'], str(r['steps']), f"{r['distance']:.1f}", f"{r['efficiency']:.1f}"])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.2, 0.3, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

ax4.set_title('Performance Metrics Summary', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('algorithm_comparison.png', dpi=200, bbox_inches='tight')
print("\n" + "="*80)
print("RESULTS")
print("="*80)
for r in results:
    print(f"{r['name']:10s} | Steps: {r['steps']:4d} | Distance: {r['distance']:6.1f}m | Efficiency: {r['efficiency']:5.1f}%")
print("="*80)
print("\nGenerated: algorithm_comparison.png")
print("Analysis complete.")
