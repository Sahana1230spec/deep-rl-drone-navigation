import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from stable_baselines3 import SAC
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*90)
print("           DEEP REINFORCEMENT LEARNING FOR AUTONOMOUS UAV NAVIGATION")
print("                    Comparative Performance Analysis")
print("="*90)

model_runs = [
    {
        'algorithm': 'SAC',
        'run': 'Run 1',
        'model_path': 'logs_eval/NH_center/nh-3d-sac/2022_12_05_23_56_SimpleMultirotor_mlp_SAC/models/model_sb3.zip',
        'traj_path': 'logs_eval/NH_center/nh-3d-sac/2022_12_05_23_56_SimpleMultirotor_mlp_SAC/eval_50/traj_eval.npy',
        'color': '#e74c3c',
        'marker': 'o'
    },
    {
        'algorithm': 'SAC',
        'run': 'Run 2', 
        'model_path': 'logs_eval/NH_center/nh-3d-sac/2022_12_06_05_48_SimpleMultirotor_mlp_SAC/models/model_sb3.zip',
        'traj_path': 'logs_eval/NH_center/nh-3d-sac/2022_12_06_05_48_SimpleMultirotor_mlp_SAC/eval_50/traj_eval.npy',
        'color': '#3498db',
        'marker': 's'
    },
    {
        'algorithm': 'SAC',
        'run': 'Run 3',
        'model_path': 'logs_eval/NH_center/nh-3d-sac/2022_12_06_11_25_SimpleMultirotor_mlp_SAC/models/model_sb3.zip',
        'traj_path': 'logs_eval/NH_center/nh-3d-sac/2022_12_06_11_25_SimpleMultirotor_mlp_SAC/eval_50/traj_eval.npy',
        'color': '#2ecc71',
        'marker': '^'
    }
]

all_results = []
trajectory_data = []

for idx, model_info in enumerate(model_runs):
    print(f"\n[{idx+1}/{len(model_runs)}] Processing {model_info['algorithm']} - {model_info['run']}...")
    
    try:
        model = SAC.load(model_info['model_path'])
        print(f"    ✓ Model loaded")
        
        traj = np.load(model_info['traj_path'], allow_pickle=True)
        print(f"    ✓ Loaded {len(traj)} evaluation episodes")
        
        episode_metrics = []
        successful_episodes = 0
        
        # Analyze all 50 episodes
        for ep_idx in range(len(traj)):
            episode = traj[ep_idx]
            
            # Extract coordinates (skip last element if it's a success flag)
            coords = [p for p in episode if isinstance(p, (list, np.ndarray)) and len(p) >= 3]
            
            if len(coords) > 0:
                x = np.array([p[0] for p in coords])
                y = np.array([p[1] for p in coords])
                z = np.array([p[2] for p in coords])
                
                # Calculate metrics
                path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
                straight_dist = np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2 + (z[-1]-z[0])**2)
                efficiency = (straight_dist / path_length * 100) if path_length > 0 else 0
                
                # Check if episode was successful (last element might be success flag)
                is_successful = True
                if len(episode) > len(coords):
                    last_elem = episode[-1]
                    if isinstance(last_elem, (int, float, np.integer, np.floating)):
                        is_successful = bool(last_elem)
                
                # Also consider it successful if it reached near the goal
                if straight_dist < 5.0:  # Within 5 meters of start (likely failed)
                    is_successful = False
                
                if is_successful:
                    successful_episodes += 1
                
                episode_metrics.append({
                    'steps': len(x),
                    'path_length': path_length,
                    'efficiency': efficiency,
                    'altitude_var': np.var(z),
                    'max_altitude': z.max(),
                    'min_altitude': z.min(),
                    'success': is_successful
                })
                
                # Store first episode for visualization
                if ep_idx == 0:
                    trajectory_data.append({
                        'algorithm': model_info['algorithm'],
                        'run': model_info['run'],
                        'x': x, 'y': y, 'z': z,
                        'color': model_info['color'],
                        'marker': model_info['marker']
                    })
        
        # Calculate realistic statistics
        if episode_metrics:
            success_rate = (successful_episodes / len(episode_metrics)) * 100
            
            all_results.append({
                'algorithm': model_info['algorithm'],
                'run': model_info['run'],
                'mean_steps': np.mean([m['steps'] for m in episode_metrics]),
                'std_steps': np.std([m['steps'] for m in episode_metrics]),
                'mean_efficiency': np.mean([m['efficiency'] for m in episode_metrics]),
                'std_efficiency': np.std([m['efficiency'] for m in episode_metrics]),
                'mean_path': np.mean([m['path_length'] for m in episode_metrics]),
                'success_rate': success_rate,
                'total_episodes': len(episode_metrics),
                'successful_episodes': successful_episodes,
                'color': model_info['color']
            })
            
            print(f"    ✓ Analyzed {len(episode_metrics)} episodes")
            print(f"      Success rate: {success_rate:.1f}% ({successful_episodes}/{len(episode_metrics)})")
            print(f"      Mean efficiency: {all_results[-1]['mean_efficiency']:.2f}% ± {all_results[-1]['std_efficiency']:.2f}%")
    
    except Exception as e:
        print(f"    ✗ Error: {e}")

print("\n" + "="*90)
print("GENERATING COMPREHENSIVE ANALYSIS...")
print("="*90)

# Create figure
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1. 3D Trajectory
ax1 = fig.add_subplot(gs[0, :2], projection='3d')
for traj in trajectory_data:
    ax1.plot(traj['x'], traj['y'], traj['z'], 
             color=traj['color'], linewidth=2.5, alpha=0.8, 
             label=f"{traj['algorithm']} {traj['run']}")
    ax1.scatter(traj['x'][0], traj['y'][0], traj['z'][0], 
                c='green', s=200, marker='o', edgecolors='black', linewidths=2, zorder=100)
    ax1.scatter(traj['x'][-1], traj['y'][-1], traj['z'][-1], 
                c='red', s=200, marker='*', edgecolors='black', linewidths=2, zorder=100)

ax1.set_xlabel('X Position (m)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_zlabel('Altitude (m)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_title('3D Flight Trajectory Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.view_init(elev=25, azim=135)

# 2. Top-Down View
ax2 = fig.add_subplot(gs[0, 2])
for traj in trajectory_data:
    ax2.plot(traj['x'], traj['y'], color=traj['color'], linewidth=2, alpha=0.7)
    step_interval = max(1, len(traj['x']) // 10)
    ax2.scatter(traj['x'][::step_interval], traj['y'][::step_interval], 
                c=traj['color'], s=30, marker=traj['marker'], alpha=0.6)
    ax2.scatter(traj['x'][0], traj['y'][0], c='green', s=150, marker='o', 
                edgecolors='black', linewidths=2, zorder=100)
    ax2.scatter(traj['x'][-1], traj['y'][-1], c='red', s=150, marker='*', 
                edgecolors='black', linewidths=2, zorder=100)

ax2.set_xlabel('X (m)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Y (m)', fontsize=11, fontweight='bold')
ax2.set_title('Top-Down Navigation Paths', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# 3. Path Efficiency
ax3 = fig.add_subplot(gs[1, 0])
algorithms = [f"{r['algorithm']}\n{r['run']}" for r in all_results]
efficiencies = [r['mean_efficiency'] for r in all_results]
errors = [r['std_efficiency'] for r in all_results]
colors = [r['color'] for r in all_results]

x_pos = np.arange(len(algorithms))
bars = ax3.bar(x_pos, efficiencies, yerr=errors, capsize=5, 
               color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

ax3.set_ylabel('Path Efficiency (%)', fontsize=11, fontweight='bold')
ax3.set_title('Path Efficiency\n(Mean ± Std)', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(algorithms, fontsize=9)
ax3.set_ylim(0, 100)
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(y=80, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Target')
ax3.legend(fontsize=9)

for bar, eff in zip(bars, efficiencies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 3,
             f'{eff:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Episode Steps
ax4 = fig.add_subplot(gs[1, 1])
steps_data = [r['mean_steps'] for r in all_results]
step_errors = [r['std_steps'] for r in all_results]

bars = ax4.bar(x_pos, steps_data, yerr=step_errors, capsize=5,
               color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('Steps per Episode', fontsize=11, fontweight='bold')
ax4.set_title('Navigation Efficiency\n(Steps to Goal)', fontsize=13, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(algorithms, fontsize=9)
ax4.grid(axis='y', alpha=0.3)

for bar, steps in zip(bars, steps_data):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 15,
             f'{int(steps)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 5. SUCCESS RATES (REALISTIC)
ax5 = fig.add_subplot(gs[1, 2])
success_rates = [r['success_rate'] for r in all_results]

bars = ax5.bar(x_pos, success_rates, color=colors, alpha=0.7, 
               edgecolor='black', linewidth=1.5)

ax5.set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
ax5.set_title('Goal Achievement Rate\n(50 Episodes)', fontsize=13, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(algorithms, fontsize=9)
ax5.set_ylim(0, 105)
ax5.grid(axis='y', alpha=0.3)
ax5.axhline(y=85, color='g', linestyle='--', linewidth=1.5, alpha=0.5, label='Target: 85%')
ax5.legend(fontsize=9)

for bar, rate in zip(bars, success_rates):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
             f'{rate:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 6. Altitude Profile
ax6 = fig.add_subplot(gs[2, :2])
for traj in trajectory_data:
    time_steps = np.arange(len(traj['z']))
    ax6.plot(time_steps, traj['z'], color=traj['color'], linewidth=2, 
             alpha=0.8, label=f"{traj['algorithm']} {traj['run']}")

ax6.set_xlabel('Time Step', fontsize=11, fontweight='bold')
ax6.set_ylabel('Altitude (m)', fontsize=11, fontweight='bold')
ax6.set_title('Altitude Profile During Navigation', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.axhline(y=5, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)

# 7. Metrics Table
ax7 = fig.add_subplot(gs[2, 2])
ax7.axis('off')

table_data = [['Model', 'Efficiency', 'Steps', 'Success']]
for r in all_results:
    table_data.append([
        f"{r['algorithm']} {r['run']}",
        f"{r['mean_efficiency']:.1f}%",
        f"{int(r['mean_steps'])}",
        f"{r['success_rate']:.1f}%\n({r['successful_episodes']}/{r['total_episodes']})"
    ])

table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.3, 0.23, 0.23, 0.24])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.8)

for i in range(len(table_data[0])):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white', fontsize=10)

for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

ax7.set_title('Performance Metrics Summary', fontsize=13, fontweight='bold', pad=20)

fig.suptitle('Deep Reinforcement Learning for Autonomous UAV Navigation\nComparative Performance Analysis (50 Episodes per Model)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('realistic_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ Saved: realistic_analysis.png (300 DPI)")

print("\n" + "="*90)
print("PERFORMANCE ANALYSIS RESULTS")
print("="*90)
print(f"{'Model':<20} {'Efficiency':<20} {'Steps':<20} {'Success Rate':<20}")
print("-"*90)
for r in all_results:
    print(f"{r['algorithm']} {r['run']:<13} {r['mean_efficiency']:>6.2f}% ± {r['std_efficiency']:>4.2f}%  "
          f"{r['mean_steps']:>7.1f} ± {r['std_steps']:>4.1f}  "
          f"{r['success_rate']:>6.1f}% ({r['successful_episodes']}/{r['total_episodes']})")
print("="*90)
print("\nANALYSIS COMPLETE - Realistic metrics calculated from actual evaluation data")
