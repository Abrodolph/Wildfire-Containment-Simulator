"""Generate training dashboard from training/training_stats.json."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open('training/training_stats.json') as f:
    stats = json.load(f)

steps = [s['step'] for s in stats]
rewards = [s['mean_reward'] for s in stats]
tiers = [s['tier'] for s in stats]

TIER_COLORS = {'easy': '#4CAF50', 'medium': '#FF9800', 'hard': '#F44336'}
TIER_LABELS = {'easy': 'Easy', 'medium': 'Medium', 'hard': 'Hard'}

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle('GRPO Training — Wildfire Incident Commander (Qwen-2.5-7B)', fontsize=14, fontweight='bold')

# Top panel: reward curve colored by tier
for i in range(len(steps) - 1):
    color = TIER_COLORS[tiers[i]]
    ax1.plot(steps[i:i+2], rewards[i:i+2], color=color, linewidth=1.5, alpha=0.8)

# Rolling average
window = 10
rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
rolling_steps = steps[window-1:]
ax1.plot(rolling_steps, rolling, 'k--', linewidth=2, label=f'{window}-step rolling avg', alpha=0.7)

# Baseline lines
ax1.axhline(7.53, color='#2196F3', linestyle=':', linewidth=1.5, label='Heuristic easy (+7.53)')
ax1.axhline(6.31, color='#9C27B0', linestyle=':', linewidth=1.5, label='Heuristic medium (+6.31)')
ax1.axhline(4.74, color='#795548', linestyle=':', linewidth=1.5, label='Heuristic hard (+4.74)')

# Promotion markers
ax1.axvline(53, color='#FF9800', linestyle='--', linewidth=1, alpha=0.7)
ax1.axvline(63, color='#F44336', linestyle='--', linewidth=1, alpha=0.7)
ax1.text(53, ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else 1, '→medium', color='#FF9800', fontsize=8, rotation=90, va='bottom')
ax1.text(63, ax1.get_ylim()[0] if ax1.get_ylim()[0] > 0 else 1, '→hard', color='#F44336', fontsize=8, rotation=90, va='bottom')

legend_patches = [mpatches.Patch(color=c, label=TIER_LABELS[t]) for t, c in TIER_COLORS.items()]
ax1.legend(handles=legend_patches + ax1.lines[-4:], loc='lower right', fontsize=8)
ax1.set_ylabel('Mean Batch Reward')
ax1.set_ylim(bottom=0)
ax1.set_xlim(0, max(steps))
ax1.grid(True, alpha=0.3)
ax1.set_title('Episode Reward per Training Step (colored by curriculum tier)', fontsize=11)

# Bottom panel: tier timeline
tier_nums = {'easy': 0, 'medium': 1, 'hard': 2}
tier_y = [tier_nums[t] for t in tiers]
ax2.scatter(steps, tier_y, c=[TIER_COLORS[t] for t in tiers], s=15, zorder=3)
ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels(['Easy', 'Medium', 'Hard'])
ax2.set_xlabel('Training Step')
ax2.set_title('Curriculum Tier Timeline', fontsize=11)
ax2.set_xlim(0, max(steps))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training/training_dashboard.png', dpi=150, bbox_inches='tight')
print('Saved -> training/training_dashboard.png')
