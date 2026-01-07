import matplotlib.pyplot as plt
import numpy as np

stages = ['1. Baseline\n(Pointers)', '2. Devirtualized/cont. world array\n', '3. Final Optimized\n(reduced precision)']


theoretical_occupancy = [100.0, 62.5, 74.8] 
achieved_occupancy    = [81.8,  50.0, 60.8]

x = np.arange(len(stages)) 
width = 0.35                

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, theoretical_occupancy, width, label='Theoretical Occupancy', color="#b8d0e0", hatch='//')
rects2 = ax.bar(x + width/2, achieved_occupancy, width, label='Achieved Occupancy', color="#40a169")


ax.set_ylabel('Occupancy (%)')
ax.set_title('Optimization impact on Occupancy')
ax.set_xticks(x)
ax.set_xticklabels(stages)
ax.set_ylim(0, 110)
ax.legend()

ax.grid(axis='y', linestyle='--', alpha=0.7)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

plt.show()