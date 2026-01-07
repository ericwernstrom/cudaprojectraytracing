import matplotlib.pyplot as plt
import numpy as np


labels = ['Before Optimizations', 'After Mem Optimization', 'After precision reduc']


render_times = [2.86, 0.334, 0.340] 
overhead_times = [0.045, 0.01, 0.005]

width = 0.5

fig, ax = plt.subplots(figsize=(8, 6))


p1 = ax.bar(labels, render_times, width, label='Render kernel', color='#d62728')

p2 = ax.bar(labels, overhead_times, width, bottom=render_times, label='Other kernels/Overhead', color='#1f77b4')

ax.set_ylabel('Time (Seconds)')
ax.set_title('Ray Tracing Performance: pointers vs cont. array')
ax.legend()

for i, rect in enumerate(p1):
    height = rect

plt.show()