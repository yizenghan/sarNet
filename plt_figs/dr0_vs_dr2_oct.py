import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
from matplotlib import rcParams
rc('text', usetex=True)
rcParams['backend'] = 'ps'
rcParams['text.latex.preamble'] = ["\\usepackage{gensymb}"]
rcParams['font.size'] = 5
rcParams['legend.fontsize'] = 12
rc('font', **{'family':'serif', 'serif':['Computer Modern'], 'monospace': ['Computer Modern Typewriter']})

pp = PdfPages('dr0_vs_dr2_50g2a2s2.pdf')

max_y = 15
max_x = 4.5e8
fig, ax = plt.subplots(figsize=(6, 4.585))

ls = [':']

plt.scatter(x=0.74, y=100-94.72, c='k', marker='o', label='Oct-ResNet50')

def plot(x, y, label=None, accumulated=False, linestyle='-', linewidth=2, isError = False,
         isPercentage=False, alpha=1., color=None, marker=None, edgecolor=None, step=False, zorder=None):
    if accumulated:
        for i in range(1, len(x)):
            x[i] += x[i - 1]
    if isPercentage:
        y = [100 * e for e in y]
    if isError:
        y = [100 - e for e in y]
    if step:
        x = [0] + x
        y = [0] + y
        x = np.array(x) 
        return ax.step(x, y, where='post', label=label, linestyle=linestyle, linewidth=linewidth, alpha=alpha, color=color,
                marker=marker, markersize=6, zorder=zorder)
    else:
        x = np.array(x) 
        return ax.plot(x, y, label=label, linestyle=linestyle, linewidth=linewidth, alpha=alpha, color=color,
                marker=marker, markersize=6, zorder=zorder)
                


flops = [0.59,0.64,0.74, 0.84]
flops = [a*1 for a in flops]
acc = [94.31, 94.48, 94.67, 95.30]
obj = plot(flops, acc, label="T1, 300 eps, add loss at 150", isError = True, marker='D', linestyle='-', linewidth=1, color='#00ccff')

flops = [0.605, 0.674, 0.7413, 0.84]
flops = [a*1 for a in flops]
acc = [94.36,95.08, 95.20, 95.30]
obj = plot(flops, acc, label="T1, 300 eps, dr=2, 90to150", isError = True, marker='D', linestyle='-', linewidth=1, color='#A0522D')


ax.spines['left'].set_color('k')
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_color('k')
ax.spines['right'].set_linewidth(0.5)
ax.spines['top'].set_color('k')
ax.spines['top'].set_linewidth(0.5)
ax.spines['bottom'].set_color('k')
ax.spines['bottom'].set_linewidth(0.5)
ax.tick_params(axis='x', length=4, labelsize=12)
ax.tick_params(axis='y', length=4, labelsize=12)


plt.grid(color='#000000', alpha=0.1, linestyle='-', linewidth=0.5)
plt.ylim(4.5, 6)
plt.xlim(0.55, 0.85)
# plt.grid()
ax.yaxis.set_ticks(np.arange(4.5, 6, 0.2))
ax.xaxis.set_ticks(np.arange(0.55, 0.85, 0.05))
# ax.set_xscale("log", nonposx='clip')

plt.tight_layout(pad=2.5, w_pad=2, h_pad=1)

plt.xlabel('GFLOPs', size=15)
plt.ylabel('Error (\%)', size=15)
# plt.title('Error-FLOPs on ImageNet', size=16)


plt.legend(loc=3, scatterpoints=1, fontsize=9)
plt.savefig(pp, format='pdf', bbox_inches='tight')
pp.close()


