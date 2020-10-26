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
rcParams['font.size'] = 12
rcParams['legend.fontsize'] = 12
rc('font', **{'family':'serif', 'serif':['Computer Modern'], 'monospace': ['Computer Modern Typewriter']})

pp = PdfPages('_plot_res50.pdf')

max_y = 15
max_x = 4.5e8
fig, ax = plt.subplots(figsize=(6, 4.585))

ls = [':']


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
                
plt.scatter(x=4.1007, y=100-77.26, c='k', marker='o', label='ResNet50')

plt.scatter(x=2.5, y=22.20, c='r', marker='o', label='oct-ResNet50 (preAct)')

flops_g1 = [2.48,2.85]
flops_g1 = [a*1 for a in flops_g1]
acc_g1 = [76.85, 77.31]
obj = plot(flops_g1, acc_g1, label="BL-ResNet50", isError = True, marker='o', linestyle='--', linewidth=1, color='#cc00cc')


flops = [2.90, 3.01, 3.26]
flops = [a*1 for a in flops]
acc = [77.34, 77.41, 78.11]
obj = plot(flops, acc, label="SAR-ResNet (BL) g1a2s2", isError = True, marker='*', linestyle='-', linewidth=1, color='#99d6ff')

flops = [2.82, 2.89]
flops = [a*1 for a in flops]
acc = [77.50, 77.64]
obj = plot(flops, acc, label="SAR-ResNet (BL) g2a2s2", isError = True, marker='*', linestyle='-', linewidth=1, color='#33adff')

flops = [2.55, 2.81]
flops = [a*1 for a in flops]
acc = [77.15, 77.62]
obj = plot(flops, acc, label="SAR-ResNet (BL) g4a2, s=2,4", isError = True, marker='*', linestyle='-', linewidth=1, color='#006bb3')


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
plt.ylim(21.75,23.25)
plt.xlim(2.25, 4.25)
# plt.grid()
ax.yaxis.set_ticks(np.arange(21.75,23.25, 0.25))
ax.xaxis.set_ticks(np.arange(2.25, 4.25, 0.25))
# ax.set_xscale("log", nonposx='clip')

plt.tight_layout(pad=2.5, w_pad=2, h_pad=1)

plt.xlabel('GFLOPs', size=15)
plt.ylabel('Error (\%)', size=15)
# plt.title('Error-FLOPs on ImageNet', size=16)


plt.legend(loc=1, scatterpoints=1, fontsize=9)
plt.savefig(pp, format='pdf', bbox_inches='tight')
pp.close()


