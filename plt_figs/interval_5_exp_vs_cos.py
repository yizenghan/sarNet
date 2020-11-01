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

pp = PdfPages('_interval_exp_vs_cos.pdf')

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
                


flops = [0.07376, 0.09214, 0.11016, 0.13072]
flops = [a*1 for a in flops]
acc = [93.32, 93.73, 94.05, 94.49]
obj = plot(flops, acc, label="T1, add loss at 150", isError = True, marker='D', linestyle='-', linewidth=1, color='#00ccff')

flops = [0.07496, 0.09267, 0.11012]
flops = [a*1 for a in flops]
acc = [93.42, 93.97, 94.55]
obj = plot(flops, acc, label="T1, ep300, dr=2, 90to150", isError = True, marker='D', linestyle='-', linewidth=1, color='#A0522D')

flops = [0.0744, 0.0930, 0.1099]
flops = [a*1 for a in flops]
acc = [93.11, 93.94, 93.99]
obj = plot(flops, acc, label="T1, internal5, 0 to 200, exp", isError = True, marker='D', linestyle='-', linewidth=1, color='#008000')

flops = [0.0729, 0.0879, 0.1093]
flops = [a*1 for a in flops]
acc = [92.84, 93.00, 93.63]
obj = plot(flops, acc, label="T1, internal5, 0 to 200, cosine", isError = True, marker='D', linestyle='-', linewidth=1, color='#ff66ff')

flops = [0.0862, 0.0967, 0.1100]
flops = [a*1 for a in flops]
acc = [93.60, 94.31, 94.41]
obj = plot(flops, acc, label="T1, internal5, 100 to 300, cosine", isError = True, marker='D', linestyle='-', linewidth=1, color='#ff3300')

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
plt.ylim(5.4, 7.1)
plt.xlim(0.069, 0.145)
# plt.grid()
ax.yaxis.set_ticks(np.arange(5.4, 7.1, 0.1))
ax.xaxis.set_ticks(np.arange(0.069, 0.145, 0.01))
# ax.set_xscale("log", nonposx='clip')

plt.tight_layout(pad=2.5, w_pad=2, h_pad=1)

plt.xlabel('GFLOPs', size=15)
plt.ylabel('Error (\%)', size=15)
# plt.title('Error-FLOPs on ImageNet', size=16)


plt.legend(loc=3, scatterpoints=1, fontsize=9)
plt.savefig(pp, format='pdf', bbox_inches='tight')
pp.close()


