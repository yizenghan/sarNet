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

pp = PdfPages('_plot_ablation_res50.pdf')

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
                
plt.scatter(x=4.1007*1, y=100-76.78, c='k', marker='o', label='ResNet50')

flops_g1 = [3.2149, 3.6750, 4.1249]
flops_g1 = [a*1 for a in flops_g1]
acc_g1 = [76.91, 77.14, 77.43]
obj = plot(flops_g1, acc_g1, label="SAR-ResNet50, G=1", isError = True, marker='*', linestyle='--', linewidth=1, color='#00ccff')
plt.text(2.53, 23.25, 't=0.3', family='serif', style='italic', ha='left', wrap=True)
plt.text(2.8, 23.13, 't=0.3', family='serif', style='italic', ha='left', wrap=True)
plt.text(3.16, 23.115, 't=0.3', family='serif', style='italic', ha='left', wrap=True)
plt.text(2.58, 23.05, 't=0.5', family='serif', style='italic', ha='left', wrap=True)
plt.text(2.92, 22.95, 't=0.5', family='serif', style='italic', ha='left', wrap=True)
plt.text(3.5, 22.82, 't=0.5', family='serif', style='italic', ha='left', wrap=True)
plt.text(3.13, 22.52, 't=0.7', family='serif', style='italic', ha='left', wrap=True)
plt.text(2.7, 22.7, 't=0.7', family='serif', style='italic', ha='left', wrap=True)
plt.text(3.95, 22.52, 't=0.7', family='serif', style='italic', ha='left', wrap=True)
# plt.annotate('t = 0.3', xy=(3.2149, 23.09), xytext=(2.85, 23.3), arrowprops=dict(facecolor='black',width=0.2,headwidth=3, shrink=0.02))
# plt.annotate('t = 0.3', xy=(2.8525, 23.18), xytext=(2.85, 23.3), arrowprops=dict(facecolor='black',width=0.2,headwidth=3, shrink=0.02))
# plt.annotate('t = 0.3', xy=(2.6671, 23.22), xytext=(2.85, 23.3), arrowprops=dict(facecolor='black',width=0.2,headwidth=3, shrink=0.02))
flops = [2.8525, 3.0787, 3.3119]
flops = [a*1 for a in flops]
acc = [76.82, 76.99, 77.42]
obj = plot(flops, acc, label="SAR-ResNet50, G=2", isError = True, marker='P', linestyle='--', linewidth=1, color='#3399ff')

flops = [2.6671, 2.7819, 2.8983]
flops = [a*1 for a in flops]
acc = [76.78, 76.93, 77.25]
obj = plot(flops, acc, label="SAR-ResNet50, G=4", isError = True, marker='d', linestyle='--', linewidth=1, color='#0066ff')


flops = [2.97,3.18,3.38]
flops = [a*1 for a in flops]
acc = [77.15,77.06,77.39]
obj = plot(flops, acc, label="SAR-ResNet50-new, G=2", isError = True, marker='P', linestyle='-', linewidth=1, color='#339966')
flops = [3.18,3.38]
flops = [a*1 for a in flops]
acc = [77.47,77.55]
obj = plot(flops, acc, label="SAR-ResNet50-new, G=2, warmup", isError = True, marker='P', linestyle='-', linewidth=1, color='#26734d')
flops = [2.82,2.92,3.02]
flops = [a*1 for a in flops]
acc = [77.05, 77.16, 77.34]
obj = plot(flops, acc, label="SAR-ResNet50-new, G=4", isError = True, marker='d', linestyle='-', linewidth=1, color='#339933')

ax.arrow(4.1007, 23.22, 2.6671-4.1007+0.06, 0, fc='r', ec='r', linestyle=':', head_width=0.03, head_length=0.06)
plt.text(3.35, 23.23, '-35\%', family='serif', style='italic', ha='left', wrap=True,c='r')

ax.arrow(4.1007, 23.22, 0, 22.57-23.22+0.08, fc='r', ec='r', linestyle=':', head_width=0.03, head_length=0.06)
plt.text(4.13, 22.9, '-0.65', family='serif', style='italic', ha='left', wrap=True,c='r')

# ax.annotate("hhh", xy=(2.6671, 23.33), xytext=(4.1007, 23.22), arrowprops=dict(arrowstyle="->"))
# flops = [3.4023, 3.8230, 4.2700]
# flops = [a*1 for a in flops]
# acc = [77.63, 77.92, 78.15]
# obj = plot(flops, acc, label="ResNeXt50 32x4d, refine g=1", isError = True, marker='*', linestyle='-', linewidth=1, color='#00ccff')

# flops = [3.0587,3.2729, 3.5179]
# flops = [a*1 for a in flops]
# acc = [77.77, 77.98, 78.07]
# obj = plot(flops, acc, label="ResNeXt50 32x4d, refine g=2", isError = True, marker='P', linestyle='-', linewidth=1, color='#3399ff')

# flops = [2.8644, 2.9972,3.1374]
# flops = [a*1 for a in flops]
# acc = [77.38, 77.66, 77.91]
# obj = plot(flops, acc, label="ResNeXt50 32x4d, refine g=4", isError = True, marker='.', linestyle='-', linewidth=1, color='#0066ff')

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
plt.ylim(22.25,23.5)
plt.xlim(2.5, 4.5)
# plt.grid()
ax.yaxis.set_ticks(np.arange(22.25,23.5, 0.25))
ax.xaxis.set_ticks(np.arange(2.5, 4.5, 0.5))
# ax.set_xscale("log", nonposx='clip')

plt.tight_layout(pad=2.5, w_pad=2, h_pad=1)

plt.xlabel('GFLOPs', size=15)
plt.ylabel('Error (\%)', size=15)
# plt.title('Error-FLOPs on ImageNet', size=16)


plt.legend(loc=1, scatterpoints=1, fontsize=9)
plt.savefig(pp, format='pdf', bbox_inches='tight')
pp.close()


