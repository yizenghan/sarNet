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

pp = PdfPages('_plot_temp7.pdf')

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
                
# plt.scatter(x=0.74, y=100-77.58, c='k', marker='o', label='Octave')
# plt.scatter(x=0.08081, y=100-93.46, c='k', marker='4', label='alphaBase32x2, T0.1, g2a2s2, lambda1.0')
# plt.scatter(x=0.07263, y=100-93.33, c='k', marker='x', label='alphaBase32x2, T0.1, g2a2s2, lambda3.0')
# plt.scatter(x=0.7368, y=100-76.93, c='r', marker='h', label='wRsm, g8a22s4, t0.7l0.1d0')
# plt.scatter(x=1.30, y=100-77.47, c='k', marker='+', label='Res50')
# ax.plot(0.7355, 100-77.36, label="wR, g8a22s4, t0.7l0.1d0", color='#FF0000', marker='4', markersize=5)
# ax.plot(0.7281, 100-78.00, label="wR, g8a22s4, t0.7l0.5d0", color='#FF0000', marker='x', markersize=5)
# ax.plot(0.7436, 100-77.75, label="wR, g8a22s4, t0.7l0.1d1", color='#FF0000', marker=',', markersize=5)
# ax.plot(0.7586, 100-78.13, label="wR, g8a22s2, t0.7l0.1d0", color='#FF0000', marker='+', markersize=5)
# ax.plot(0.7436, 100-78.14, label="wR, g8a22s2, t0.7l0.5d0", color='#FF0000', marker='2', markersize=5)
# ax.plot(0.7682, 100-77.97, label="wR, g8a22s2, t0.7l0.1d1", color='#FF0000', marker='D', markersize=5)
# ax.plot(0.7368, 100-76.93, label="wR_sm, g8a22s4, t0.7l0.1d0", color='#FF0000', marker='h', markersize=5)
# plt.text(0.06943, 100-93.036, 'Res32', family='serif', style='italic', ha='left', wrap=True)
# plt.text(0.12628, 100-93.604, 'Res56', family='serif', style='italic', ha='left', wrap=True)
# plt.text(0.25420, 100-93.961, 'Res110', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.13846, 100-94.27, 'no target(0.8699)', family='serif', style='italic', ha='left', wrap=True)

plt.text(0.08081, 100-93.46, 'target=0.25(0.3378), l=1', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.07203, 100-93.37, 'target=0.25(0.2502), l=3', family='serif', style='italic', ha='left', wrap=True)
# plt.text(0.07259, 100-92.36, 'target=0.25(0.2501), l=5', family='serif', style='italic', ha='left', wrap=True)
# plt.text(0.07267, 100-92.60, 'target=0.25(0.2501), l=10', family='serif', style='italic', ha='left', wrap=True)

#T10
plt.text(0.07369, 100-93.09, 'target=0.25(0.2597)', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.08983, 100-93.56, 'target=0.4(0.4132)', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.12227, 100-93.93, 'no target(0.7214)', family='serif', style='italic', ha='left', wrap=True)
#T5
plt.text(0.07422, 100-93.27, 'target=0.25(0.2653)', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.08779, 100-93.86, 'target=0.4(0.4106)', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.13182, 100-94.21, 'no target(0.8105)', family='serif', style='italic', ha='left', wrap=True)
#T1
plt.text(0.08335, 100-93.30, 'target=0.25(0.3473)', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.09404, 100-93.82, 'target=0.4(0.4530)', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.11055, 100-94.34, 'target=0.6(0.6070)', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.14246, 100-94.48, 'no target(0.9092)', family='serif', style='italic', ha='left', wrap=True)
#T0.1
plt.text(0.08882, 100-93.65, 'target=0.25(0.4011), l=0.5', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.09618, 100-93.94, 'target=0.4(0.4792)', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.11004, 100-93.97, 'target=0.6(0.6019)', family='serif', style='italic', ha='left', wrap=True)
plt.text(0.14084, 100-94.19, 'no target(0.8963)', family='serif', style='italic', ha='left', wrap=True)
# plt.annotate('t = 0.3', xy=(3.2149, 23.09), xytext=(2.85, 23.3), arrowprops=dict(facecolor='black',width=0.2,headwidth=3, shrink=0.02))
# plt.annotate('t = 0.3', xy=(2.8525, 23.18), xytext=(2.85, 23.3), arrowprops=dict(facecolor='black',width=0.2,headwidth=3, shrink=0.02))
# plt.annotate('t = 0.3', xy=(2.6671, 23.22), xytext=(2.85, 23.3), arrowprops=dict(facecolor='black',width=0.2,headwidth=3, shrink=0.02))

# flops = [1.01, 2.23, 3.93]
# flops = [a*1 for a in flops]
# acc = [71.89, 74.46, 75.96]
# obj = plot(flops, acc, label="Baseline", isError = True, marker='H', linestyle='-', linewidth=1, color='#000000')


flops_g1 = [0.08882, 0.09678, 0.11004, 0.14084]
flops_g1 = [a*1 for a in flops_g1]
acc_g1 = [93.65, 93.91, 93.97, 94.19]
obj = plot(flops_g1, acc_g1, label="T0.1", isError = True, marker='*', linestyle='-', linewidth=1, color='#00ccff')

flops = [0.08271, 0.09404, 0.11055, 0.14246]
flops = [a*1 for a in flops]
acc = [93.30, 93.82, 94.34, 94.48]
obj = plot(flops, acc, label="T1", isError = True, marker='D', linestyle='-', linewidth=1, color='#008000')

flops = [0.0756, 0.0913, 0.1106, 0.1325]
flops = [a*1 for a in flops]
acc = [93.30, 93.80, 94.05, 93.95]
obj = plot(flops, acc, label="T3", isError = True, marker='D', linestyle='-', linewidth=1, color='#cc00cc')

flops = [0.07422, 0.08819, 0.13182]
flops = [a*1 for a in flops]
acc = [93.27, 93.82, 94.24]
obj = plot(flops, acc, label="T5", isError = True, marker='H', linestyle='-', linewidth=1, color='#A0522D')

flops = [0.07369, 0.08983, 0.12227]
flops = [a*1 for a in flops]
acc = [93.09, 93.56, 93.93]
obj = plot(flops, acc, label="T10", isError = True, marker='h', linestyle='-', linewidth=1, color='#FF69B4')

flops_g1 = [0.08882, 0.08081, 0.07263, 0.07259, 0.07267]
flops_g1 = [a*1 for a in flops_g1]
acc_g1 = [93.65, 93.46, 93.33, 92.36, 92.60]
obj = plot(flops_g1, acc_g1, label="T0.1, different lambda", isError = True, marker='4', linestyle='-', linewidth=1, color='#0000ff')

flops = [0.13846, 0.13182]
flops = [a*1 for a in flops]
acc = [94.24, 94.24]
obj = plot(flops, acc, label="T5, target=1 for 1st half", isError = True, marker='2', linestyle='-', linewidth=1, color='#FFA500')


# flops = [0.06943, 0.12628, 0.25420]
# flops = [a*1 for a in flops]
# acc = [93.036, 93.604, 93.961]
# obj = plot(flops, acc, label="Res32, 56, 110", isError = True, marker='4', linestyle='-', linewidth=1, color='#FFFF00')


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
plt.ylim(5.5, 7.05)
plt.xlim(0.069, 0.145)
# plt.grid()
ax.yaxis.set_ticks(np.arange(5.5, 7.05, 0.1))
ax.xaxis.set_ticks(np.arange(0.069, 0.145, 0.01))
# ax.set_xscale("log", nonposx='clip')

plt.tight_layout(pad=2.5, w_pad=2, h_pad=1)

plt.xlabel('GFLOPs', size=15)
plt.ylabel('Error (\%)', size=15)
# plt.title('Error-FLOPs on ImageNet', size=16)


plt.legend(loc=3, scatterpoints=1, fontsize=9)
plt.savefig(pp, format='pdf', bbox_inches='tight')
pp.close()


