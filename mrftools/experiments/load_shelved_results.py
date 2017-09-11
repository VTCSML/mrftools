"""
This script loads shelved files from experiment trials to regenerate the plots. 
"""
import sys
sys.path.insert(0, '../src')
sys.path.insert(0, '../util')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
from random import shuffle
from scipy import signal as sg
import copy
import itertools
import networkx as nx
import os
import argparse
import shelve
from matplotlib.font_manager import FontProperties


# l1 = 0.075
l1 = 0.05

shelve_file = 'shelves/synthetic_results_1800_%s_1.0' % str(l1)
output_dir = 'new_figures_%s' % str(l1)

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

data = shelve.open(shelve_file)
METHODS = data['methods']
M_time_stamps = data['time_stamps']
objs = data['objs']
f1_scores = data['f1']
recalls = data['recall']

alphas = {1:([1, 0.5, 0.0],'o'), .75:([0.35, 1, 0.85],'v'), .5:([0.2, 0.6, .7],'<'), .25:([.9, .3, .7],'>'), 0:([.5, .7, .2],'s') }

METHOD_COLORS = dict()
METHOD_legend = dict()
METHOD_marker = dict()

for alpha in alphas.keys():
	meth = '$ '+ str(alpha) + '$ '
	print('>>>>>>>>>>>>>>>>>>>>>METHOD: ' + meth)
	METHOD_legend[meth] = meth
	METHOD_COLORS[meth] = alphas[alpha][0]
	METHOD_marker[meth] = alphas[alpha][1]


print('>>>>>>>>>>>>>>>>>>>>>METHOD: First Hit')
meth = 'First Hit'
METHOD_COLORS[meth] = [0, 0, 0]
METHOD_legend[meth] = meth
METHOD_marker[meth] = '8'


print('>>>>>>>>>>>>>>>>>>>>>METHOD: Graft' )
meth = 'EG'
METHOD_COLORS[meth] = [0.75, 0.75, 0.75]
METHOD_legend[meth] = meth
METHOD_marker[meth] = 'h'


###############MAKE PLOTS

print('>Making plots')

plt.close()
fig, ax1 = plt.subplots()
for i in range(len(METHODS)):
	if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
		ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2, label=METHOD_legend[METHODS[i]])
	else:
		ax1.plot(M_time_stamps[METHODS[i]], objs[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])

ax1.set_xlabel('Time')
ax1.set_ylabel('LOSS')
ax1.legend(loc='best', framealpha=0.5, fancybox=True)
plt.savefig(output_dir + '/OBJ.eps', format='eps', dpi=1000)
plt.close()


# #UNCOMMENT TO PLOT test nll SCORES EVOLUTION
# plt.close()
# fig, ax1 = plt.subplots()
# for i in range(len(METHODS)):
# 	if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
# 		ax1.plot(subsampled_time_stamps[METHODS[i]], test_nlls[METHODS[i]], color= METHOD_COLORS[METHODS[i]], linewidth=2, label=METHOD_legend[METHODS[i]])
# 	else:
# 		ax1.plot(subsampled_time_stamps[METHODS[i]], test_nlls[METHODS[i]], color= METHOD_COLORS[METHODS[i]], linewidth=2, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Test NLL')
# ax1.legend(loc='best', framealpha=0.5, fancybox=True)
# plt.savefig(output_dir + '/NLL.eps', format='eps', dpi=1000)
# plt.close()

#UNCOMMENT TO PLOT F1 SCORES EVOLUTION
plt.close()
fig, ax1 = plt.subplots()
for i in range(len(METHODS)):
	f1 = f1_scores[METHODS[i]]
	if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
		ax1.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0,  label=METHOD_legend[METHODS[i]])
	else:
		ax1.plot(M_time_stamps[METHODS[i]], f1_scores[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])

ax1.set_xlabel('Time')
ax1.set_ylabel('F1 Score')
fontP = FontProperties()
fontP.set_size('small')
lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('F1 VS Time')
if f1[-1] > .4:
	plt.savefig(output_dir + '/F1*.eps',linewidth=2.5, format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
else:
	plt.savefig(output_dir + '/F1.eps',linewidth=2.5, format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

#UNCOMMENT TO PLOT Recall SCORES EVOLUTION
plt.close()
fig, ax1 = plt.subplots()
for i in range(len(METHODS)):
	if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
		ax1.plot(M_time_stamps[METHODS[i]], recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0, label=METHOD_legend[METHODS[i]])
	else:
		ax1.plot(M_time_stamps[METHODS[i]], recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':',marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])
ax1.set_xlabel('Time')
ax1.set_ylabel('Recall Score')
lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(output_dir + '/Recall.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()

fig, ax1 = plt.subplots()
for i in range(len(METHODS)):
	if METHODS[i] == 'EG' or METHODS[i] == 'First Hit':
		ax1.plot(recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]],fillstyle='full', markeredgewidth=0.0, label=METHOD_legend[METHODS[i]])
	else:
		ax1.plot(recalls[METHODS[i]], color=METHOD_COLORS[METHODS[i]], linewidth=2.5, linestyle=':', marker=METHOD_marker[METHODS[i]], fillstyle='full', markeredgewidth=0.0, label=r'$\alpha = $'+ METHOD_legend[METHODS[i]])
ax1.set_xlabel('Activation Iteration')
ax1.set_ylabel('Recall Score')
lgd = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(output_dir + '/IterRecall.eps', format='eps', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close()
