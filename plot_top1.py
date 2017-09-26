#!/usr/bin/env python

import sys
import argparse
try:
    import seaborn as sns
    sns.set(style="dark")
except ImportError:
    pass
import matplotlib.pyplot as plt

from common_plot import parse_files, plot_accuracy

TOP_K = 1

parser = argparse.ArgumentParser(description='Plot top {}% from log files.'.format(TOP_K))
parser.add_argument('-v', dest='value_at_hover', action='store_true',
    help="Display plot values at cursor hover")
parser.add_argument('-s', dest='separate', action='store_true',
    help="plot each log separately, don't concatenate them")
parser.add_argument('log', nargs = '*', help = "list of log files.")
args = parser.parse_args()

data = parse_files(files=args.log, top_k=TOP_K, separate=args.separate)
plt = plot_accuracy(top_k=TOP_K, data=data, value_at_hover=args.value_at_hover)

plt.show()
