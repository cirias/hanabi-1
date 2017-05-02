
import csv
import sys
import numpy as np

with open(sys.argv[1], 'r') as csvfile:
    valreader = csv.reader(csvfile)
    next(valreader, None) # skip header
    valdat = list(map(lambda x: list(map(float, x)), valreader))


dat = np.array(valdat)

plotxcol = 0
plotycol = 3

xvals = dat[:,plotxcol]
yvals = dat[:,plotycol]

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.plot(xvals, yvals)
plt.xlabel("Iteration")
plt.ylabel("Average Score")
plt.title("Average Hanabi Score vs. Number of Training Iterations")
plt.savefig('plot.pdf', bbox_inches='tight')
