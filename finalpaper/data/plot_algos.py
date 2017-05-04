import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {"size": 16}
matplotlib.rc("font", **font)

def plot_algo(name, filename):
    # 0: length
    # 1: reward
    # 2: score
    data = np.genfromtxt(filename, skip_header=True, delimiter=",")
    scores = data[:,2].astype(np.int)
    mean = np.mean(scores)
    stddev = np.std(scores)

    plt.figure()
    plt.hist(data[:,2].astype(np.int), edgecolor="black", align="left", bins=range(11))
    plt.xticks(range(10))
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("{} self-learned mini Hanabi policy\nmean={}, stddev={:0.4}"
              .format(name, mean, stddev))
    plt.savefig("{}.pdf".format(name), bbox_inches="tight")
    plt.close()

def main():
    algos = [
        ("CEM", "sweep2/mini_cem_self.txt"),
        ("CMA-ES", "sweep2/mini_cma_es_self.txt"),
        ("TRPO", "sweep2/mini_trpo_self.txt"),
    ]

    for name, filename in algos:
        plot_algo(name, filename)

if __name__ == "__main__":
    main()
