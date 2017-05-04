import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {"size": 16}
matplotlib.rc("font", **font)

def hist(name, bins, learning, color, filename):
    # 0: length
    # 1: reward
    # 2: score
    data = np.genfromtxt(filename, skip_header=True, delimiter=",")
    scores = data[:,2].astype(np.int)
    mean = np.mean(scores)
    stddev = np.std(scores)

    plt.figure()
    plt.hist(data[:,2].astype(np.int), color=color, edgecolor="black",
             align="left", bins=range(bins))
    plt.xticks(range(0, bins - 1, 1 if bins <= 11 else 2))
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.title("mean={}, stddev={:0.4}".format(mean, stddev))
    plt.savefig("{}_{}.pdf".format(name.replace(" ", "_"),
                                   learning.replace(" ", "_")),
                                   bbox_inches="tight")
    plt.close()

def main():
    mini_bins = 11
    medium_bins = 18
    bins = 27

    policies = [
        ("mini Hanabi",   mini_bins,   "dynamic self-learned", "C0", "TBD"),
        ("medium Hanabi", medium_bins, "dynamic self-learned", "C0", "TBD"),
        ("Hanabi",        bins,        "dynamic self-learned", "C0", "TBD"),
        ("mini Hanabi",   mini_bins,   "guided-learned",       "C1", "TBD"),
        ("medium Hanabi", medium_bins, "guided-learned",       "C1", "TBD"),
        ("Hanabi",        bins,        "guided-learned",       "C1", "TBD"),
        ("mini Hanabi",   mini_bins,   "static self-learned",  "C2", "TBD"),
        ("medium Hanabi", medium_bins, "static self-learned",  "C2", "TBD"),
        ("Hanabi",        bins,        "static self-learned",  "C2", "TBD"),
        ("mini Hanabi",   mini_bins,   "heuristic",            "C3", "heuristic/mini_heuristic_self.txt"),
        ("medium Hanabi", medium_bins, "heuristic",            "C3", "heuristic/medium_heuristic_self.txt"),
        ("Hanabi",        bins,        "heuristic",            "C3", "heuristic/heuristic_self.txt"),
    ]

    for args in policies:
        hist(*args)

if __name__ == "__main__":
    main()
