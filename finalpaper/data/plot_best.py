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
    plt.title("TRPO {} {} policy\nmean={}, stddev={:0.4}"
              .format(learning, name, mean, stddev))
    plt.savefig("{}_{}.pdf".format(name.replace(" ", "_"),
                                   learning.replace(" ", "_")),
                                   bbox_inches="tight")
    plt.close()

def main():
    mini_bins = 11
    medium_bins = 18
    bins = 27

    policies = [
        ("mini Hanabi",   mini_bins,   "dynamic self-learned", "C0", "../../gym_hanabi/policies/logs/mini_trpo_self_best.txt"),
        ("medium Hanabi", medium_bins, "dynamic self-learned", "C0", "../../gym_hanabi/policies/logs/medium_trpo_self_best.txt"),
        ("Hanabi",        bins,        "dynamic self-learned", "C0", "../../gym_hanabi/policies/logs/trpo_self_best.txt"),
        ("mini Hanabi",   mini_bins,   "guided-learned",       "C1", "../../gym_hanabi/policies/logs/mini_trpo_ai_best.txt"),
        ("medium Hanabi", medium_bins, "guided-learned",       "C1", "../../gym_hanabi/policies/logs/medium_trpo_ai_best.txt"),
        ("Hanabi",        bins,        "guided-learned",       "C1", "../../gym_hanabi/policies/logs/trpo_ai_best.txt"),
        ("mini Hanabi",   mini_bins,   "static self-learned",  "C2", "../../gym_hanabi/policies/logs/mini_trpo_staggered_best.txt"),
        ("medium Hanabi", medium_bins, "static self-learned",  "C2", "../../gym_hanabi/policies/logs/medium_trpo_staggered_best.txt"),
        ("Hanabi",        bins,        "static self-learned",  "C2", "../../gym_hanabi/policies/logs/trpo_staggered_best.txt"),
    ]

    for args in policies:
        hist(*args)

if __name__ == "__main__":
    main()
