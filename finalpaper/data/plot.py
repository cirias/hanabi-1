import csv
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {"size": 16}
matplotlib.rc('font', **font)

def main(tabular_filename):
    # For whatever reason, the header order is random.
    with open(tabular_filename, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        x_index = header.index("Iteration")
        y_index = header.index("AverageReturn")

    dat = np.genfromtxt(tabular_filename, skip_header=True, delimiter=',')
    xs = dat[:,x_index]
    ys = dat[:,y_index]

    plt.plot(xs, ys, linewidth=3)
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs.\nNumber of Training Iterations")
    plt.grid()
    plt.savefig('reward_vs_iteration.pdf', bbox_inches='tight')

def usage(script_name):
    print("python {} <tabular_csv>")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage(sys.argv[0])
        sys.exit(1)
    main(sys.argv[1])
