import csv
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {"size": 16}
matplotlib.rc("font", **font)

def read_tabular_csv(tabular_filename):
    # For whatever reason, the header order is random.
    with open(tabular_filename, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        x_index = header.index("Iteration")
        y_index = header.index("AverageReturn")

    dat = np.genfromtxt(tabular_filename, skip_header=True, delimiter=",")
    return dat[:,x_index], dat[:,y_index]

def plot_sweep(name, vals, filenames):
    plt.figure()
    for val, filename in zip(vals, filenames):
        xs, ys = read_tabular_csv(filename)
        plt.plot(xs, ys, linewidth=3, label="{}={}".format(name, val))
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title("Average Reward vs.\nNumber of Training Iterations")
    plt.grid()
    plt.legend()
    plt.savefig("reward_vs_iteration_{}.pdf".format(name), bbox_inches="tight")
    plt.close()

def main():
    prefix = "sweep/mini_trpo_self"

    tuple_to_str = lambda t: "({})".format(",".join(str(x) for x in t))
    hidden_sizes_name = "hidden_sizes"
    hidden_sizes_vals = [(8,8), (16,16), (32,32), (64,64), (64,64,64)]
    hidden_sizes_files = [
        "{}_hidden_size={}/tabular.csv".format(prefix, tuple_to_str(t))
        for t in hidden_sizes_vals]
    plot_sweep(hidden_sizes_name, hidden_sizes_vals, hidden_sizes_files)

    batch_size_name = "batch_size"
    batch_size_vals = [1000, 4000, 10000]
    batch_size_files = [
        "{}_batch_size={}/tabular.csv".format(prefix, bs)
        for bs in batch_size_vals]
    plot_sweep(batch_size_name, batch_size_vals, batch_size_files)

    discount_name = "discount"
    discount_vals = [1, 0.999, 0.99]
    discount_files = [
        "{}_discount={}/tabular.csv".format(prefix, d)
        for d in discount_vals]
    plot_sweep(discount_name, discount_vals, discount_files)

    step_size_name = "step_size"
    step_size_vals = [1, 0.1, 0.01, 0.001]
    step_size_files = [
        "{}_step_size={}/tabular.csv".format(prefix, ss)
        for ss in step_size_vals]
    plot_sweep(step_size_name, step_size_vals, step_size_files)

    reward_name = "reward"
    reward_vals = ["constant", "linear", "squared", "skewed"]
    reward_files = [
        "sweep/mini_trpo__self/tabular.csv",
        "sweep/mini_trpo_LinearReward_self/tabular.csv",
        "sweep/mini_trpo_SquaredReward_self/tabular.csv",
        "sweep/mini_trpo_SkewedReward_self/tabular.csv"
    ]
    plot_sweep(reward_name, reward_vals, reward_files)

if __name__ == "__main__":
    main()
