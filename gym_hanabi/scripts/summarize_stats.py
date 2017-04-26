import argparse
import json

def average(xs):
    return float(sum(xs)) / len(xs)

def main(stats_file):
    """
    When you evaluate a policy on a gym environment (see run_ai.py and
    run_self.py), gym produces a directory with a bunch of statistics files
    summarizing the performance of the policy. The stats files are named
    something like openaigym.episode_batch.0.17502.stats.json and look
    something like this:

        {
          "episode_types": ["t", ...],
          "initial_reset_timestamp": 1493231366.3479626,
          "episode_lengths": [34, ...],
          "episode_rewards": [9.0, ...],
          "timestamps": [1493231366.369614, ...]
        }

    This script will summarize these statistics.
    """
    with open(stats_file, "r") as f:
        j = json.load(f)
        print("num_episodes = {}".format(len(j["episode_rewards"])))
        print("avg_length   = {}".format(average(j["episode_lengths"])))
        print("avg_reward   = {}".format(average(j["episode_rewards"])))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file",
        type=str, help="JSON stats file produced by OpenAI Gym.")
    return parser

if __name__ == "__main__":
    main(get_parser().parse_args().stats_file)
