import csv

import argparse
import gym_hanabi

def write_csv(filename, header, rows):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([str(x) for x in row])

def get_self_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_policy",
        default=None, help="Filename of starting pickled policy")
    parser.add_argument("env_id",
        choices=gym_hanabi.SELF_ENV_IDS, help="Environment id")
    parser.add_argument("output_policy",
        help="Filename of final pickled policy")
    parser.add_argument("snapshot_dir", help="Snapshot directory")
    return parser

def get_ai_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_policy",
        default=None, help="Filename of starting pickled policy")
    parser.add_argument("env_id",
        choices=gym_hanabi.AI_ENV_IDS, help="Environment id")
    parser.add_argument("ai_policy", help="Filename of AI pickled policy")
    parser.add_argument("output_policy",
        help="Filename of final pickled policy")
    parser.add_argument("snapshot_dir", help="Snapshot directory")
    return parser
