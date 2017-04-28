import argparse
import gym_hanabi

def get_self_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_policy",
        default=None, help="Filename of starting pickled policy")
    parser.add_argument("env_id",
        choices=gym_hanabi.SELF_ENV_IDS, help="Environment id")
    parser.add_argument("output_policy",
        help="Filename of final pickled policy")
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
    return parser
