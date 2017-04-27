#! /usr/bin/env python

from gym_hanabi.policies import *
import argparse
import gym
import pickle

def main(args):
    env = gym.make(args.env_id)
    env = gym.wrappers.Monitor(env, args.directory, force=args.force)

    with open(args.pickled_player_policy, "rb") as f:
        player_policy = pickle.load(f)
    with open(args.pickled_ai_policy, "rb") as f:
        ai_policy = pickle.load(f)
        env.unwrapped.ai_policy = ai_policy

    for _ in range(args.num_games):
        observation = env.reset()
        done = False
        while not done:
            if args.render:
                env.render()
            action = player_policy.get_action(observation)[0]
            observation, reward, done, info = env.step(action)
        if args.render:
            env.render()

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--render",
        action="store_true", help="Render the game.")
    parser.add_argument("-f", "--force",
        action="store_true", help="Overwrite monitoring directory.")
    parser.add_argument("-n", "--num_games",
        type=int, default=1, help="Number of Hanabi games to play.")

    parser.add_argument("env_id",
        type=str, help="Environment id (e.g. HanabiAi-v0, MiniHanabiAi-v0.")
    parser.add_argument("pickled_player_policy",
        type=str, help="Pickled player policy file.")
    parser.add_argument("pickled_ai_policy",
        type=str, help="Pickled ai policy file.")
    parser.add_argument("directory",
        type=str, help="Monitoring directory.")

    return parser

if __name__ == "__main__":
    main(get_parser().parse_args())
