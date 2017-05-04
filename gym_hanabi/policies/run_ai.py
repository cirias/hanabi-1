#! /usr/bin/env python

from gym_hanabi.policies import *
import argparse
import gym
import gym_hanabi
import pickle
import progressbar

def average(xs):
    return float(sum(xs)) / len(xs)

def nats():
    i = 1
    while True:
        yield i
        i += 1

def main(args):
    env = gym.make(args.env_id)

    with open(args.pickled_player_policy, "rb") as f:
        player_policy = pickle.load(f)
    with open(args.pickled_ai_policy, "rb") as f:
        ai_policy = pickle.load(f)
        env.unwrapped.ai_policy = ai_policy

    lengths = []
    rewards = []
    scores = []
    bar = progressbar.ProgressBar()
    for _ in bar(range(args.num_games)):
        observation = env.reset()
        i = 0
        for i in nats():
            if args.render:
                env.render()
            action = player_policy.get_action(observation)[0]
            observation, reward, done, info = env.step(action)
            if done:
                break
        if args.render:
            env.render()

        gs = env.env.game_state
        lengths.append(i * 2)
        rewards.append(env.env.reward.current_reward(gs))
        scores.append(gs.current_score())

    print("average length = {}".format(average(lengths)))
    print("average reward = {}".format(average(rewards)))
    print("average score  = {}".format(average(scores)))
    if args.output != "":
        header = ["length","reward","score"]
        assert len(lengths) == len(rewards) == len(scores)
        data = list(zip(lengths, rewards, scores))
        gym_hanabi.policies.common.write_csv(args.output, header, data)

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--render",
        action="store_true", help="Render the game.")
    parser.add_argument("-n", "--num_games",
        type=int, default=1, help="Number of Hanabi games to play.")
    parser.add_argument("-o", "--output",
        default="", help="File to write output.")

    parser.add_argument("env_id",
        choices=gym_hanabi.AI_ENV_IDS, help="Environment id")
    parser.add_argument("pickled_player_policy", help="Pickled player policy file.")
    parser.add_argument("pickled_ai_policy", help="Pickled ai policy file.")

    return parser

if __name__ == "__main__":
    main(get_parser().parse_args())
