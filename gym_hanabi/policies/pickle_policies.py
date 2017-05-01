#! /usr/bin/env python

import pickle

import gym
from gym_hanabi.policies.heuristic_policy import HeuristicPolicy
from gym_hanabi.policies.heuristic_simple_policy import HeuristicSimplePolicy
from gym_hanabi.policies.keyboard_policy import KeyboardPolicy
from gym_hanabi.policies.random_policy import RandomPolicy

def main():
    envs_and_name_templates = [
        (gym.make("HanabiSelf-v0").env,
         "{}Policy"),
        (gym.make("MediumHanabiSelf-v0").env,
         "Medium{}Policy"),
        (gym.make("MiniHanabiSelf-v0").env,
         "Mini{}Policy"),
        (gym.make("MiniHanabiLotsOfInfoSelf-v0").env,
         "Mini{}LotsOfInfoPolicy"),
        (gym.make("MiniHanabiLotsOfTurnsSelf-v0").env,
         "Mini{}LotsOfTurnsPolicy"),
        (gym.make("MiniHanabiLinearRewardSelf-v0").env,
         "Mini{}LinearRewardPolicy"),
        (gym.make("MiniHanabiSquaredRewardSelf-v0").env,
         "Mini{}SquaredRewardPolicy"),
        (gym.make("MiniHanabiSkewedRewardSelf-v0").env,
         "Mini{}SkewedRewardPolicy"),
    ]

    for env, name_template in envs_and_name_templates:
        name = name_template.format("Keyboard")
        with open("pickled_policies/{}.pickle".format(name), "wb") as f:
            pickle.dump(KeyboardPolicy(env), f)

        name = name_template.format("Heuristic")
        with open("pickled_policies/{}.pickle".format(name), "wb") as f:
            pickle.dump(HeuristicPolicy(env), f)

        name = name_template.format("HeuristicSimple")
        with open("pickled_policies/{}.pickle".format(name), "wb") as f:
            pickle.dump(HeuristicSimplePolicy(env), f)

        name = name_template.format("Random")
        with open("pickled_policies/{}.pickle".format(name), "wb") as f:
            pickle.dump(RandomPolicy(env), f)

if __name__ == "__main__":
    main()
