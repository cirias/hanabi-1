from gym_hanabi.envs import hanabi_env
import pickle

class RandomPolicy(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, _observation):
        return (self.action_space.sample(), )

if __name__ == "__main__":
    configs_and_names = [
        (hanabi_env.HANABI_CONFIG, "RandomPolicy"),
        (hanabi_env.MEDIUM_HANABI_CONFIG, "MediumRandomPolicy"),
        (hanabi_env.MINI_HANABI_CONFIG, "MiniRandomPolicy"),
        (hanabi_env.MINI_HANABI_LOTSOFINFO_CONFIG, "MiniRandomLotsOfInfoPolicy"),
        (hanabi_env.MINI_HANABI_LOTSOFTURNS_CONFIG, "MiniRandomLotsOfTurnsPolicy"),
    ]

    for config, name in configs_and_names:
        with open("pickled_policies/{}.pickle".format(name), "wb") as f:
            action_space = hanabi_env.move_space(config)
            pickle.dump(RandomPolicy(action_space), f)
