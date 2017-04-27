import gym_hanabi
import pickle

class RandomPolicy(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, _observation):
        return (self.action_space.sample(), )

if __name__ == "__main__":
    hanabi_env = gym_hanabi.envs.hanabi_env
    with open("pickled_policies/RandomPolicy.pickle", "wb") as f:
        action_space = hanabi_env.move_space(hanabi_env.HANABI_CONFIG)
        pickle.dump(RandomPolicy(action_space), f)
    with open("pickled_policies/MiniRandomPolicy.pickle", "wb") as f:
        action_space = hanabi_env.move_space(hanabi_env.MINI_HANABI_CONFIG)
        pickle.dump(RandomPolicy(action_space), f)
