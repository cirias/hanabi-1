import gym_hanabi
import pickle

class RandomPolicy(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, _observation):
        return (self.action_space.sample(), )

if __name__ == "__main__":
    with open("pickled_policies/RandomPolicy.pickle", "wb") as f:
        action_space = gym_hanabi.envs.hanabi_env.MOVE_SPACE
        pickle.dump(RandomPolicy(action_space), f)
