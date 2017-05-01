class RandomPolicy(object):
    def __init__(self, env):
        self.env = env
        self.config = self.env.config
        self.reward = self.env.reward
        self.spaces = self.env.spaces

    def get_action(self, _observation):
        return (self.spaces.action_space().sample(), )
