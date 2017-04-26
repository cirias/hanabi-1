class RandomPolicy(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, _observation):
        return (self.action_space.sample(), )
