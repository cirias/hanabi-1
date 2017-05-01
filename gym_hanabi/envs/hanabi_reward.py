class Reward(object):
    def current_reward(self, game_state):
        raise NotImplementedError()

    def illegal_move_reward(self, game_state):
        raise NotImplementedError()

class ConstantReward(Reward):
    def current_reward(self, game_state):
        return game_state.current_score()

    def illegal_move_reward(self, game_state):
        return -self.current_reward(game_state)

class LinearReward(Reward):
    def current_reward(self, game_state):
        f = lambda v: sum(range(1, v + 1))
        return sum(f(v) for v in game_state.played_cards.values())

    def illegal_move_reward(self, game_state):
        return -self.current_reward(game_state)

class SquaredReward(Reward):
    def current_reward(self, game_state):
        f = lambda v: sum(x**2 for x in range(1, v + 1))
        return sum(f(v) for v in game_state.played_cards.values())

    def illegal_move_reward(self, game_state):
        return -self.current_reward(game_state)

class SkewedReward(Reward):
    def current_reward(self, game_state):
        f = lambda v: sum(10**(x-1) for x in range(1, v + 1))
        return sum(f(v) for v in game_state.played_cards.values())

    def illegal_move_reward(self, game_state):
        return -self.current_reward(game_state)
