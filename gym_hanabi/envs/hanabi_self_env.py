from gym_hanabi.envs import hanabi_env

class HanabiSelfEnv(hanabi_env.HanabiEnv):
    def __init__(self, config, reward, spaces):
        self.config = config
        self.reward = reward
        self.spaces = spaces
        self.action_space = spaces.action_space()
        self.observation_space = spaces.observation_space()
        self._seed()

    def _step(self, action_sample):
        try:
            move = self.spaces.sample_to_action(action_sample,
                    self.game_state.get_current_cards())
            reward, done = self.play_move(move)
            observation = self.game_state.to_observation()
            observation_sample = self.spaces.observation_to_sample(observation)
            info = {"game_state": self.game_state, "illegal": False}
            return (observation_sample, reward, done, info)
        except ValueError:
            reward = self.reward.illegal_move_reward(self.game_state)
            info = {"game_state": self.game_state, "illegal": True}
            return (None, reward, True, info)
