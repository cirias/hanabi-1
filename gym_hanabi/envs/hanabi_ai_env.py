from gym_hanabi.envs.hanabi_env import *

class HanabiAiEnv(HanabiEnv):
    def __init__(self, config, reward, spaces, ai_policy=None):
        self.config = config
        self.reward = reward
        self.spaces = spaces
        self.ai_policy = ai_policy
        self.action_space = spaces.action_space()
        self.observation_space = spaces.observation_space()
        self._seed()

    def _step(self, action):
        spaces = self.spaces
        try:
            move = spaces.sample_to_action(action,
                    self.game_state.get_current_cards())
            reward, done = self.play_move(move)
            if not done:
                observation = self.game_state.to_observation()
                observation_sample = spaces.observation_to_sample(observation)
                ai_action_sample = self.ai_policy.get_action(observation_sample)[0]
                ai_action = spaces.sample_to_action(ai_action_sample,
                        self.game_state.get_current_cards())
                ai_reward, done = self.play_move(ai_action)
                reward += ai_reward
            observation = self.game_state.to_observation()
            observation_sample = spaces.observation_to_sample(observation)
            info = {"game_state": self.game_state, "illegal": False}
            return (observation_sample, reward, done, info)
        except ValueError as e:
            reward = self.reward.illegal_move_reward(self.game_state)
            info = {"game_state": self.game_state, "illegal": True}
            return (None, reward, True, info)
