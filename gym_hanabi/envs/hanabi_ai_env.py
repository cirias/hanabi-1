from gym_hanabi.envs.hanabi_env import *

class HanabiAiEnv(HanabiEnv):
    def _step(self, action):
        # Note that for some reason, returning a non-empty info dict causes
        # rllab to crash. Until we figure out why that is, we use an empty info
        # dict.
        empty_info = dict()

        move = sample_to_move(action)
        try:
            reward, done = self.play_move(move)
            ai_move = None
            if not done:
                observation = game_state_to_sample(self.game_state)
                ai_move = self.ai_policy(observation)
                ai_reward, done = self.play_move(ai_move)
                reward += ai_reward
            observation = game_state_to_sample(self.game_state)
            return (observation, reward, done, empty_info)
        except ValueError:
            # The final reward is 0 if we break the rules.
            reward = -1 * self.game_state.current_reward()
            return (None, reward, True, empty_info)
