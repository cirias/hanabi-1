from gym_hanabi.envs.hanabi_env import *

class HanabiAiEnv(HanabiEnv):
    def _step(self, action):
        move = sample_to_move(action)
        try:
            reward, done = self.play_move(move)
            ai_move = None
            if not done:
                observation = game_state_to_sample(self.game_state)
                ai_move = self.ai_policy(observation)
                ai_reward, done = self.play_move(ai_move)
                reward += ai_reward
            # Return (observation, reward, done, info).
            return (game_state_to_sample(self.game_state),
                    reward,
                    done,
                    {"state": self.game_state, "move": ai_move})
        except ValueError as e:
            # TODO: Log this instead of printing it.
            print(e)
            # Final reward of 0 if we break the rules.
            reward = -1 * self.game_state.current_reward()
            return (None, reward, True, self.game_state)
