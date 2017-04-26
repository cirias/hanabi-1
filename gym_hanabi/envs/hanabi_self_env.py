from gym_hanabi.envs.hanabi_env import *

class HanabiSelfEnv(HanabiEnv):
    def _step(self, action):
        move = sample_to_move(action)
        try:
            reward, done = self.play_move(move)
            return (game_state_to_sample(self.game_state),
                    reward,
                    done,
                    {})
                    # {"state": self.game_state, "move": move})
        except ValueError as e:
            # TODO: Log this instead of printing it.
            # print(e)
            # Final reward of 0 if we break the rules.
            reward = -1 * self.game_state.current_reward()
            # return (None, reward, True, {"state": self.game_state})
            return (None, reward, True, {})
