import gym
import gym.utils
import gym.utils.seeding
from gym_hanabi.envs import hanabi
import six

class HanabiEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def play_move(self, move):
        # The reward of this move is the current reward after the move minus
        # the current reward before the move.
        reward = -self.reward.current_reward(self.game_state)
        self.game_state.play_move(move)
        reward += self.reward.current_reward(self.game_state)

        # The game is over when there are no turns left.
        done = self.game_state.num_turns_left == 0
        return reward, done

    def _step(self, action):
        raise NotImplementedError()

    def _reset(self):
        self.game_state = hanabi.GameState(self.config, self.np_random)
        observation = self.game_state.to_observation()
        return self.spaces.observation_to_sample(observation)

    def _render(self, mode='human', close=False):
        if close:
            return

        if mode == "human":
            print(self.game_state.render())
        elif mode == "ansi":
            s = six.StringIO()
            s.write(self.game_state.render() + "\n")
            return s
        else:
            super(HanabiEnv, self).render(mode=mode)

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

if __name__ == "__main__":
    import doctest
    doctest.testmod()
