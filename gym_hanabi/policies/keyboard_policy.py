import gym
from gym_hanabi.envs import hanabi
from gym_hanabi.envs import hanabi_env
from six.moves import input

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

class KeyboardPolicy(object):
    def __init__(self, env):
        self.env = env
        self.config = self.env.config
        self.reward = self.env.reward
        self.spaces = self.env.spaces

    def usage(self):
        return ("usage:\n" +
                "  discard [0-{}]\n".format(self.config.hand_size - 1) +
                "  play    [0-{}]\n".format(self.config.hand_size - 1) +
                "  color   [{}] [0-{}]\n".format(
                    "|".join(self.config.colors), self.config.num_players - 2) +
                "  number  [1-{}] [0-{}]".format(
                    len(self.config.card_counts), self.config.num_players - 2))

    def valid_card_index(self, s):
        return is_int(s) and 0 <= int(s) < self.config.hand_size

    def valid_card_number(self, s):
        return is_int(s) and 0 < int(s) <= len(self.config.card_counts)

    def get_move(self):
        parts = input("> ").split()
        if len(parts) < 2:
            print(self.usage())
            return self.get_move()
        if len(parts) == 2:
            command, arg = parts
            if command == "discard" and self.valid_card_index(arg):
                return hanabi.DiscardMove(int(arg))
            elif command == "play" and self.valid_card_index(arg):
                return hanabi.PlayMove(int(arg))
            else:
                print(self.usage())
                return self.get_move()
        elif len(parts) == 3:
            command, arg, player_index = parts
            if command == "color" and arg in self.config.colors:
                return hanabi.InformColorMove(arg, int(player_index))
            elif command == "number" and self.valid_card_number(arg):
                return hanabi.InformNumberMove(int(arg), int(player_index))
            else:
                print(self.usage())
                return self.get_move()
        else:
            print(self.usage())
            return self.get_move()

    def get_action(self, _observation):
        return (self.spaces.action_to_sample(self.get_move()), )
