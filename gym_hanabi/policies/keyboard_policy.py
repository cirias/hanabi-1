from gym_hanabi.envs import hanabi_env
from six.moves import input
import pickle

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

class KeyboardPolicy(object):
    def __init__(self, config):
        self.config = config

    def usage(self):
        return ("usage:\n" +
                "  discard [0-{}]\n".format(self.config.hand_size - 1) +
                "  play    [0-{}]\n".format(self.config.hand_size - 1) +
                "  color   [{}]\n".format("|".join(self.config.colors)) +
                "  number  [1-{}]".format(len(self.config.card_counts)))

    def valid_card_index(self, s):
        return is_int(s) and 0 <= int(s) < self.config.hand_size

    def valid_card_number(self, s):
        return is_int(s) and 0 < int(s) <= len(self.config.card_counts)

    def get_move(self):
        parts = input("> ").split()
        if len(parts) != 2:
            print(self.usage())
            return self.get_move()
        command, arg = parts

        if command == "discard" and self.valid_card_index(arg):
            return hanabi_env.DiscardMove(int(arg))
        elif command == "play" and self.valid_card_index(arg):
            return hanabi_env.PlayMove(int(arg))
        elif command == "color" and arg in self.config.colors:
            return hanabi_env.InformColorMove(arg)
        elif command == "number" and self.valid_card_number(arg):
            return hanabi_env.InformNumberMove(int(arg))
        else:
            print(self.usage())
            return self.get_move()

    def get_action(self, _observation):
        return (hanabi_env.move_to_sample(self.config, self.get_move()), )

if __name__ == "__main__":
    configs_and_names = [
        (hanabi_env.HANABI_CONFIG, "KeyboardPolicy"),
        (hanabi_env.MEDIUM_HANABI_CONFIG, "MediumKeyboardPolicy"),
        (hanabi_env.MINI_HANABI_CONFIG, "MiniKeyboardPolicy"),
        (hanabi_env.MINI_HANABI_LOTSOFINFO_CONFIG, "MiniKeyboardLotsOfInfoPolicy"),
        (hanabi_env.MINI_HANABI_LOTSOFTURNS_CONFIG, "MiniKeyboardLotsOfTurnsPolicy"),
    ]

    for config, name in configs_and_names:
        with open("pickled_policies/{}.pickle".format(name), "wb") as f:
            pickle.dump(KeyboardPolicy(config), f)
