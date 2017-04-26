import gym_hanabi
from gym_hanabi.envs import hanabi_env
from six.moves import input

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

class KeyboardPolicy(object):
    @staticmethod
    def usage():
        return ("usage:\n" +
                "  discard [0-{}]\n".format(hanabi_env.HAND_SIZE - 1) +
                "  play    [0-{}]\n".format(hanabi_env.HAND_SIZE - 1) +
                "  color   [{}]\n".format("|".join(hanabi_env.Colors.COLORS)) +
                "  number  [1-{}]".format(len(hanabi_env.CARD_COUNTS)))

    @staticmethod
    def valid_card_index(s):
        return is_int(s) and 0 <= int(s) < hanabi_env.HAND_SIZE

    @staticmethod
    def valid_card_number(s):
        return is_int(s) and 0 < int(s) <= len(hanabi_env.CARD_COUNTS)

    @staticmethod
    def get_move():
        parts = input("> ").split()
        if len(parts) != 2:
            print(KeyboardPolicy.usage())
            return KeyboardPolicy.get_move()
        command, arg = parts

        if command == "discard" and KeyboardPolicy.valid_card_index(arg):
            return hanabi_env.DiscardMove(int(arg))
        elif command == "play" and KeyboardPolicy.valid_card_index(arg):
            return hanabi_env.PlayMove(int(arg))
        elif command == "color" and arg in hanabi_env.Colors.COLORS:
            return hanabi_env.InformColorMove(arg)
        elif command == "number" and KeyboardPolicy.valid_card_number(arg):
            return hanabi_env.InformNumberMove(int(arg))
        else:
            print(KeyboardPolicy.usage())
            return KeyboardPolicy.get_move()

    def get_action(self, _observation):
        return (hanabi_env.move_to_sample(KeyboardPolicy.get_move()), )
