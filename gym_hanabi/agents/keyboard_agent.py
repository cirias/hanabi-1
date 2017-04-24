import gym
import gym_hanabi
from gym_hanabi.envs import hanabi_env

HAND_SIZE = hanabi_env.HAND_SIZE
COLORS = hanabi_env.Colors.COLORS

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def valid_card_index(s):
    return is_int(s) and 0 <= int(s) < HAND_SIZE

def usage():
    return ("usage:\n" +
            "  discard [0-{}]\n".format(HAND_SIZE - 1) +
            "  play    [0-{}]\n".format(HAND_SIZE - 1) +
            "  color   [{}]\n".format("|".join(COLORS)) +
            "  number  [0-{}]".format(HAND_SIZE - 1))

def act():
    parts = input("> ").split()

    if len(parts) != 2:
        print(usage())
        return act()
    command, arg = parts

    if command == "discard" and valid_card_index(arg):
        return hanabi_env.move_to_sample(hanabi_env.DiscardMove(int(arg)))
    elif command == "play" and valid_card_index(arg):
        return hanabi_env.move_to_sample(hanabi_env.PlayMove(int(arg)))
    elif command == "color" and arg in COLORS:
        return hanabi_env.move_to_sample(hanabi_env.InformColorMove(arg))
    elif command == "number" and valid_card_index(arg):
        return hanabi_env.move_to_sample(hanabi_env.InformNumberMove(int(arg)))
    else:
        print(usage())
        return act()

def main():
    env = gym.make('Hanabi-v0')
    observation = env.reset()
    done = False
    while not done:
        env.render()
        observation, reward, done, info = env.step(act())

if __name__ == "__main__":
    main()
