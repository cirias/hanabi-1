import gym
import gym_hanabi
from gym_hanabi.envs import hanabi_env
import sys

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

def get_python_version():
    version = sys.version_info[0]
    assert version in [2, 3]
    return version

def usage():
    return ("usage:\n" +
            "  discard [0-{}]\n".format(HAND_SIZE - 1) +
            "  play    [0-{}]\n".format(HAND_SIZE - 1) +
            "  color   [{}]\n".format("|".join(COLORS)) +
            "  number  [0-{}]".format(HAND_SIZE - 1))

def act():
    # In Python 3, raw_input was renamed to just input.
    if get_python_version() == 2:
        parts = raw_input("> ").split()
    else:
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

def main(env_id):
    env = gym.make(env_id)
    observation = env.reset()
    done = False
    while not done:
        env.render()
        observation, reward, done, info = env.step(act())
        print(info["move"])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python keyboard_agent.py [HanabiSelf-v0|HanabiAi-v0]")
        sys.exit(1)
    main(sys.argv[1])
