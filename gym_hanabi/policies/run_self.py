import gym
import gym_hanabi
import argparse

def main(args):
    # TODO: For now, this policy is hardcoded. In the future, we should be able
    # to conveniently plug in any policy we want.
    policy = gym_hanabi.policies.KeyboardPolicy()

    env = gym.make("HanabiSelf-v0")
    env = gym.wrappers.Monitor(env, args.directory, force=args.force)

    for _ in range(args.num_games):
        observation = env.reset()
        done = False
        while not done:
            if args.render:
                env.render()
            action = policy.get_action(observation)[0]
            observation, reward, done, info = env.step(action)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render",
        action="store_true", help="Render the game.")
    parser.add_argument("-f", "--force",
        action="store_true", help="Overwrite monitoring directory.")
    parser.add_argument("num_games",
        type=int, help="Number of Hanabi games to play.")
    parser.add_argument("directory",
        type=str, help="Monitoring directory.")
    return parser

if __name__ == "__main__":
    main(get_parser().parse_args())
