import gym
import gym_hanabi
import argparse

def main(args):
    env = gym.make("HanabiAi-v0")
    env = gym.wrappers.Monitor(env, args.directory, force=args.force)

    # TODO: For now, these policies are hardcoded. In the future, we should be
    # able to conveniently plug in any policy we want.
    player_policy = gym_hanabi.policies.KeyboardPolicy()
    ai_policy = gym_hanabi.policies.HeuristicPolicy()
    env.unwrapped.ai_policy = ai_policy

    for _ in range(args.num_games):
        observation = env.reset()
        done = False
        while not done:
            if args.render:
                env.render()
            action = player_policy.get_action(observation)[0]
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
