from gym_hanabi.policies import *
from rllab.algos.vpg import VPG
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
import argparse
import gym_hanabi
import pickle

def main(args):
    env = GymEnv(args.env_id)

    # If the user provided a starting policy, use it. Otherwise, we start with
    # a fresh policy.
    if args.input_policy is not None:
        with open(args.input_policy, "rb") as f:
            policy = pickle.load(f)
    else:
        policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(32, 32))

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = VPG(
        env=env,
        policy=policy,
        baseline=baseline,
#        n_itr=2000,
#        max_path_length=env.horizon,
#        discount=0.99,
#        batch_size=4000,
    )
    algo.train()
    with open(args.output_policy, "wb") as f:
        pickle.dump(policy, f)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_policy",
        default=None, help="Filename of starting pickled policy")
    parser.add_argument("env_id",
        choices=["HanabiSelf-v0", "MiniHanabiSelf-v0"], help="Environment id")
    parser.add_argument("output_policy",
        help="Filename of final pickled policy")
    return parser

if __name__ == "__main__":
    main(get_parser().parse_args())
