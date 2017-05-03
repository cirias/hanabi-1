from gym_hanabi.policies import *
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
import gym_hanabi
import lasagne.nonlinearities
import os
import pickle

def main(args):
    logger.set_snapshot_dir(args.snapshot_dir)
    logger.set_snapshot_mode("none")
    logger.add_tabular_output(os.path.join(args.snapshot_dir, "tabular.csv"))
    env = GymEnv(args.env_id)

    # If the user provided a starting policy, use it. Otherwise, we start with
    # a fresh policy.
    if args.input_policy is not None:
        with open(args.input_policy, "rb") as f:
            policy = pickle.load(f)
    else:
        policy = CategoricalMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=args.hidden_sizes)

        # policy = CategoricalMLPPolicy(
        #     env_spec=env.spec,
        #     hidden_sizes=(16, 16),
        #     hidden_nonlinearity=lasagne.nonlinearities.rectify)

        # policy = CategoricalGRUPolicy(env_spec=env.spec)

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=args.batch_size,
        max_path_length=env.horizon,
        n_itr=args.n_itr,
        discount=args.discount,
        step_size=args.step_size,
        gae_lambda=args.gae_lambda,
    )
    algo.train()
    with open(args.output_policy, "wb") as f:
        pickle.dump(policy, f)

def get_parser():
    def parse_tuple(s):
        return eval(s)

    parser = gym_hanabi.policies.common.get_self_parser()
    parser.add_argument("--hidden_sizes", type=parse_tuple, default=(16, 16))
    parser.add_argument("--batch_size", type=int, default=4000)
    parser.add_argument("--n_itr", type=int, default=50)
    parser.add_argument("--discount", type=float, default=1)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--gae_lambda", type=float, default=0.9)

    return parser

if __name__ == "__main__":
    main(get_parser().parse_args())
