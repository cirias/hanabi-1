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
        policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(16, 16))
        # policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(16, 16),
                # hidden_nonlinearity=lasagne.nonlinearities.rectify)
        # policy = CategoricalGRUPolicy(env_spec=env.spec)

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4 * 1000,
        max_path_length=env.horizon,
        n_itr=50,
        discount=1,
        step_size=0.001,
        gae_lambda=0.5,
    )
    algo.train()
    with open(args.output_policy, "wb") as f:
        pickle.dump(policy, f)

if __name__ == "__main__":
    main(gym_hanabi.policies.common.get_self_parser().parse_args())
