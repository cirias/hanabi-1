from gym_hanabi.policies import *
from rllab.algos.cem import CEM
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
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
    algo = CEM(
        env=env,
        policy=policy,
        n_itr=1000,
        max_path_length=env.horizon,
        discount=1,
        batch_size=4000,
    )
    algo.train()
    with open(args.output_policy, "wb") as f:
        pickle.dump(policy, f)

if __name__ == "__main__":
    main(gym_hanabi.policies.common.get_self_parser().parse_args())
