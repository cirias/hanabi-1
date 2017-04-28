from gym_hanabi.policies import *
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
import gym_hanabi
import pickle

def main(args):
    env = GymEnv(args.env_id)

    # Load the AI policy.
    with open(args.ai_policy, "rb") as f:
        env.env.unwrapped.ai_policy = pickle.load(f)

    # If the user provided a starting policy, use it. Otherwise, we start with
    # a fresh policy.
    if args.input_policy is not None:
        with open(args.input_policy, "rb") as f:
            policy = pickle.load(f)
    else:
        policy = CategoricalMLPPolicy(env_spec=env.spec, hidden_sizes=(8, 8))

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=env.horizon,
        n_itr=50,
        discount=0.99,
        step_size=0.01,
    )
    algo.train()
    with open(args.output_policy, "wb") as f:
        pickle.dump(policy, f)

if __name__ == "__main__":
    main(gym_hanabi.policies.common.get_ai_parser().parse_args())
