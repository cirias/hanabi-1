from gym.envs.registration import register

import gym_hanabi.envs
import gym_hanabi.policies
from gym_hanabi.envs import hanabi
from gym_hanabi.envs import hanabi_config
from gym_hanabi.envs import hanabi_reward
from gym_hanabi.envs import hanabi_spaces

# Self.
register(
    id='HanabiSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={
        "config": hanabi_config.HANABI_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MediumHanabiSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={
        "config": hanabi_config.MEDIUM_HANABI_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.MEDIUM_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.MINI_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiLotsOfInfoSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_LOTSOFINFO_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.NestedSpaces(
            hanabi_config.MINI_HANABI_LOTSOFINFO_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiLotsOfTurnsSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_LOTSOFTURNS_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.NestedSpaces(
            hanabi_config.MINI_HANABI_LOTSOFTURNS_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiLinearRewardSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_CONFIG,
        "reward": hanabi_reward.LinearReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.MINI_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiSquaredRewardSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_CONFIG,
        "reward": hanabi_reward.SquaredReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.MINI_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiSkewedRewardSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_CONFIG,
        "reward": hanabi_reward.SkewedReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.MINI_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabi3PSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_3P_CONFIG,
        "reward": hanabi_reward.SkewedReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.MINI_HANABI_3P_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiFlattenedSpaceSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.FlattenedSpaces(hanabi_config.MINI_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiFlattenedSpace3PSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_3P_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.FlattenedSpaces(hanabi_config.MINI_HANABI_3P_CONFIG),
    },
    max_episode_steps=200,
)

SELF_ENV_IDS = [
    "HanabiSelf-v0",
    "MediumHanabiSelf-v0",
    "MiniHanabiSelf-v0",
    "MiniHanabiLotsOfInfoSelf-v0",
    "MiniHanabiLotsOfTurnsSelf-v0",
    "MiniHanabiLinearRewardSelf-v0",
    "MiniHanabiSquaredRewardSelf-v0",
    "MiniHanabiSkewedRewardSelf-v0",
    "MiniHanabi3PSelf-v0",
    "MiniHanabiFlattenedSpaceSelf-v0",
    "MiniHanabiFlattenedSpace3PSelf-v0",
]

# Ai.
register(
    id='HanabiAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={
        "config": hanabi_config.HANABI_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MediumHanabiAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={
        "config": hanabi_config.MEDIUM_HANABI_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.MEDIUM_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.MINI_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiLotsOfInfoAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_LOTSOFINFO_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.NestedSpaces(
            hanabi_config.MINI_HANABI_LOTSOFINFO_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiLotsOfTurnsAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_LOTSOFTURNS_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.NestedSpaces(
            hanabi_config.MINI_HANABI_LOTSOFTURNS_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiLinearRewardAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_CONFIG,
        "reward": hanabi_reward.LinearReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.MINI_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiSquaredRewardAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_CONFIG,
        "reward": hanabi_reward.SquaredReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.MINI_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiSkewedRewardAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_CONFIG,
        "reward": hanabi_reward.SkewedReward(),
        "spaces": hanabi_spaces.NestedSpaces(hanabi_config.MINI_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

register(
    id='MiniHanabiFlattenedSpaceAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={
        "config": hanabi_config.MINI_HANABI_CONFIG,
        "reward": hanabi_reward.ConstantReward(),
        "spaces": hanabi_spaces.FlattenedSpaces(hanabi_config.MINI_HANABI_CONFIG),
    },
    max_episode_steps=200,
)

AI_ENV_IDS = [
    "HanabiAi-v0",
    "MediumHanabiAi-v0",
    "MiniHanabiAi-v0",
    "MiniHanabiLotsOfInfoAi-v0",
    "MiniHanabiLotsOfTurnsAi-v0",
    "MiniHanabiLinearRewardAi-v0",
    "MiniHanabiSquaredRewardAi-v0",
    "MiniHanabiSkewedRewardAi-v0",
    "MiniHanabiFlattenedSpaceAi-v0",
]
