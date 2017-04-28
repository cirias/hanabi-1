from gym.envs.registration import register
import gym_hanabi.envs
import gym_hanabi.policies

# Self.
register(
    id='HanabiSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.HANABI_CONFIG},
    max_episode_steps=200,
)

register(
    id='MediumHanabiSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.MEDIUM_HANABI_CONFIG},
    max_episode_steps=200,
)

register(
    id='MiniHanabiSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.MINI_HANABI_CONFIG},
    max_episode_steps=200,
)

register(
    id='MiniHanabiLotsOfInfoSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.MINI_HANABI_LOTSOFINFO_CONFIG},
    max_episode_steps=200,
)

register(
    id='MiniHanabiLotsOfTurnsSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.MINI_HANABI_LOTSOFTURNS_CONFIG},
    max_episode_steps=200,
)

SELF_ENV_IDS = [
    "HanabiSelf-v0",
    "MediumHanabiSelf-v0",
    "MiniHanabiSelf-v0",
    "MiniHanabiLotsOfInfoSelf-v0",
    "MiniHanabiLotsOfTurnsSelf-v0",
]

# Ai.
register(
    id='HanabiAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.HANABI_CONFIG},
    max_episode_steps=200,
)

register(
    id='MediumHanabiAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.MEDIUM_HANABI_CONFIG},
    max_episode_steps=200,
)

register(
    id='MiniHanabiAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.MINI_HANABI_CONFIG},
    max_episode_steps=200,
)

register(
    id='MiniHanabiLotsOfInfoAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.MINI_HANABI_LOTSOFINFO_CONFIG},
    max_episode_steps=200,
)

register(
    id='MiniHanabiLotsOfTurnsAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.MINI_HANABI_LOTSOFTURNS_CONFIG},
    max_episode_steps=200,
)

AI_ENV_IDS = [
    "HanabiAi-v0",
    "MediumHanabiAi-v0",
    "MiniHanabiAi-v0",
    "MiniHanabiLotsOfInfoAi-v0",
    "MiniHanabiLotsOfTurnsAi-v0",
]
