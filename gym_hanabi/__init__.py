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
    id='MiniHanabiSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.MINI_HANABI_CONFIG},
    max_episode_steps=200,
)

# Ai.
register(
    id='HanabiAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.HANABI_CONFIG},
    max_episode_steps=200,
)

register(
    id='MiniHanabiAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    kwargs={"config": gym_hanabi.envs.hanabi_env.MINI_HANABI_CONFIG},
    max_episode_steps=200,
)
