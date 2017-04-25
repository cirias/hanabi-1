from gym.envs.registration import register

register(
    id='HanabiSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
)

register(
    id='HanabiAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
)
