from gym.envs.registration import register

register(
    id='hanabi-v0',
    entry_point='gym_hanabi.envs:HanabiEnv',
)
