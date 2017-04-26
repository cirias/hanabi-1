from gym.envs.registration import register
import gym_hanabi.envs
import gym_hanabi.policies

register(
    id='HanabiSelf-v0',
    entry_point='gym_hanabi.envs:HanabiSelfEnv',
    max_episode_steps=200,
)

register(
    id='HanabiAi-v0',
    entry_point='gym_hanabi.envs:HanabiAiEnv',
    max_episode_steps=200,
)
