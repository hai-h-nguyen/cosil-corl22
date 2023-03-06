from gym.envs.registration import register
import pdomains

register(
    "Bumps-1D-v0",
    entry_point='pdomains.bumps_1d:Bumps1DEnv',
    max_episode_steps=100,
)

register(
    "Bumps-2D-v0",
    entry_point='pdomains.bumps_2d:Bumps2DEnv',
    max_episode_steps=100,
)

register(
    "Car-Flag-Continuous-v0",
    entry_point='pdomains.car_flag_continuous:CarEnv',
    max_episode_steps=160,
)

register(
    "Lunar-Lander-P-v0",
    entry_point='pdomains.lunarlander_p:LunarLanderEnv',
    max_episode_steps=160,
)

register(
    "Lunar-Lander-V-v0",
    entry_point='pdomains.lunarlander_v:LunarLanderEnv',
    max_episode_steps=160,
)

register(
    "Block-Picking-v0",
    entry_point='pdomains.block_picking:BlockEnv',
    max_episode_steps=50,
)
