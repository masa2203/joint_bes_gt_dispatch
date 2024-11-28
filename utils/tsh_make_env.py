from typing import Optional, Dict, Any

from gymnasium.wrappers import FlattenObservation, NormalizeObservation, FrameStack, NormalizeReward
from tianshou.env import DummyVectorEnv, VectorEnvNormObs

from utils.wrappers import *


def tsh_make_env(
        env,
        env_kwargs,
        discrete_actions: Optional[list] = None,
        frame_stack: Optional[int] = None,
        # norm_obs: bool = True,  # Disabled, as updating statistics cannot be paused during eval
        norm_reward: bool = False,
        tracking: bool = True
):
    env = env(**env_kwargs, tracking=tracking)
    env.is_async = False
    env = RescaleActionSpace(env)
    env = FlattenObservation(env)
    # Add discrete action wrapper
    if discrete_actions is not None:
        env = DiscreteActions(env, discrete_actions)
    # if norm_obs:
    #     env = NormalizeObservation(env)
    if norm_reward:
        env = NormalizeReward(env)
    # Stack observation
    if frame_stack is not None:
        env = FrameStack(env, frame_stack)
    env = TemporalInfoAdjustWrapper(env)
    return env


def tsh_make_vec_env(
        env_id,
        env_kwargs,
        discrete_actions: Optional[list] = None,
        frame_stack: Optional[int] = None,
        norm_obs: bool = True,  # Using VectorEnvNormObs
        norm_reward: bool = False,
        tracking: bool = True,
        n_envs: int = 1,
):
    # Convert to vec-env
    env = DummyVectorEnv(
        [lambda: tsh_make_env(
            env=env_id,
            env_kwargs=env_kwargs,
            discrete_actions=discrete_actions,
            frame_stack=frame_stack,
            norm_reward=norm_reward,
            tracking=tracking
        ) for _ in range(n_envs)]
    )
    if norm_obs:
        env = VectorEnvNormObs(env)

    return env
