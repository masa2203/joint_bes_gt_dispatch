from typing import Optional, Dict, Any

from gymnasium.wrappers import FlattenObservation
import stable_baselines3.common.vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from utils.wrappers import *


def make_env(env,
             env_kwargs: dict,
             tracking: bool = False,
             allow_early_resets: bool = True,
             path: Optional[str] = None,
             perfect_forecasts: Optional[Dict[str, list]] = None,
             forecasts: Optional[Dict[str, Any]] = None,
             re_forecasts: Optional[Dict[str, Any]] = None,  # Special case of RE prediction  (from wind + PV)
             combine_gt_actions: bool = False,
             use_predefined_discrete_actions: bool = False,
             flatten_obs: bool = True,
             discrete_actions: Optional[list] = None,
             frame_stack: Optional[int] = None,
             norm_obs: bool = True,
             norm_reward: bool = True,
             gamma: float = 0.99,
             ):
    """
    Creates a gym environment and applies a set of wrappers.

    :param env: A subclass of gym.Env that represents the environment.
    :param env_kwargs: A dictionary that represents the keyword arguments to pass to the environment.
    :param tracking: A boolean that indicates whether to track the variables. Default is False.
    :param allow_early_resets: A boolean that indicates whether to allow early resets. Default is True.
    :param path: A string that represents the path to save the monitor. Default is None.
    :param perfect_forecasts: A dictionary with forecasted variables (str) forecast intervals (list). Default is None.
    :param forecasts: A dictionary with paths pointing to trained models and data files. Default is None.
    :param re_forecasts: A dictionary with paths pointing to trained models and data files. Default is None.
    :param combine_gt_actions: A boolean that indicates whether all GT actions are combined into one dimension.
    Default is False.
    :param use_predefined_discrete_actions: A boolean that indicates whether a set of predefined discrete actions
    is used. Default is False.
    :param flatten_obs: A boolean that indicates whether to flatten the observation. Default is True.
    :param discrete_actions: A list that represents the discrete actions. Default is None.
    :param frame_stack: An integer that represents the number of frames to stack. Default is None.
    :param norm_obs: A boolean that indicates whether to normalize the observation. Default is True.
    :param norm_reward: A boolean that indicates whether to normalize the reward. Default is True.
    :param gamma: A float that represents the gamma value. Default is 0.99.
    :return: A stable_baselines3.common.vec_env.VecNormalize object that represents the wrapped environment.
    """
    e = Monitor(env=env(**env_kwargs, tracking=tracking),
                allow_early_resets=allow_early_resets,  # allow finish rollout for PPO -> throws error otherwise
                filename=path)

    if perfect_forecasts is not None:
        e = PerfectForecasts(e, forecasts=perfect_forecasts)

    if forecasts is not None:
        for f in forecasts['models']:  # One wrapper for each pre-trained forecaster
            e = RegularForecasts(env=e,
                                 forecasted_var=f,
                                 log_folder_path=forecasts['models'][f],
                                 path_datafile=forecasts['path_datafile'])

    if re_forecasts is not None:
        e = RenewableForecasts(env=e,
                               log_folder_path=re_forecasts['models'],
                               path_datafile=re_forecasts['path_datafile'])

    if combine_gt_actions:
        e = CombineGTActions(e)

    if use_predefined_discrete_actions:
        e = PreDefinedDiscreteActions(e)

    # Rescale from [0,1] to [-1,1], applies only to first dimension (GT)
    # Must rescale before discretization (if applicable) as RescaleActionSpace doesn't support discrete spaces
    if not use_predefined_discrete_actions:
        e = RescaleActionSpace(e)

    if flatten_obs:
        e = FlattenObservation(e)

    # Add discrete action wrapper
    if discrete_actions is not None:
        e = DiscreteActions(e, discrete_actions)

    e = DummyVecEnv([lambda: e])

    # Stack observation
    if frame_stack is not None:
        e = stable_baselines3.common.vec_env.VecFrameStack(e, n_stack=frame_stack)

    e = VecNormalize(e, norm_obs=norm_obs, norm_reward=norm_reward, gamma=gamma)

    return e
