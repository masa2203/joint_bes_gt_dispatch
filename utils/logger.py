import os
import time
import warnings
import pandas as pd
import pickle
import json
import numpy as np
from typing import Optional

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

from tianshou.env import BaseVectorEnv, DummyVectorEnv
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from envs.gt_model import A35, A05
from envs.storage_model import BESS, DODDegradingBESS

from utils.utilities import start_count


def get_env_log_data(env, mean_reward, start_time):
    """
    Gets the data to be logged for a given environment.

    :param env: A gym environment.
    :param mean_reward: A float that represents the mean reward.
    :param start_time: A float that represents the start time.
    :return: A dictionary that contains the data to be logged.
    """
    gt_classes = [A35, A05]
    bes_classes = [BESS, DODDegradingBESS]

    # Environment wrapped for stable-baslines3
    if isinstance(env, VecNormalize) or isinstance(env, DummyVecEnv):
        # episode_info = env.env_method('return_episode_info')[0]
        # episode_info = env.venv.envs[0].return_episode_info()
        episode_info = env.unwrapped.envs[0].unwrapped.return_episode_info()
        has_gt = ((hasattr(env.unwrapped.envs[0].unwrapped, 'gt') and
                  isinstance(env.unwrapped.envs[0].unwrapped.gt, tuple(gt_classes))) or
                  (hasattr(env.unwrapped.envs[0].unwrapped, 'gts') and
                  isinstance(env.unwrapped.envs[0].unwrapped.gts[0], tuple(gt_classes))))
        has_bes = (hasattr(env.unwrapped.envs[0].unwrapped, 'storage') and
                   isinstance(env.unwrapped.envs[0].unwrapped.storage, tuple(bes_classes)))

    # Environment wrapped for Tianshou
    elif isinstance(env, BaseVectorEnv) or isinstance(env, DummyVectorEnv):
        episode_info = env.get_env_attr('env_log')[0]
        # Attempt to retrieve multiple GTs first
        try:
            gts = env.get_env_attr('gts')[0]  # This may raise an AttributeError if 'gts' does not exist
            if isinstance(gts, list) and len(gts) > 0:  # Check if it's a non-empty list
                has_gt = any(isinstance(gt, tuple(gt_classes)) for gt in gts)
            else:
                has_gt = False  # Fallback to False if 'gts' is not a non-empty list
        except AttributeError:
            # If 'gts' does not exist, check for a single GT
            try:
                gt = env.get_env_attr('gt')[0]  # This might also raise an AttributeError if 'gt' does not exist
                has_gt = isinstance(gt, tuple(gt_classes))
            except AttributeError:
                has_gt = False  # Neither 'gts' nor 'gt' exists
        # try:
        #     has_gt = isinstance(env.get_env_attr('gt')[0], tuple(gt_classes))
        # except AttributeError:
        #     has_gt = False
        try:
            has_bes = isinstance(env.get_env_attr('storage')[0], tuple(bes_classes))
        except AttributeError:
            has_bes = False

    # Unwrapped environment
    else:
        episode_info = env.return_episode_info()
        has_gt = (hasattr(env, 'gt') and isinstance(env.gt, tuple(gt_classes)) or
                  hasattr(env, 'gts') and isinstance(env.gts[0], tuple(gt_classes)))
        has_bes = hasattr(env, 'storage') and isinstance(env.storage, tuple(bes_classes))

    stats = {
        'reward_sum': mean_reward,
        'compute_time': time.time() - start_time,
    }

    # If GT(s) is/are part of the env
    if has_gt:
        if has_bes:
            gt_actions = np.array([a[:-1] for a in episode_info['actions']])  # bes action is last action
        else:
            gt_actions = np.array(episode_info['actions'])

        # Ensure gt_actions is at least 2D (where the second dimension is GTs)
        if gt_actions.ndim == 1:
            gt_actions = np.expand_dims(gt_actions, axis=-1)  # Make single GT actions 2D

        # Calculate operating hours and starts
        gt_oper_hours = [sum(1 for action in gt if action > 0) for gt in gt_actions.T]
        number_of_starts = [start_count(gt) for gt in gt_actions.T]

        # Calculate average loads
        avg_gt_load = [np.mean(gt) for gt in gt_actions.T]
        avg_gt_load_when_on = [np.mean([action for action in gt if action > 0]) for gt in gt_actions.T]

        # Get list of sums by GT for fuel, carbon tax, maintenance
        if isinstance(episode_info['fuel_costs'][0], list):
            gt_fuel_costs = list(map(list, zip(*episode_info['fuel_costs'])))
            fuel_cost_sum = [sum(gt) for gt in gt_fuel_costs]
        else:
            fuel_cost_sum = [sum(episode_info['fuel_costs'])]

        if isinstance(episode_info['carbon_taxes'][0], list):
            gt_carbon_taxes = list(map(list, zip(*episode_info['carbon_taxes'])))
            carbon_tax_sum = [sum(gt) for gt in gt_carbon_taxes]
        else:
            carbon_tax_sum = [sum(episode_info['carbon_taxes'])]

        if isinstance(episode_info['maintenance_costs'][0], list):
            gt_maintenance_costs = list(map(list, zip(*episode_info['maintenance_costs'])))
            maint_cost_sum = [sum(gt) for gt in gt_maintenance_costs]
        else:
            maint_cost_sum = [sum(episode_info['maintenance_costs'])]

        gt_stats = {
            'fuel_cost_sum': fuel_cost_sum,
            'carbon_tax_sum': carbon_tax_sum,
            'maint_cost_sum': maint_cost_sum,
            'avg_GT_load': avg_gt_load,
            'avg_GT_load_when_on': avg_gt_load_when_on,
            'operating_hours_GT': gt_oper_hours,
            'number_of_starts': number_of_starts,
        }
        stats = stats | gt_stats

    # If BES is part of the env
    if has_bes:
        discharge_count = len(list(filter(lambda x: (x > 0), episode_info['bes_power_flows'])))
        charge_count = len(list(filter(lambda x: (x < 0), episode_info['bes_power_flows'])))

        bes_stats = {
            'degr_cost_sum': sum(episode_info['degr_costs']),
            'avg_soc': sum(episode_info['socs']) / len(episode_info['socs']),
            'num_charging': charge_count,
            'num_discharging': discharge_count,
        }
        stats = stats | bes_stats

    # If balances were tracked
    if episode_info['e_balances']:
        oversupply_values = list(filter(lambda x: (x > 0), episode_info['e_balances']))
        num_oversupply = len(oversupply_values)
        avg_oversupply = sum(oversupply_values) / num_oversupply if num_oversupply > 0 else 0

        undersupply_values = list(filter(lambda x: (x < 0), episode_info['e_balances']))
        num_undersupply = len(undersupply_values)
        avg_undersupply = sum(undersupply_values) / num_undersupply if num_undersupply > 0 else 0

        demand_balancing_stats = {
            'num_oversupply': num_oversupply,
            'avg_oversupply': avg_oversupply,
            'num_undersupply': num_undersupply,
            'avg_undersupply': avg_undersupply,
        }
        stats = stats | demand_balancing_stats

    # Add tracked time-series
    log_data = stats | episode_info

    return log_data


def sb3_create_stats_file(path, exp_params, style: str = 'eval'):
    """
    Creates a CSV file that contains the training and evaluation statistics of multiple independent runs
    based on stable-baseline3's monitor-files.

    :param path: A string that represents the path to the directory where the CSV file will be created.
    :param exp_params: A dictionary that contains the experiment parameters.
    :param style: A string that can be 'eval' or 'train' to indicate which file to create.
    """
    assert style == 'eval' or style == 'train', 'Valid styles are "eval" and "train"!'

    def make_stats_frame(monitor_df: pd.DataFrame):
        """Returns a pd.Dataframe with episodes and steps to be populated with rewards."""
        len_mon = len(monitor_df) - 1  # -1 due to monitor header
        n_eps = exp_params['n_episodes']
        data = {}
        if style == 'train':
            if n_eps + 1 != len_mon:  # +1 due to 'mandatory' final eval episode
                warnings.warn('Mismatch between chosen and conducted episodes.')
            n_cols = len_mon
            data = {
                'episodes': [i for i in range(n_cols)],
                'steps': [i * exp_params['len_episode'] for i in range(n_cols)]
            }
        elif style == 'eval':
            if int(n_eps * exp_params['len_episode'] / exp_params['eval_freq']) + 1 != len_mon:
                print('In here')
                warnings.warn('Mismatch between chosen and conducted episodes.')
            n_cols = len_mon
            data = {
                'episodes': [i for i in range(n_cols)],
                'steps': [i * exp_params['eval_freq'] for i in range(n_cols)]
            }
        else:
            warnings.warn('Unsupported stats style chosen. Created stats file with no columns. '
                          'Supported styles are "train" and "valid"')

        frame = pd.DataFrame(data).T
        return frame

    stats_frame = None
    count = 0
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('monitor.csv') and style in file:
                df = pd.read_csv(os.path.join(subdir, file))
                stats_frame = make_stats_frame(df) if stats_frame is None else stats_frame
                stats_frame.loc[count] = [float(i) for i in df.index.values.tolist()[1:]]
                count += 1

    numerical_rows = stats_frame.iloc[2:]  # This excludes the first two non-numerical rows

    # Calculate mean and standard deviation for the numerical part
    mean_row = numerical_rows.mean(axis=0)
    std_row = numerical_rows.std(axis=0)

    # Add these as new rows to the DataFrame
    stats_frame.loc['mean'] = mean_row
    stats_frame.loc['std'] = std_row

    stats_frame.to_csv(os.path.join(path, style+'_stats.csv'))

    if style == 'eval':
        # Print evaluation stats if available
        print('Mean episodic rewards over all evaluation runs: ')
        print(mean_row)


def tsh_create_stats_file(path, style: str = 'eval'):
    """
    Creates a CSV file that contains the training and evaluation statistics of multiple independent runs
    based on csv-files extracted from Tianshou's TensorboardLogger.

    :param path: A string that represents the path to the directory where the CSV file will be created.
    :param style: A string that can be 'eval' or 'train' to indicate which file to create.
    """
    assert style == 'eval' or style == 'train', 'Valid styles are "eval" and "train"!'

    def make_stats_frame(monitor_df: pd.DataFrame):
        """Returns a pd.Dataframe with episodes and steps to be populated with rewards."""
        n_cols = len(monitor_df) - 1  # -1 due to monitor header

        data = {
            'episodes': monitor_df.Episode.tolist(),
            'steps': monitor_df.Step.tolist()
        }
        frame = pd.DataFrame(data).T
        return frame

    stats_frame = None
    count = 0
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(f'{style}_rewards.csv'):
                df = pd.read_csv(os.path.join(subdir, file))
                assert all(column in df.columns for column in
                           ['Step', 'Episode', 'Reward']), "DataFrame is missing one or more required columns."
                stats_frame = make_stats_frame(df) if stats_frame is None else stats_frame
                stats_frame.loc[count] = [float(i) for i in df.Reward.values.tolist()]
                count += 1

    if stats_frame is None:
        return

    numerical_rows = stats_frame.iloc[2:]  # This excludes the first two non-numerical rows

    # Calculate mean and standard deviation for the numerical part
    mean_row = numerical_rows.mean(axis=0)
    std_row = numerical_rows.std(axis=0)

    # Add these as new rows to the DataFrame
    stats_frame.loc['mean'] = mean_row
    stats_frame.loc['std'] = std_row

    stats_frame.to_csv(os.path.join(path, style+'_stats.csv'))

    if style == 'eval':
        # Print evaluation stats if available
        print('Mean episodic rewards over all evaluation runs: ')
        print(mean_row)


def cem_create_stats_file(path: str, len_episode: int):
    """
    Creates a CSV file that contains the training statistics for CEM.

    :param path: A string that represents the path to the directory where the CSV file will be created.
    :param len_episode: An integer that represents the episode length in time-steps.
    """
    data = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('output.json'):
                with open(os.path.join(subdir, file)) as f:
                    log = json.load(f)
                    data.append(log['rewards_vs_steps'])

    eps_and_steps = {
        'episodes': [(i[1]/len_episode)-1 for i in data[0]],  # data[0] pick first run as all should have same pattern
        'steps': [i[1]-len_episode for i in data[0]]  # -1 to subtract first episode and align plots at zero
    }
    stats_frame = pd.DataFrame(eps_and_steps).T

    count = 0
    for run in data:
        stats_frame.loc[count] = [i[0] for i in run]
        count += 1

    numerical_rows = stats_frame.iloc[2:]  # This excludes the first two non-numerical rows

    # Calculate mean and standard deviation for the numerical part
    mean_row = numerical_rows.mean(axis=0)
    std_row = numerical_rows.std(axis=0)

    # Add these as new rows to the DataFrame
    stats_frame.loc['mean'] = mean_row
    stats_frame.loc['std'] = std_row

    stats_frame.to_csv(os.path.join(path, 'stats.csv'))
    print('Mean episodic rewards over all runs: ')
    print()
    print(mean_row)


def print_and_save_tune_logs(study: optuna.study,
                             save_path: Optional[str],
                             run_id: str
                             ):
    """
    Prints and saves the results of an Optuna study.

    :param study: An optuna study object.
    :param save_path: A string that represents the path to the directory where the results will be saved.
    :param run_id: A string that represents the ID of the run.
    """
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    best_trial = study.best_trial

    print(f"  Value: {best_trial.value}")

    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in best_trial.user_attrs.items():
        print(f"    {key}: {value}")

    if save_path is not None:
        # Write report
        study.trials_dataframe().to_csv(os.path.join(save_path, "results.csv"))
        # Save sampler
        with open(os.path.join(save_path, "sampler.pkl"), "wb") as fout:
            pickle.dump(study.sampler, fout)

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    if save_path is not None:
        # Not that saving freezes with the most recent version of Kaleido, an older version must be installed.
        # See: https://github.com/plotly/Kaleido/issues/134
        fig1.write_image(os.path.join(save_path, f"history_{run_id}.png"))
        fig2.write_image(os.path.join(save_path, f"importance_{run_id}.png"))
    else:
        fig1.show()
        fig2.show()