import os
import pickle
import json
import random
import itertools
from collections import deque
from typing import Optional, Union, Tuple, Any, Deque, List, Dict

import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils.net_design import activation_fn_dict


def set_seeds(seed):
    """
    Fixes the random seed for all relevant packages.

    :param seed: An integer that represents the seed to be set.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def start_count(lst_of_actions_taken):
    """
    Calculates the number of GT starts from a list of GT dispatches.

    :param lst_of_actions_taken: A list that represents the GT dispatch history.
    :type lst_of_actions_taken: list
    :return: An integer that represents the number of GT starts.
    """
    count = 0
    count += 1 if lst_of_actions_taken[0] else 0
    for i in range(len(lst_of_actions_taken) - 1):
        if lst_of_actions_taken[i] == 0 and lst_of_actions_taken[i + 1] != 0:
            count += 1
    return count


def scaling(x: np.ndarray, lb: np.ndarray, ub: np.ndarray, operation: float):
    """
    Scales or unscales a 2D array of datapoints.

    :param x: A 2D numpy array of size n * nsamples of datapoints.
    :param lb: A 1D numpy array of length n that specifies the lower range of features.
    :param ub: A 1D numpy array of length n that specifies the upper range of features.
    :param operation: A string that indicates whether to scale or unscale.
    :return: A 2D numpy array of size n * nsamples of unscaled datapoints.
    """

    if operation == 'scale':
        # scale
        x_out = (x - lb) / (ub - lb)
        return x_out
    elif operation == 'unscale':
        # unscale
        x_out = lb + x * (ub - lb)
        return x_out


def generate_discrete_actions(
        gt_specs: Optional[List[Dict[str, float]]] = None,
        bes_specs: Optional[List[Dict[str, float]]] = None
) -> List[np.ndarray]:
    """
    Generate discrete action space for GTs and/or BESs based on provided specifications.
    Can also generate actions for exclusively GTs or BESs if only one of them is provided.

    Parameters:
    gt_specs (Optional[List[Dict[str, float]]]): List of dictionaries, each containing
                                                 'start', 'stop', and 'num' for np.linspace for each GT.
    bes_specs (Optional[List[Dict[str, float]]]): List of dictionaries, each containing
                                                  'start', 'stop', and 'num' for np.linspace for each BES.

    Returns:
    List[np.ndarray]: A list containing numpy arrays of all possible action combinations.

    Example:
        gt_specs = [{'start': -1, 'stop': 1, 'num': 9}]
        bes_specs = [{'start': -1, 'stop': 1, 'num': 9}]
        discrete_actions = generate_discrete_actions(gt_specs, bes_specs)
    """
    # Initialize action lists
    gt_actions, bes_actions = [], []

    if gt_specs is not None:
        gt_actions = [np.linspace(**spec) for spec in gt_specs]

    if bes_specs is not None:
        bes_actions = [np.linspace(**spec) for spec in bes_specs]

        # Combine the action sets if both are provided, otherwise use the provided set
    all_actions = gt_actions if len(bes_actions) == 0 else bes_actions if len(
        gt_actions) == 0 else gt_actions + bes_actions

    # Get all combinations
    combinations = list(itertools.product(*all_actions))

    # Convert each combination tuple to a numpy array
    discrete_actions = [np.array(combination) for combination in combinations]

    return discrete_actions


def extract_scalar_from_event(
        event_file_path: str,
        scalar_tag: str,
        len_episode: float | int
) -> pd.DataFrame:
    """
    Extracts scalar data from a TensorFlow event file and converts it into a pandas DataFrame.

    :param event_file_path: The file path to the TensorFlow event file as a string.
    :param scalar_tag: The tag of the scalar data to be extracted as a string.
    :param len_episode: The length of the episode, used for calculating the relative episode value, as a float.

    :return: A pandas DataFrame containing the step, episode, and reward values extracted from the scalar data.
    """
    # Initialize an accumulator and reload it to read from the event file
    accumulator = EventAccumulator(event_file_path)
    accumulator.Reload()

    # Extract scalar data based on the provided tag
    scalar_data = accumulator.Scalars(scalar_tag)

    # Convert the scalar data to a pandas DataFrame and return
    data = pd.DataFrame([(s.step, s.step/len_episode, s.value) for s in scalar_data],
                        columns=['Step', 'Episode', 'Reward'])
    return data
