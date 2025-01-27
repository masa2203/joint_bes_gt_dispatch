import os
import json
import socket
from typing import Any

import torch

from config import src_dir
from envs.environments import *
from envs.env_params import *
from train.tsh_train_dsac import tsh_train_dsac_agent
from utils.wrappers import *
from utils.utilities import generate_discrete_actions
from utils.logger import tsh_create_stats_file
from utils.net_design import activation_fn_dict, net_arch_dict

# PLANT PARAMS
ENV = GasTurbineBatteryRenewablesDemandEnv
ENV_KWARGS = cs1

# LOG
CREATE_LOG = True
VERBOSE = False
LOGGER_TYPE = ["tensorboard"]  # ["tensorboard"] currently only option
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'tsh_dsac', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

# ACTIONS
discretization_params = {
    'gt_specs': [{'start': -1, 'stop': 1, 'num': 8}],
    'bes_specs': [{'start': -1, 'stop': 1, 'num': 13}],
}
DISCRETE_ACTIONS = generate_discrete_actions(**discretization_params)

# EXP PARAMS
EXP_PARAMS: dict[str, Any] = {
    'n_runs': 5,
    'n_epochs': 200,
    'step_per_epoch': ENV_KWARGS['modeling_period_h'] * 100,
    'seed': 22,
    # Env
    'flatten_obs': True,
    'frame_stack': 1,
    'n_train_env': 1,
    'n_test_env': 1,
    # Normalization
    'norm_obs': True,
    'norm_reward': True,  # Always False for eval
    # Evaluation
    'eval_while_training': True,  # Runs one evaluation episode after each training epoch
    'device': "cuda" if torch.cuda.is_available() else "cpu"
}

# DQN PARAMS
RL_PARAMS: dict[str, Any] = {
    'lr_actor': 2.1511e-5,  # Default: 1e-3
    'lr_critic': 1.251e-4,  # Default: 1e-3
    'use_lr_scheduler': True,  # Default: False
    'actor_softmax': True,  # Default: True
    'share_net': False,  # No default
    'buffer_size': 1_000_000,  # No default
    'batch_size': 128,  # No default
    'gamma': 0.9162,  # Default: 0.99
    'tau': 0.001,  # Default: 0.005
    'auto_alpha': True,  # Default: True (automatic learning of temperature)
    'alpha': 0.2,  # Default: 0.2 (temperature, float or tuple including optimizer)
    'train_freq': 400,  # No default, 219 matches 8760h
    'estimation_step': 2,  # Default: 1, for n-step return
    'update_per_step': 1,  # Default: 1.0
    'net_arch': 'sac',  # No default
    'activation_fn': 'tanh',  # Default: relu
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'DISCRETE_ACTIONS': discretization_params,
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
            'PC_NAME': socket.gethostname(),
        }, f)

RL_PARAMS['net_arch'] = net_arch_dict[RL_PARAMS['net_arch']]['qf']
RL_PARAMS['activation_fn'] = activation_fn_dict[RL_PARAMS['activation_fn']]

# TRAINING
for run in range(EXP_PARAMS['n_runs']):
    tsh_train_dsac_agent(
        run=run,
        path=SAVE_PATH,
        exp_params=EXP_PARAMS,
        env_id=ENV,
        env_kwargs=ENV_KWARGS,
        discrete_actions=DISCRETE_ACTIONS,
        rl_params=RL_PARAMS,
        verbose=VERBOSE,
        logger_type=LOGGER_TYPE,
    )

# GET STATISTICS FROM MULTIPLE RUNS
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
    if EXP_PARAMS['eval_while_training']:
        tsh_create_stats_file(path=SAVE_PATH, style='eval')
    tsh_create_stats_file(path=SAVE_PATH, style='train')
