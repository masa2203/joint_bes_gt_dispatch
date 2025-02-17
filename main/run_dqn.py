import os
import json
import socket
from typing import Any

from config import src_dir
from envs.environments import *
from envs.env_params import *
from utils.net_design import activation_fn_dict, net_arch_dict
from utils.scheduler import linear_scheduler_sb3
from utils.utilities import generate_discrete_actions
from utils.logger import sb3_create_stats_file
from train.train import train_rl_agent

# PLANT PARAMS
ENV = GasTurbineBatteryRenewablesDemandEnv
ENV_KWARGS = cs2_base

# LOG
CREATE_LOG = True
VERBOSE = 0
LOGGER_TYPE = ["csv"]
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'dqn', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

# ACTIONS
discretization_params = {
    'gt_specs': [
        {'start': -1, 'stop': 1, 'num': 8}  # First GT, CS1 best: 12
        # {'start': -1, 'stop': 1, 'num': 8},  # Second GT ...
        # {'start': -1, 'stop': 1, 'num': 2},
        # {'start': -1, 'stop': 1, 'num': 2},
        # {'start': -1, 'stop': 1, 'num': 2}
    ],
    'bes_specs': [{'start': -1, 'stop': 1, 'num': 19}],  # CS1 best: 12
}
DISCRETE_ACTIONS = generate_discrete_actions(**discretization_params)

# EXP PARAMS
EXP_PARAMS = {
    'n_runs': 5,
    'n_episodes': 200,
    'len_episode': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']),
    'seed': 22,
    # Env
    'combine_gt_actions': False,
    'flatten_obs': True,
    'frame_stack': 7,  # CS1 best: 6
    # Normalization
    'norm_obs': True,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    'eval_freq': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']) * 1,
}

# DQN PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    # 'learning_rate': 0.00037,  # Default: 1e-4
    'learning_rate': linear_scheduler_sb3(0.00037),  # Default: 1e-4, CS1 best: 0.00029
    'buffer_size': 1_000_000,  # Default: 1e6, CS1 best: 1_000_000
    'learning_starts': 1_000,  # Default: 50_000, CS1 best: 10_000
    'batch_size': 128,  # Default: 32, CS1 best: 128
    'tau': 0.335,  # Default: 1.0, CS1 best: 0.9188
    'gamma': 0.965,  # Default: 0.99, CS1 best: 0.9996
    'train_freq': 75,  # Default: 4, CS1 best: 58
    'gradient_steps': -1,  # Default: 1, CS1 best: -1
    'target_update_interval': 10_000,  # Default: 1e4, CS1 best: 1_000
    'exploration_fraction': 0.5,  # Default: 0.1, CS1 best: 0.25
    'exploration_initial_eps': 0.8,  # Default: 1.0, CS1 best: 1.0
    'exploration_final_eps': 0.01,  # Default: 0.05, CS1 best: 0.01
    'max_grad_norm': 0.6,  # Default: 10, CS1 best: 5

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'extra_large',  # Default: None, CS1 best: 'large'
        'activation_fn': 'leaky_relu',  # Default: tanh, CS1 best: 'leaky_relu'
    }
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

RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]['qf']
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

for run in range(EXP_PARAMS['n_runs']):
    train_rl_agent(
        agent='dqn',
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
        sb3_create_stats_file(path=SAVE_PATH, exp_params=EXP_PARAMS, style='eval')
    sb3_create_stats_file(path=SAVE_PATH, exp_params=EXP_PARAMS, style='train')
