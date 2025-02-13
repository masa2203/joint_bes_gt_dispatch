import itertools
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
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'ppo', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

# ACTIONS
discretization_params = {
    'gt_specs': [{'start': -1, 'stop': 1, 'num': 12}],  # CS1 best: 6 (disc.)
    'bes_specs': [{'start': -1, 'stop': 1, 'num': 7}],  # CS1 best: 14 (disc.)
}
# DISCRETE_ACTIONS = None  # None -> continuous PPO
DISCRETE_ACTIONS = generate_discrete_actions(**discretization_params)


# EXP PARAMS
EXP_PARAMS = {
    'n_runs': 5,
    'n_episodes': 200,
    'len_episode': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']),
    'seed': 22,
    # Env
    'flatten_obs': True,
    'frame_stack': 2,  # CS1 best: 5 (disc.) | 1 (cont.)
    # Normalization
    'norm_obs': True,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    'eval_freq': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']) * 1,
}

# PPO PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    'device': 'cpu',
    'learning_rate': linear_scheduler_sb3(0.000311),  # Default: 3e-4, CS1 best: 0.000739 (disc.) | 0.000662 (cont.)
    'n_steps': 2048,  # Default: 2048, CS1 best: 512 (disc.) | 512 (cont.)
    'batch_size': 512,  # Default: 64, CS1 best: 256 (disc.) | 64 (cont.)
    'n_epochs': 20,  # Default: 10, CS1 best: 5 (disc.) | 10 (cont.)
    'gamma': 0.9096,  # Default: 0.99, CS1 best: 0.99997 (disc.) | 0.9985 (cont.)
    'gae_lambda': 0.9126,  # Default: 0.95, CS1 best: 0.91006 (disc.) | 0.9175 (cont.)
    'clip_range': 0.1,  # Default: 0.2, CS1 best: 0.3 (disc.) | 0.1 (cont.)
    'clip_range_vf': 0.2,  # Default: None, CS1 best: 0.3 (disc.) | 0.2 (cont.)
    'normalize_advantage': True,  # Default: True, CS1 best: True (disc.) | True (cont.)
    'ent_coef': 0.0,  # Default: 0.0, CS1 best: 0.23727 (disc.) | 0.0 (cont.)
    'vf_coef': 0.5,  # Default: 0.5, CS1 best: 0.74066 (disc.) | 0.5 (cont.)
    'max_grad_norm': 0.277,  # Default: 0.5, CS1 best: 2.508 (disc.) | 0.5 (cont.)
    'use_sde': False,  # Default: False, CS1 best: False (disc.) | False (cont.)
    'target_kl': None,  # Default: None, CS1 best: None (disc.) | None (cont.)

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'extra_large',  # Default: None, CS1 best: 'large' (disc.) | 'large' (cont.)
        'activation_fn': 'relu',  # Default: tanh, CS1 best: 'relu' (disc.) | 'relu' (cont.)
        'ortho_init': True,  # Default: True, CS1 best: False (disc.) | True (cont.)
        'use_expln': False,  # Default: False, CS1 best: False (disc.) | False (cont.)
        'squash_output': False,  # Default: False, CS1 best: False (disc.) | False (cont.)
        'share_features_extractor': False,  # Default: True, CS1 best: True (disc.) | True (cont.)
    }
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'DISCRETE_ACTIONS': discretization_params if DISCRETE_ACTIONS is not None else None,
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
            'PC_NAME': socket.gethostname(),
        }, f)


RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

for run in range(EXP_PARAMS['n_runs']):
    train_rl_agent(
        agent='ppo',
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
