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
    'gt_specs': [{'start': -1, 'stop': 1, 'num': 13}],
    'bes_specs': [{'start': -1, 'stop': 1, 'num': 8}],
}
# DISCRETE_ACTIONS = None
DISCRETE_ACTIONS = generate_discrete_actions(**discretization_params)


# EXP PARAMS
EXP_PARAMS = {
    'n_runs': 5,
    'n_episodes': 200,
    'len_episode': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']),
    'seed': 22,
    # Env
    'flatten_obs': True,
    'frame_stack': 1,
    # Normalization
    'norm_obs': True,
    'norm_reward': True,
    # Evaluation
    'eval_while_training': True,
    'eval_freq': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']) * 100,
}

# PPO PARAMS
RL_PARAMS: dict[str, Any] = {
    'policy': "MlpPolicy" if EXP_PARAMS['flatten_obs'] else 'MultiInputPolicy',
    'device': 'cpu',
    'learning_rate': linear_scheduler_sb3(0.000739109639891389),  # Default: 3e-4
    # 'learning_rate': linear_schedule(3e-4),  # Default: 3e-4
    'n_steps': 2048,  # Default: 2048
    'batch_size': 512,  # Default: 64
    'n_epochs': 20,  # Default: 10
    # 'gamma': 1-0.003512209278341583,  # Default: 0.99
    'gamma': 0.9040311190204684,  # Default: 0.99
    'gae_lambda': 0.9490833250619164,  # Default: 0.95
    'clip_range': 0.299162760983835,  # Default: 0.2
    'clip_range_vf': linear_scheduler_sb3(0.3),  # Default: None
    # 'clip_range_vf': 0.3,  # Default: None
    'normalize_advantage': True,  # Default: True
    'ent_coef': 0.237273583000828,  # Default: 0.0
    'vf_coef': 0.740664718469597,  # Default: 0.5
    'max_grad_norm': 2.50805793585801,  # Default: 0.5
    'use_sde': False,  # Default: False
    'target_kl': None,

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'large',  # Default: None
        'activation_fn': 'relu',  # Default: tanh
        'ortho_init': False,  # Default: True
        'use_expln': False,  # Default: False
        'squash_output': False,  # Default: False
        'share_features_extractor': True,  # Default: True
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
