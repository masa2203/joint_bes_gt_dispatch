import itertools
import os
import json
import socket
from typing import Any

from stable_baselines3.common.noise import NormalActionNoise

from config import src_dir
from envs.environments import *
from envs.env_params import *
from utils.net_design import activation_fn_dict, net_arch_dict
from utils.scheduler import linear_scheduler_sb3
from utils.make_env import make_env
from utils.logger import sb3_create_stats_file
from train.train import train_rl_agent


# PLANT PARAMS
ENV = GasTurbineBatteryRenewablesDemandEnv
ENV_KWARGS = cs2_base

# LOG
CREATE_LOG = True
VERBOSE = 0
LOGGER_TYPE = ["csv"]
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'td3', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

ACTION_DIM = make_env(ENV, ENV_KWARGS).action_space.shape[-1]

# EXP PARAMS
EXP_PARAMS = {
    'n_runs': 5,
    'n_episodes': 200,
    'len_episode': int(ENV_KWARGS['modeling_period_h'] / ENV_KWARGS['resolution_h']),
    'seed': 22,
    # Env
    'flatten_obs': True,
    'frame_stack': 4,  # CS1 best: 1
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
    # 'device': 'cpu',
    # 'learning_rate': 0.001,  # Default: 1e-3
    'learning_rate': linear_scheduler_sb3(0.00118),  # Default: 1e-3, CS1 best:
    'buffer_size': 1_000_000,  # Default: 1M, CS1 best: 1M
    'learning_starts': 10_000,  # Default: 100, CS1 best: 1_000
    'batch_size': 256,  # Default: 256, CS1 best: 256
    'tau': 0.00065,  # Default: 0.005, CS1 best: 0.0053
    'gamma': 0.9885,  # Default: 0.99, CS1 best: 0.9983
    'train_freq': 2,  # Default: 1, CS1 best: 10
    'gradient_steps': 5,  # Default: 1, CS1 best: -1
    'action_noise': NormalActionNoise(mean=np.zeros(ACTION_DIM), sigma=0.1 * np.ones(ACTION_DIM)),  # Default: None, CS1 best: same
    'policy_delay': 2,  # Default: 2, CS1 best: 2
    'target_policy_noise': 0.3,  # Default: 0.2, CS1 best: 0.2
    'target_noise_clip': 0.4,  # Default: 0.5, CS1 best: 0.4

    'policy_kwargs': {
        # Defaults reported for MultiInputPolicy
        'net_arch': 'ddpg',  # Default: None, CS1 best:
        'activation_fn': 'relu',  # Default: 'relu', CS1 best: 'leaky_relu'
        'n_critics': 2,  # Default: 2, CS1 best: 2
    }
}

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'EXP_PARAMS': EXP_PARAMS,
            'RL_PARAMS': str(RL_PARAMS),
            'PLANT_PARAMS': ENV_KWARGS,
            'PC_NAME': socket.gethostname(),
        }, f)


RL_PARAMS['policy_kwargs']['net_arch'] = net_arch_dict[RL_PARAMS['policy_kwargs']['net_arch']]
RL_PARAMS['policy_kwargs']['activation_fn'] = activation_fn_dict[RL_PARAMS['policy_kwargs']['activation_fn']]

for run in range(EXP_PARAMS['n_runs']):
    train_rl_agent(
        agent='td3',
        run=run,
        path=SAVE_PATH,
        exp_params=EXP_PARAMS,
        env_id=ENV,
        env_kwargs=ENV_KWARGS,
        discrete_actions=None,
        rl_params=RL_PARAMS,
        verbose=VERBOSE,
        logger_type=LOGGER_TYPE,
    )

# GET STATISTICS FROM MULTIPLE RUNS
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
    if EXP_PARAMS['eval_while_training']:
        sb3_create_stats_file(path=SAVE_PATH, exp_params=EXP_PARAMS, style='eval')
    sb3_create_stats_file(path=SAVE_PATH, exp_params=EXP_PARAMS, style='train')
