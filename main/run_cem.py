import os
import json
import socket
import torch

from config import src_dir
from envs.environments import *
from envs.env_params import *
from utils.logger import cem_create_stats_file
from models.cem import train_cem

# PLANT PARAMS
ENV = GasTurbineBatteryRenewablesDemandEnv
ENV_KWARGS = cs2_base

# LOG
CREATE_LOG = True
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'cem', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

# EXP_PARAMS
EXP_PARAMS = {
    'n_runs': 5,
    'n_iterations': 50,
    'seed': 22,
    # Env
    'flatten_obs': True,  # requires changes in cem.py if set to False
    'frame_stack': None,
    # Normalization
    'norm_obs': True,
    'norm_reward': False,
    # Hps
    'gamma': 1,
    'size_hl': 128,
    'activation_fn': 'relu',
    'out_activ': None,  # or 'sigmoid'
    'layer_norm': True,
    'pop_size': 40,
    'elite_frac': 0.1,
    'init_sigma': 0.2,  # controls gaussian noise added, initial value, decays linearly to final_sigma
    'final_sigma': 0.0001,  # controls gaussian noise added, final value
    'max_t': ENV_KWARGS['modeling_period_h'],  # maximum number of timesteps a set of weights is evaluated for
}

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_num_threads(10)

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'EXP_PARAMS': EXP_PARAMS,
            'PLANT_PARAMS': ENV_KWARGS,
            'PC_NAME': socket.gethostname(),
        }, f)

for run in range(EXP_PARAMS['n_runs']):
    train_cem(
        run=run,
        path=SAVE_PATH,
        exp_params=EXP_PARAMS,
        env_id=ENV,
        env_kwargs=ENV_KWARGS,
        device=device
    )

# GET STATISTICS FROM MULTIPLE RUNS AND CONVERT STEPS TO EPISODES
if CREATE_LOG and EXP_PARAMS['n_runs'] > 1:
    cem_create_stats_file(path=SAVE_PATH, len_episode=int(ENV_KWARGS['modeling_period_h']/ENV_KWARGS['resolution_h']))
