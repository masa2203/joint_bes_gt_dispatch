import random
import json
import gymnasium as gym
import numpy as np
import time

from envs.base_envs import *
from envs.environments import *
from envs.env_params import *
from utils.utilities import set_seeds
from utils.logger import get_env_log_data


# PLANT PARAMS
ENV = GasTurbineBatteryRenewablesDemandEnv
ENV_KWARGS = cs2_base

# LOG
CREATE_LOG = True
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'rb', input('Save in folder: ')) \
    if CREATE_LOG else None

set_seeds(22)

start = time.time()


# print(ENV_KWARGS)
bes = ENV_KWARGS['storage']
env = ENV(**ENV_KWARGS, tracking=True, verbose=False)
max_GT_ISO = 32.6
gt_tolerance = 0.0

random.seed(22)

count = 0
acc_reward = 0
done = False
obs, _ = env.reset()
print(obs)

# Get maximum possible SOC change per timestep for discharge
max_soc_change_discharge = ((ENV_KWARGS['resolution_h'] * bes['max_discharge_rate'] / bes['discharge_eff']) /
                            bes['total_cap'])

while done is not True:
    # RULE-BASED ACTION SELECTION
    diff = (obs['demand'] - obs['re_power']).item()
    if diff <= 0:  # enough re power available, charge BES with surplus
        bes_action = max(diff / bes['max_charge_rate'], -1.0)
        action = np.array([0.0, bes_action])
    # Not enough re power for demand
    else:
        # Deficiency can be met by BES
        if (diff <= (bes['max_discharge_rate'] * bes['discharge_eff'])
                and obs['soc'] >= (bes['min_soc'] + max_soc_change_discharge)):
            bes_action = min(diff / bes['max_discharge_rate'] / bes['discharge_eff'], 1.0)
            action = np.array([0.0, bes_action])
        # Deficiency must be met by GT
        else:
            gt_action = min(diff/max_GT_ISO + gt_tolerance, 1.0)
            action = np.array([gt_action, 0.0])

    obs, reward, done, _, _ = env.step(action)
    acc_reward += reward
    count += 1


# print(obs)
print('Acc. reward: ', acc_reward)
print('Degr. cost: ', sum(env.storage.degr_costs))

log_data = get_env_log_data(env, acc_reward, start)
print('Sum of maintenance cost: ', log_data["maint_cost_sum"])
print('Avg. GT load when on: ', log_data['avg_GT_load_when_on'])
print('GT operating hours: ', log_data['operating_hours_GT'])
print('Number of starts: ', log_data['number_of_starts'])
print('Number of charging: ', log_data['num_charging'])
print('Number of discharging: ', log_data['num_discharging'])
print('Number of oversupplies: ', log_data['num_oversupply'])
print('Avg. oversupply: ', log_data['avg_oversupply'])
print('Number of undersupplies: ', log_data['num_undersupply'])
print('Avg. undersupply: ', log_data['avg_undersupply'])
# print(log_data)

# print('Discharging hours: ', len(list(filter(lambda x: (x > 0), episode_info['bes_energy_flows']))))
# print('Charging hours: ', len(list(filter(lambda x: (x < 0), episode_info['bes_energy_flows']))))
#
end = time.time()
print()
print(f'Execution time = {end - start}s')

if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'output.json'), 'w') as f:
        json.dump(log_data, f)
