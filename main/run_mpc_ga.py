import json
import time
import socket

from config import src_dir
from envs.environments import *
from envs.env_params import *
from models.genetic_algorithm import GeneticAlgorithm
from utils.logger import get_env_log_data

start = time.time()

# PLANT PARAMS
ENV = GasTurbineBatteryRenewablesDemandEnv
ENV_KWARGS = cs2_base

env = ENV(**ENV_KWARGS, tracking=True, precision_level="low")

# LOG
CREATE_LOG = True
SAVE_PATH = os.path.join(src_dir, 'log', ENV_KWARGS['env_name'], 'ga', 'run', input('Save in folder: ')) \
    if CREATE_LOG else None

bounds = [env.action_space.low, env.action_space.high]
seed = 22

# EXP PARAMS
ga_params = {
    'len_horizon': 6,  # length of rollouts over which GA optimizes
    'n_iter': 200,  # iterations/generations done per optimization loop
    'n_bits_per_action': 4,  # bits used to encode one action dimension
    'n_pop': 100,  # population size
    'r_cross': 0.9,  # crossover rate, 1 - r_cross will be parent clones
    'r_mut': 0.025,  # mutation rate
    'allow_battery_idle': True,  # change binary encoding for battery to enable idle
}


if SAVE_PATH is not None:
    os.makedirs(SAVE_PATH, exist_ok=True)
    with open(os.path.join(SAVE_PATH, 'inputs.json'), 'w') as f:
        json.dump({
            'GA_PARAMS': ga_params,
            'PLANT_PARAMS': ENV_KWARGS,
            'SEED': seed,
            'PC_NAME': socket.gethostname(),
        }, f)


lst_of_actions = []

done = False
acc_reward = 0
obs = env.reset(seed=seed)

rng = np.random.default_rng(seed)

# Initialize the GA-agent
ga = GeneticAlgorithm(**ga_params, action_bounds=bounds, rng=rng, verbose=0)

# Iterate over time-steps in environment
while done is not True:
    print(env.count)
    # Conduct one optimization with the GA
    best_h_policy, best_h_eval, simulation_done = ga.run(env)
    # print('\tBest policy: ', best_h_policy)

    if simulation_done:
        # If episode ends run entire best sequence
        # print('simulation done')
        for action in best_h_policy:
            next_obs, reward, done, t, _ = env.step(np.array(action, dtype=np.float32))
            lst_of_actions.append(action)
            acc_reward += reward
    else:
        # Run first action of best sequence
        best_action = best_h_policy[0]

        next_obs, reward, done, t, _ = env.step(np.array(best_action, dtype=np.float32))
        lst_of_actions.append(best_action)
        acc_reward += reward

    # if env.count == 10:
    #     break

print('Sum of rewards: ', acc_reward)
# print('Actions: ', lst_of_actions)


if SAVE_PATH is not None:
    log_data = get_env_log_data(env=env, mean_reward=acc_reward, start_time=start)
    with open(os.path.join(SAVE_PATH, f'output.json'), 'w') as f:
        json.dump(log_data, f)

end = time.time()
print()
print(f'Execution time = {end - start}s')
