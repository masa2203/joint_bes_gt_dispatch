import os
import time
import math
import json
from typing import Optional, List, Dict, Tuple, Any

import gymnasium
import numpy as np

import torch
import torch.nn as nn

from utils.make_env import make_env
from utils.net_design import activation_fn_dict
from utils.logger import get_env_log_data


class CrossEntropyMethod:
    """
    Implements the Cross-Entropy Method (CEM).

    :param env: The environment instance from gymnasium.
    :param size_hl: An integer representing the size of the hidden layer in the policy network.
    :param activation_fn: A string specifying the activation function to use in the policy network.
    :param out_activ: The output activation function for the policy network, or None if not used.
    :param layer_norm: A boolean indicating whether to use layer normalization in the policy network. Default is False.
    :param device: The torch.device on which tensors will be allocated. Default is CPU.
    """

    def __init__(
            self,
            env: gymnasium.Env,
            size_hl: int,
            activation_fn: str,
            out_activ: Optional[str] = None,
            layer_norm: bool = False,
            device: torch.device = torch.device('cpu')
    ) -> None:
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        # self.obs_dim = len(env.observation_space)
        self.action_dim = env.action_space.shape[0]
        self.min_val = env.action_space.low
        self.max_val = env.action_space.high

        self.size_hl = size_hl
        self.activation_fn = activation_fn_dict[activation_fn]()
        self.out_activ = None if out_activ is None else activation_fn_dict[out_activ]()
        self.layer_norm = layer_norm

        self.step_count = 0
        self.device = device

        self.actor = Actor(self.obs_dim, self.action_dim, self.size_hl, self.activation_fn,
                           self.out_activ, self.layer_norm).to(self.device)

    def get_weights_dim(self) -> int:
        """
        Calculates the total dimension of all weights in the policy network.

        :return: The total number of weights as an integer.
        """
        # required for sampling, returns overall number of weights
        return (self.obs_dim + 1) * self.size_hl + (self.size_hl + 1) * self.action_dim

    def set_weights(self, weights: np.ndarray) -> None:
        """
        Sets the weights of the policy network to the provided values.

        :param weights: A numpy.ndarray containing the flattened array of weights to be set in the policy network.
        """
        # separate the weights for each layer
        fc1_end = (self.obs_dim * self.size_hl) + self.size_hl
        fc1_W = torch.from_numpy(weights[:self.obs_dim * self.size_hl].reshape(self.obs_dim, self.size_hl))
        fc1_b = torch.from_numpy(weights[self.obs_dim * self.size_hl:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end + (self.size_hl * self.action_dim)].reshape(self.size_hl,
                                                                                                     self.action_dim))
        fc2_b = torch.from_numpy(weights[fc1_end + (self.size_hl * self.action_dim):])
        # set the weights for each layer
        self.actor.fc1.weight.data.copy_(fc1_W.view_as(self.actor.fc1.weight.data))
        self.actor.fc1.bias.data.copy_(fc1_b.view_as(self.actor.fc1.bias.data))
        self.actor.fc2.weight.data.copy_(fc2_W.view_as(self.actor.fc2.weight.data))
        self.actor.fc2.bias.data.copy_(fc2_b.view_as(self.actor.fc2.bias.data))

    def evaluate(self, weights: np.ndarray, gamma: float = 1.0, max_t: int = 8760) -> float:
        """
        Evaluates the policy defined by the given weights in the environment.

        :param weights: A numpy.ndarray containing the flattened array of weights for the policy network.
        :param gamma: A float representing the discount factor for future rewards. Default is 1.0.
        :param max_t: An integer representing the maximum number of timesteps for each episode. Default is 8760.
        :return: The total accumulated reward as a float.
        """
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            # state = np.fromiter(state.values(), dtype=np.float32)
            # state = torch.from_numpy(state).float().to(self.device)
            state = torch.from_numpy(state.flatten()).float().to(self.device)
            action = np.array([self.actor(state).cpu().detach().numpy()])
            action = np.clip(action, self.min_val, self.max_val)
            state, reward, done, _ = self.env.step(action)
            self.step_count += 1
            episode_return += reward * math.pow(gamma, t)

        return episode_return.item()


class Actor(nn.Module):
    """
    Defines the policy network architecture for the CEM agent. Currently supports a single hidden layer.

    :param obs_dim: An integer representing the dimension of the observation space.
    :param action_dim: An integer representing the dimension of the action space.
    :param size_hl: An integer representing the size of the hidden layer.
    :param activation: The activation function to use in the network.
    :param out_activ: The output activation function for the network, or None if not used.
    :param layernorm: A boolean indicating whether to use layer normalization. Default is False.
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            size_hl: int,
            activation: nn.Module,
            out_activ: Optional[nn.Module],
            layernorm: bool) -> None:
        super(Actor, self).__init__()
        self.action_dim = action_dim
        # Input to hidden layer
        self.fc1 = nn.Linear(obs_dim, size_hl)
        layers = [self.fc1]
        if layernorm:
            layers.append(nn.LayerNorm(size_hl))
        layers.append(activation)
        # Hidden to output layer
        self.fc2 = nn.Linear(size_hl, action_dim)
        layers.append(self.fc2)
        if out_activ is not None:
            layers.append(out_activ)

        # Wrap with sequential module
        self.structure = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the policy network.

        :param x: A torch.Tensor representing the input observations.
        :return: The output actions as a torch.Tensor.
        """
        x = self.structure(x)
        # return torch.squeeze(x)
        return x


def resize_proportional(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Rescales an array to a new size while maintaining the proportional distribution of its values.

    :param arr: A numpy.ndarray representing the original array to be resized.
    :param n: An integer representing the target size of the resized array.
    :return: A numpy.ndarray of size `n` that maintains the proportional distribution of `arr`'s values.
    """
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(arr)), arr)


def train_cem(run: int,
              path: Optional[str],
              exp_params: Dict[str, Any],
              env_id: str,
              env_kwargs: Dict[str, Any],
              device: torch.device
              ) -> None:
    """
    Conducts the training process for the Cross-Entropy Method (CEM) agent within a specified environment.

    :param run: An integer representing the run ID for differentiation between multiple training runs.
    :param path: An optional string specifying the directory path where training logs and outputs should be saved.
                 If `None`, logging is disabled.
    :param exp_params: A dictionary containing key experiment parameters such as 'seed', 'pop_size', 'elite_frac',
                       'sigma', 'n_iterations', 'gamma', 'max_t', 'size_hl', 'activation_fn', 'out_activ', 'layer_norm'.
    :param env_id: A string representing the ID of the environment to be used for training.
    :param env_kwargs: A dictionary containing key-value pairs for environment configuration.
    :param device: A torch.device indicating where tensors should be allocated (CPU or GPU).
    """
    run_path = os.path.join(path, f'run_{run}') if path is not None else None

    if path is not None:
        os.makedirs(run_path, exist_ok=True)

    start = time.time()

    seed = exp_params['seed'] + run
    print('|| Run #{} | Seed #{} ||'.format(run, seed))

    # CREATE ENVIRONMENT
    env = make_env(env=env_id,
                   env_kwargs=env_kwargs,
                   tracking=True,
                   path=run_path,
                   flatten_obs=exp_params['flatten_obs'],
                   frame_stack=exp_params['frame_stack'],
                   norm_obs=exp_params['norm_obs'],
                   norm_reward=exp_params['norm_reward'],
                   gamma=1,
                   )

    # DEFINE AGENT
    agent = CrossEntropyMethod(env=env,
                               size_hl=exp_params['size_hl'],
                               activation_fn=exp_params['activation_fn'],
                               out_activ=exp_params['out_activ'],
                               layer_norm=exp_params['layer_norm'],
                               device=device)

    init_sigma = exp_params['init_sigma']
    final_sigma = exp_params['final_sigma']
    rewards_vs_steps = []
    n_elite = int(exp_params['pop_size'] * exp_params['elite_frac'])

    best_weight = init_sigma * np.random.randn(agent.get_weights_dim())
    reward = agent.evaluate(best_weight, gamma=1.0, max_t=exp_params['max_t'])
    rewards_vs_steps.append((reward, agent.step_count))
    print('Initialization\t Env. Steps: {}\tReward: {:.2f}'.format(agent.step_count, reward))

    # Iteration or number of generations
    for iteration in range(exp_params['n_iterations']):
        sigma = init_sigma - ((init_sigma - final_sigma) * (iteration / exp_params['n_iterations']))
        weights_pop = [best_weight +
                       (sigma *
                        np.random.randn(agent.get_weights_dim())) for i in range(exp_params['pop_size'])]
        rewards = np.array([agent.evaluate(weights, exp_params['gamma'],
                                           exp_params['max_t']) for weights in weights_pop])

        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        best_weight = np.array(elite_weights).mean(axis=0)
        reward = agent.evaluate(best_weight, gamma=1.0, max_t=exp_params['max_t'])

        rewards_vs_steps.append((reward, agent.step_count))
        if iteration % 1 == 0:
            print('Iteration {}\t Env. Steps: {}\tReward: {:.2f}'.format(iteration, agent.step_count, reward))

    # Save log data
    if run_path is not None:
        log_data = get_env_log_data(env=env, mean_reward=reward, start_time=start)
        log_data['rewards_vs_steps'] = rewards_vs_steps
        # Convert to rewards vs episodes
        episodes = int(rewards_vs_steps[-1][1] / env_kwargs['modeling_period_h'])
        rewards = [i[0] for i in rewards_vs_steps]
        log_data['rewards_vs_episodes'] = resize_proportional(rewards, episodes).tolist()

        with open(os.path.join(run_path, 'output.json'), 'w') as f:
            json.dump(log_data, f)

    env.close()

    print()
    print(f'Execution time = {time.time() - start}s')
