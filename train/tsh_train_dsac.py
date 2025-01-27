import json
import os
import time
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import DiscreteSACPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger, LazyLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
from torch.utils.tensorboard import SummaryWriter

from utils.tsh_make_env import tsh_make_env, tsh_make_vec_env
from utils.scheduler import linear_scheduler_torch, MultiScheduler
from utils.utilities import set_seeds, extract_scalar_from_event
from utils.logger import get_env_log_data


def tsh_train_dsac_agent(
        run: int,
        path: Optional[str],
        exp_params: Dict[str, Any],
        env_id: str,
        env_kwargs: Dict[str, Any],
        rl_params: Dict[str, Any],
        verbose: bool = True,
        discrete_actions: Optional[list] = None,
        logger_type: Optional[List[str]] = None,
) -> float:
    """
    Trains a discrete SAC agent on a given environment and logs training statistics.

    :param run: An integer representing the current run index.
    :param path: An optional string specifying the path to the directory where the logs will be saved.
    :param exp_params: A dictionary containing experiment parameters.
    :param env_id: A string identifier for the environment to be used.
    :param env_kwargs: A dictionary of keyword arguments for environment creation.
    :param rl_params: A dictionary containing reinforcement learning parameters.
    :param verbose: A boolean flag indicating whether to print verbose output.
    :param discrete_actions: An optional list specifying discrete actions, if applicable.
    :param logger_type: An optional list of strings specifying the types of loggers to be used.
    :return: The mean return of the final evaluation episode.
    """
    # Create path for logging
    run_path = os.path.join(path, f'run_{run}') if path is not None else None
    if path is not None:
        os.makedirs(run_path, exist_ok=True)

    # Update seed
    seed = exp_params['seed'] + run
    set_seeds(seed)

    if verbose:
        print('\n|| Run #{} | Seed #{} ||'.format(run, seed))

    start = time.time()

    # Make envs
    env = tsh_make_env(env=env_id,
                       env_kwargs=env_kwargs,
                       discrete_actions=discrete_actions,
                       frame_stack=exp_params['frame_stack'])
    train_env = tsh_make_vec_env(
        env_id=env_id,
        env_kwargs=env_kwargs,
        discrete_actions=discrete_actions,
        frame_stack=exp_params['frame_stack'],
        norm_obs=exp_params['norm_obs'],
        norm_reward=exp_params['norm_reward'],
        tracking=False,
        n_envs=exp_params['n_train_env'],
    )
    test_env = tsh_make_vec_env(
        env_id=env_id,
        env_kwargs=env_kwargs,
        discrete_actions=discrete_actions,
        frame_stack=exp_params['frame_stack'],
        norm_obs=exp_params['norm_obs'],
        norm_reward=False,
        tracking=False,
        n_envs=exp_params['n_test_env'],
    )

    # model & optimizer
    if rl_params['share_net']:  # use same feature extractor for actor and critics
        net = Net(state_shape=env.observation_space.shape,
                  # action_shape=env.action_space.n,  # unnecessary when using actor/critic class
                  action_shape=0,
                  hidden_sizes=rl_params['net_arch'],
                  activation=rl_params['activation_fn'],
                  device=exp_params['device']).to(exp_params['device'])
        nets = [net, net, net]
    else:
        nets = [Net(state_shape=env.observation_space.shape,
                    # action_shape=env.action_space.n,  # unnecessary when using actor/critic class
                    action_shape=0,
                    hidden_sizes=rl_params['net_arch'],
                    activation=rl_params['activation_fn'],
                    device=exp_params['device']).to(exp_params['device']) for i in range(3)]

    actor = Actor(nets[0], env.action_space.n, device=exp_params['device'], softmax_output=rl_params['actor_softmax'])
    actor_optim = torch.optim.Adam(actor.parameters(), lr=rl_params['lr_actor'])
    critic1 = Critic(nets[1], last_size=env.action_space.n, device=exp_params['device'])
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=rl_params['lr_critic'])
    critic2 = Critic(nets[2], last_size=env.action_space.n, device=exp_params['device'])
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=rl_params['lr_critic'])

    scheduler = None
    if rl_params['use_lr_scheduler']:
        total_grad_steps = exp_params['n_epochs'] * exp_params['step_per_epoch'] * rl_params['update_per_step']
        actor_scheduler = linear_scheduler_torch(actor_optim, total_grad_steps)
        critic1_scheduler = linear_scheduler_torch(actor_optim, total_grad_steps)
        critic2_scheduler = linear_scheduler_torch(actor_optim, total_grad_steps)
        scheduler = MultiScheduler([actor_scheduler, critic1_scheduler, critic2_scheduler])

    # Automatic temperature learning
    alpha = ()
    if rl_params['auto_alpha']:
        target_entropy = 0.98 * np.log(np.prod(env.action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=exp_params['device'])
        alpha_optim = torch.optim.Adam([log_alpha], lr=rl_params['lr_actor'])
        alpha = (target_entropy, log_alpha, alpha_optim)

    # DSAC policy
    policy: DiscreteSACPolicy
    policy = DiscreteSACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        action_space=env.action_space,
        tau=rl_params['tau'],
        gamma=rl_params['gamma'],
        alpha=alpha if rl_params['auto_alpha'] else rl_params['alpha'],
        estimation_step=rl_params['estimation_step'],
        lr_scheduler=scheduler
    ).to(exp_params['device'])

    # set up replay buffer
    buffer = VectorReplayBuffer(
        total_size=rl_params['buffer_size'],
        buffer_num=exp_params['n_train_env']
    )

    # collector
    # Note: exploration_noise added in policy (categorical dist.), arg below has no effect
    train_collector = Collector(policy,
                                train_env,
                                buffer,
                                exploration_noise=False)
    test_collector = Collector(policy,
                               test_env,
                               exploration_noise=False)

    # Define logger
    logger = LazyLogger()
    if run_path is not None and logger_type is not None and "tensorboard" in logger_type:
        writer = SummaryWriter(run_path)
        logger = TensorboardLogger(writer)

    # trainer
    train_result = OffpolicyTrainer(
        policy=policy,
        batch_size=rl_params['batch_size'],
        train_collector=train_collector,
        test_collector=test_collector if exp_params['eval_while_training'] else None,
        max_epoch=exp_params['n_epochs'],
        step_per_epoch=exp_params['step_per_epoch'],
        episode_per_test=1,
        step_per_collect=rl_params['train_freq'],  # number of transitions collected before network update
        update_per_step=rl_params['update_per_step'],  # 1 should correspond to -1 for sb3's gradient step param
        # stop_fn=lambda mean_reward: mean_reward >= 195,
        # save_best_fn=save_best_fn,
        test_in_train=True,
        verbose=verbose,
        show_progress=verbose,
        logger=logger
    ).run()

    # train_result.pprint_asdict()  # print stats
    policy.eval()
    test_env.set_env_attr('tracking', True)
    test_env.reset()
    if exp_params['norm_obs']:
        test_env.update_obs_rms = False
    final_eval = test_collector.collect(n_episode=1, render=False)
    if verbose:
        print(f'Final episode reward: {final_eval.returns.mean()}, length: {final_eval.lens.mean()}')
        print(f'Execution time = {train_result.timing.total_time}')

    if run_path is not None:
        log_data = get_env_log_data(env=test_env, mean_reward=final_eval.returns.mean(), start_time=start)
        with open(os.path.join(run_path, 'output.json'), 'w') as f:
            json.dump(log_data, f)

        if isinstance(logger, TensorboardLogger):
            # Try-except statements to handle short debugging runs where keys were not logged.
            try:
                train_rewards = extract_scalar_from_event(event_file_path=run_path,
                                                          scalar_tag='train/returns_stat/mean',
                                                          len_episode=env.len_episode)
                train_rewards.to_csv(os.path.join(run_path, 'train_rewards.csv'), index=False)
            except KeyError:
                print('Train rewards over episodes not logged.')
            try:
                test_rewards = extract_scalar_from_event(event_file_path=run_path,
                                                         scalar_tag='test/returns_stat/mean',
                                                         len_episode=env.len_episode)
                test_rewards.to_csv(os.path.join(run_path, 'eval_rewards.csv'), index=False)
            except KeyError:
                print('Eval rewards over episodes not logged.')

    env.close()
    train_env.close()
    test_env.close()

    return final_eval.returns.mean()
