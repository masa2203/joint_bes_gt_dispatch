from typing import Callable

import torch


def linear_scheduler_sb3(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule compatible with stable-baselines3.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def linear_scheduler_torch(optim: torch.optim.Optimizer,
                           total_grad_steps: int):
    """
    Linear learning rate scheduler based on torch.optim.lr_scheduler.LambdaLR

    Decays learning rate from init_lr to (near) 0 throughout total_grad_steps.

    :param optim: a torch optimizer object.
    :param total_grad_steps: the number of gradient steps in the experiment.
    :return: torch's LambdaLR scheduler that computes
      current learning rate linearly depending on remaining progress
    """
    def lr_lambda(step):  # calculate factor by which lr is multiplied
        return max((total_grad_steps - step) / total_grad_steps, 1e-2)

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)


class MultiScheduler:
    def __init__(self, schedulers):
        """
        Initializes the MultiScheduler with a list of schedulers.

        :param schedulers: A list of learning rate schedulers.
        """
        self.schedulers = schedulers

    def step(self):
        """
        Calls the step method of each scheduler in the list.
        """
        for scheduler in self.schedulers:
            scheduler.step()
