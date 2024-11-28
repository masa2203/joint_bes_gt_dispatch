import gymnasium as gym
import optuna
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback


class TrialEvalCallback(EvalCallback):
    """
    A callback used for evaluating and reporting a trial.

    :param eval_env: A gym environment used for evaluation.
    :param trial: An optuna.trial object.
    :param n_eval_episodes: An integer that represents the number of evaluation episodes. Default is 1.
    :param eval_freq: An integer that represents the evaluation frequency (in calls of the callback). Default is 10000.
    :param deterministic: A boolean that indicates whether to use a deterministic or stochastic policy. Default is True.
    :param verbose: An integer that represents the verbosity level. Default is 0.
    """

    def __init__(
            self,
            eval_env: gym.Env,
            trial: optuna.Trial,
            n_eval_episodes: int = 1,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
    ):
        """
        Initializes the TrialEvalCallback class.

        :param eval_env: A gym environment used for evaluation.
        :param trial: An optuna.trial object.
        :param n_eval_episodes: An integer that represents the number of evaluation episodes. Default is 1.
        :param eval_freq: An integer that represents the evaluation frequency (in calls of the callback). Default is 10000.
        :param deterministic: A boolean that indicates whether to use a deterministic or stochastic policy. Default is True.
        :param verbose: An integer that represents the verbosity level. Default is 0.
        """

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        """
        Evaluates the policy, sends report to Optuna, and prunes the trial if needed.

        :return: A boolean that indicates whether to continue training or not.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class EvalEpisodeCallback(EvalCallback):
    """
    A callback used for evaluating each episode during training.

    :param eval_env: A gym environment used for evaluation.
    :param n_eval_episodes: An integer that represents the number of evaluation episodes. Default is 1.
    :param eval_freq: An integer that represents the evaluation frequency (in calls of the callback). Default is 10000.
    :param deterministic: A boolean that indicates whether to use a deterministic or stochastic policy. Default is True.
    :param verbose: An integer that represents the verbosity level. Default is 0.
    """

    def __init__(
            self,
            eval_env: gym.Env,
            n_eval_episodes: int = 1,
            eval_freq: int = 10000,
            deterministic: bool = True,
            verbose: int = 0,
    ):
        """
        Initializes the EvalEpisodeCallback class.

        :param eval_env: A gym environment used for evaluation.
        :param n_eval_episodes: An integer that represents the number of evaluation episodes. Default is 1.
        :param eval_freq: An integer that represents the evaluation frequency (in calls of the callback). Default is 10000.
        :param deterministic: A boolean that indicates whether to use a deterministic or stochastic policy. Default is True.
        :param verbose: An integer that represents the verbosity level. Default is 0.
        """
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.eval_idx = 0

    def _on_step(self) -> bool:
        """
        Evaluates the policy.

        :return: A boolean that indicates whether to continue training or not.
        """
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
        return True


class ProgressBarCallback(BaseCallback):
    """
    A base callback that updates a progress bar.

    :param pbar: A tqdm.pbar object.
    """

    def __init__(self, pbar):
        """
        Initializes the ProgressBarCallback class.

        :param pbar: A tqdm.pbar object.
        """
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        """
        Updates the progress bar.

        :return: A boolean that indicates whether to continue training or not.
        """
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)
        return True


# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    """
    A context manager that creates and closes a progress bar.

    :param total_timesteps: An integer that represents the total number of timesteps.
    """
    def __init__(self, total_timesteps):  # init object with total timesteps
        """
        Initializes the ProgressBarManager class.

        :param total_timesteps: An integer that represents the total number of timesteps.
        """
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self):  # create the progress bar and callback, return the callback
        """
        Creates the progress bar and callback, and returns the callback.

        :return: A ProgressBarCallback object.
        """
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        """
        Closes the progress bar.

        :param exc_type: The exception type.
        :param exc_val: The exception value.
        :param exc_tb: The exception traceback.
        """
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
