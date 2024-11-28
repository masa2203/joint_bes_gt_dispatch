import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscreteActions(gym.ActionWrapper):
    """
    A gym action wrapper that converts discrete actions to continuous actions.

    :param env: A gym environment.
    :param disc_to_cont: A list that represents the discrete actions.
    """

    def __init__(self, env, disc_to_cont):
        """
        Initializes the DiscreteActions class.

        :param env: A gym environment.
        :param disc_to_cont: A list that represents the discrete actions.
        """
        super().__init__(env)
        self.disc_to_cont = disc_to_cont

        # Check on correct action dimension
        assert env.action_space.shape[0] == len(disc_to_cont[0]), \
            "Number action dimension after discretization must match environment's action dimensions."

        self.action_space = gym.spaces.Discrete(len(disc_to_cont))

    def action(self, act):
        """
        Converts the discrete action to a continuous action.

        :param act: An integer that represents the discrete action.
        :return: A numpy array that represents the continuous action.
        """
        return np.array(self.disc_to_cont[act]).astype(self.env.action_space.dtype)

    def reverse_action(self, action):
        """
        Raises a NotImplementedError.

        :param action: A numpy array that represents the action.
        :raises: NotImplementedError.
        """
        raise NotImplementedError


class RescaleActionSpace(gym.ActionWrapper):
    """
    A gym action wrapper that rescales the action space, taking into account non-zero lower bounds.
    """

    def __init__(self, env):
        """
        Initializes the RescaleActionSpace class.

        :param env: A gym environment.
        """
        super(RescaleActionSpace, self).__init__(env)
        self.orig_action_space = self.env.action_space
        # Calculate the scale and offset for each action dimension based on original bounds
        self.scale = (self.orig_action_space.high - self.orig_action_space.low) / 2.0
        self.offset = (self.orig_action_space.high + self.orig_action_space.low) / 2.0
        # Define the new action space as [-1, 1] for all dimensions
        self.action_space = spaces.Box(low=-1, high=1, shape=self.orig_action_space.shape,
                                       dtype=self.orig_action_space.dtype)

    def action(self, action):
        """
        Rescales the action from [-1,1] to the original action space.

        :param action: A numpy array that represents the action in the [-1,1] space.
        :return: A numpy array that represents the rescaled action in the original action space.
        """
        # Rescale actions to the original space
        rescaled_action = action * self.scale + self.offset
        return rescaled_action

    def reverse_action(self, action):
        """
        Reverses the rescaling of the action from the original action space to [-1,1].

        :param action: A numpy array that represents the action in the original action space.
        :return: A numpy array that represents the reversed action in the [-1,1] space.
        """
        # Reverse scaling from original space to [-1,1]
        reversed_action = (action - self.offset) / self.scale
        return reversed_action


class TemporalInfoAdjustWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, {}


class CombineGTActions(gym.ActionWrapper):
    """
    A gym action wrapper that combines all GT action dimensions into a single dimension.
    The single action value determines how the GTs are fractionally activated.
    Optionally includes a battery action if the environment has 'storage'.
    """

    def __init__(self, env):
        """
        Initializes the CombineGTActionsFractional class.

        :param env: A gym environment.
        """
        super(CombineGTActions, self).__init__(env)
        self.num_gts = len(env.gts)
        self.has_bes = False
        orig_action_space = self.env.action_space

        # Creating a single action dimension for all GTs
        gt_low = np.array([0], dtype=orig_action_space.dtype)
        gt_high = np.array([1], dtype=orig_action_space.dtype)

        # Check if the environment has 'storage' to include battery action dimensions
        if hasattr(env, 'storage'):
            self.has_bes = True
            # Assuming the last action is for the battery, keep it as is.
            battery_low = np.array([orig_action_space.low[-1]], dtype=orig_action_space.dtype)
            battery_high = np.array([orig_action_space.high[-1]], dtype=orig_action_space.dtype)

            # Combine GT and battery action limits
            low = np.concatenate([gt_low, battery_low])  # Combined lower bounds
            high = np.concatenate([gt_high, battery_high])  # Combined upper bounds
        else:
            # No battery, only GT action limits
            low = gt_low
            high = gt_high

            # First dimension = GT, last dimension = Battery (if present)
        self.action_space = spaces.Box(low=low, high=high, dtype=orig_action_space.dtype)

    def action(self, action):
        """
        Distributes the single GT action among all GTs based on the fractional logic described.
        Optionally handles a battery action if included.

        :param action: A numpy array with elements, where the first element is the combined GT action
                       and the second (optional) is the action for the battery.
        :return: A numpy array with actions for each GT and the battery (if present).
        """
        # Check if action includes battery action
        if self.has_bes:
            gt_combined_action, battery_action = action[0], action[1]
        else:
            gt_combined_action = action[0]
            battery_action = None

            # The total GT action is distributed across the GTs
        total_gt_action = gt_combined_action * self.num_gts
        full_capacity_gts = int(np.floor(total_gt_action))
        partial_capacity_gt_action = total_gt_action - full_capacity_gts

        # Create an array of GT actions
        gt_actions = np.zeros(self.num_gts, dtype=self.action_space.dtype)
        gt_actions[:full_capacity_gts] = 1
        if full_capacity_gts < self.num_gts:
            gt_actions[full_capacity_gts] = partial_capacity_gt_action

        if battery_action is not None:
            # Combine GT actions with the battery action
            combined_actions = np.concatenate([gt_actions, np.array([battery_action])])
        else:
            combined_actions = gt_actions

        return combined_actions

    def reverse_action(self, action):
        """
        This method can be implemented if you want to convert back from the environment's action
        space to this wrapper's action space. However, it might not be straightforward due to the
        loss of information when combining GT actions.
        """
        raise NotImplementedError("This wrapper does not support reverse action.")


class PreDefinedDiscreteActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.action_space.shape[0] == 2, 'Action space not fitting to this pre-defined action wrapper!'
        assert len(self.env.gts) == 1, 'Env for this wrapper should have one GT!'

        # Define a new discrete action space with 6 actions
        self.action_space = gym.spaces.Discrete(6)
        self.num_gts = len(self.env.gts)

        self.bes = self.env.unwrapped.storage_dict

        self.avg_gt_max = 30.57931818493151  # in MW
        self.gt_tolerance = 0.00  # Increase GT action on [0,1] scale by this amount to compensate for amb. conditions

    def action(self, action):
        """
        Takes a discrete action and maps it to a continuous action.
        """
        # Map the discrete action to a continuous action
        continuous_action = self.map_action_to_continuous(action)
        # Return the continuous action to be taken in the environment
        return continuous_action

    def map_action_to_continuous(self, action):
        """
        Maps the discrete action to the continuous action space of the environment.
        Assumes that the original continuous action space is a Box with the first
        dimensions corresponding to GTs and the last dimension to the Battery.
        """
        continuous_action = np.zeros(self.num_gts + 1)  # +1 for the battery

        demand = self.env.unwrapped.obs['demand'].item()
        re_power = self.env.unwrapped.obs['re_power'].item()

        diff = demand - re_power  # positive diff => additional energy needed

        # Keep GT and BES off/idle
        if action == 0:
            pass

        # Charge BES with surplus REs (no GT usage)
        elif action == 1:
            if diff >= 0:  # If no surplus REs
                pass  # Leave BES idle
            else:  # Surplus REs
                bes_action = max(diff / self.bes['max_charge_rate'], -1.0)
                continuous_action[-1] = bes_action  # Charge BES

        # Meet deficient power supply with BES (no GT usage)
        elif action == 2:
            if diff <= 0:  # If no deficiency
                pass  # Leave BES idle
            else:
                bes_action = min(diff / self.bes['max_discharge_rate'] / self.bes['discharge_eff'], 1.0)
                continuous_action[-1] = bes_action

        # Meet deficient power supply with GT (no BES usage)
        elif action == 3:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                gt_action = min((diff / self.avg_gt_max) + self.gt_tolerance, 1.0)
                continuous_action[0] = gt_action

        # Meet deficient power supply with BES + GT (Prioritizing BES)
        elif action == 4:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                # Note: This doesn't account for insufficient SOC
                # First, use as much BES power as possible/necessary
                bes_action = min(diff / self.bes['max_discharge_rate'] / self.bes['discharge_eff'], 1.0)
                continuous_action[-1] = bes_action
                # Meet difference from GT
                bes_flow = bes_action * self.bes['max_discharge_rate'] * self.bes['discharge_eff']
                gt_action = min(((diff - bes_flow) / self.avg_gt_max) + self.gt_tolerance, 1.0)
                continuous_action[0] = max(0, gt_action)

        # Use GT for both deficient power supply + BES charging
        elif action == 5:
            if diff <= 0:  # If no deficiency
                pass  # Leave/turn GT off
            else:
                # Note: This doesn't account for full SOC
                gt_action_needed = min((diff / self.avg_gt_max) + self.gt_tolerance, 1.0)
                gt_action = min(gt_action_needed + 0.32, 1.0)  # 0.32 ~= 10 MW
                continuous_action[0] = gt_action

                surplus_gt_power = (gt_action - gt_action_needed) * self.avg_gt_max
                bes_action = max(-surplus_gt_power / self.bes['max_charge_rate'], -1.0)
                continuous_action[-1] = bes_action

        # Correct for GT startup (less power produced due to ramping)
        if continuous_action[0] != 0 and self.env.unwrapped.gts[0].GT_state == 0:
            # Note: Start time is saved in hour-fraction, e.g. 0.25 = 15min.
            start_time = self.env.unwrapped.gts[0].start_reg_h
            if ('t2m' in self.env.unwrapped.obs and
                    self.env.unwrapped.gts[0].start_long_h is not None and
                    self.env.unwrapped.obs['t2m'] < 273.15):
                start_time = self.env.unwrapped.gts[0].start_long_h

            new_gt_actions = min(continuous_action[0] * (1 / (1 - start_time)), 1)
            continuous_action[0] = new_gt_actions

        return continuous_action.astype(self.env.precision['float'])
