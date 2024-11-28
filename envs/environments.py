from typing import Optional
from gymnasium import spaces

from envs.base_envs import *


class GasTurbineRenewablesDemandEnv(GasTurbineBaseEnv):
    """
    An environment class based on gym, with one or more gas turbines (GT) as dispatchable components.

    This is an environment with additional renewable energy and industrial demand.

    :param env_name: A string that represents the name of the environment.
    :param data_file: A string that represents the path to the data file.
    :param demand_file: A string that represents the path to the demand file.
    :param state_vars: A list that represents the state variables.
    :param gt: A list of dictionaries (keys: gt_class, num_gts, gt_specs) that represents the
                gas turbine configuration of the environment.
    :param grid: A dictionary that represents the grid configuration of the environment.
    :param num_wt: An integer that represents the number of wind turbines.
    :param pv_cap_mw: An integer that represents the photovoltaic (PV) capacity (in MW).
    :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
    :param modeling_period_h: An integer that represents the modeling period (in hours). Default is 8760.
    :param tracking: A boolean that indicates whether to track the variables. Default is True.
    :param verbose: A boolean that indicates whether to print extra information. Default is False.
    :param precision_level: A string that represents the precision level of the variables. Must be either 'low',
    'medium', or 'high'. Default is 'low'.
    """
    def __init__(self,
                 env_name: str,
                 data_file: str,
                 demand_file: str,
                 # demand: int,
                 state_vars: list,
                 gt: list[dict],
                 grid: dict,
                 num_wt: int,
                 pv_cap_mw: int,
                 resolution_h: float = 1.0,
                 modeling_period_h: int = 8760,
                 tracking: bool = True,
                 verbose: bool = False,
                 precision_level: Optional[str] = 'low',
                 ):
        """
        Initializes the GasTurbineRenewableDemandEnv class.

        :param env_name: A string that represents the name of the environment.
        :param data_file: A string that represents the path to the data file.
        :param demand_file: A string that represents the path to the demand file.
        :param state_vars: A list that represents the state variables.
        :param gt: A list of dictionaries (keys: gt_class, num_gts, gt_specs) that represents the
                    gas turbine configuration of the environment.
        :param grid: A dictionary that represents the grid configuration of the environment.
        :param num_wt: An integer that represents the number of wind turbines.
        :param pv_cap_mw: An integer that represents the photovoltaic (PV) capacity (in MW).
        :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
        :param modeling_period_h: An integer that represents the modeling period (in hours). Default is 8760.
        :param tracking: A boolean that indicates whether to track the variables. Default is True.
        :param verbose: A boolean that indicates whether to print extra information. Default is False.
        :param precision_level: A string that represents the precision level of the variables. Must be either 'low',
        'medium', or 'high'. Default is 'low'.
        """
        super().__init__(env_name=env_name,
                         gt=gt,
                         grid=grid,
                         resolution_h=resolution_h,
                         modeling_period_h=modeling_period_h,
                         tracking=tracking,
                         precision_level=precision_level,
                         )

        self.data_file = data_file
        self.demand_file = demand_file
        self.state_vars = state_vars
        self.num_wt = num_wt
        self.pv_cap_mw = pv_cap_mw
        self.verbose = verbose

        # PREPARE MAIN DATA FILE
        self.data = self._init_data(data_file=data_file,
                                    state_vars=state_vars,
                                    demand_file=demand_file,
                                    num_wind_turbines=num_wt,
                                    pv_capacity_mw=pv_cap_mw)
        demand_min, demand_max = self.data['demand'].min(), self.data['demand'].max()  # get min/max of demand
        # self.demand = demand

        # DEFINE OBS SPACE
        self.observation_space = spaces.Dict(
            {
                'demand': spaces.Box(low=demand_min, high=demand_max, shape=(1,)),
                'gt_states': spaces.Box(low=0, high=2, shape=(len(self.gts),)),  # Adjusted for multiple GTs
            }
        )
        for i in self.state_vars:  # add variables from data file to observation space
            low, high = self.data[i].min(), self.data[i].max()
            self.observation_space[i] = spaces.Box(low=low, high=high, shape=(1,))

        # DEFINE ACTION SPACE
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(len(self.gts),), dtype=self.precision['float'])

    def step(self, action: np.array):
        """
        Runs one timestep of the environment's dynamics.

        :param action: A numpy array that represents the action to take.
        :return: A tuple of (observation, reward, terminated, truncated, info).
        """
        action = action.astype(self.precision["float"])
        if self.verbose:
            self._action_checker(action)  # Check for bounds and NaNs, optional since comp. expensive + redundant

        # Check on correct action dimension
        assert len(action) == len(self.gts), "Number of actions must match number of GTs."

        gt_powers = []
        gt_fuel_costs = []
        gt_overhaul_costs = []
        gt_carbon_taxes = []

        # Conduct one step with each GT model
        for i, gt in enumerate(self.gts):
            gt_power, gt_fuel_cost, gt_overhaul_cost, gt_carbon_tax = gt.step(action=action[i],
                                                                              idx=self.count)

            gt_powers.append(gt_power)
            gt_fuel_costs.append(gt_fuel_cost)
            gt_overhaul_costs.append(gt_overhaul_cost)
            gt_carbon_taxes.append(gt_carbon_tax)

        # Compute net power
        net_e_power = sum(gt_powers) + self.obs['re_power'].item()

        # Compute sales of electricity
        e_sales = self.grid.get_grid_interaction(power_flow=net_e_power, demand=self.obs['demand'].item())

        # Subtract fuel price, overhaul_cost, and carbon tax
        reward = e_sales - sum(gt_fuel_costs) - sum(gt_overhaul_costs) - sum(gt_carbon_taxes)

        if np.isnan(reward):
            raise ValueError('Reward is NAN!')

        if self.verbose:
            print('#####################################')
            print(f'Time-step: {self.count}')
            print(f'\tObservation: {self.obs}')
            print(f'\tAction: {action} (GT)')
            print(f'\tGT powers: {[round(gt_power, 3) for gt_power in gt_powers]} | '
                  f'\tTotal power: {round(net_e_power, 3)} | '
                  f'\te-Balance: {round(net_e_power - self.obs["demand"].item(), 3)}')
            print(f'\tFuel cost: {[round(gt_fuel_cost, 3) for gt_fuel_cost in gt_fuel_costs]} | '
                  f'\tMaintenance cost: {[round(gt_overhaul_cost, 3) for gt_overhaul_cost in gt_overhaul_costs]} | '
                  f'\tCarbon tax: {[round(gt_carbon_tax, 3) for gt_carbon_tax in gt_carbon_taxes]}')
            print(f'\tGrid sales: {round(e_sales, 3)} | '
                  f'\tReward: {round(reward, 3)}')
            print()

        if self.tracking:
            self._tracking(
                action=[round(a.item(), 3) for a in action],
                reward=round(reward, 3),
                total_prod_power=round(net_e_power, 3),
                gt_power=[round(gt_power, 3) for gt_power in gt_powers],
                fuel_cost=[round(gt_fuel_cost, 3) for gt_fuel_cost in gt_fuel_costs],
                carbon_tax=[round(gt_carbon_tax, 3) for gt_carbon_tax in gt_carbon_taxes],
                maintenance=[round(gt_overhaul_cost, 3) for gt_overhaul_cost in gt_overhaul_costs],
                e_balance=round(net_e_power - self.obs['demand'].item(), 3),
            )

        self.count += 1
        next_obs, done = self._get_obs()

        return next_obs, reward, done, False, self._get_info()

    def _get_obs(self):
        """
        Returns the observation from the current timestep.

        :return: A tuple of (observation, done).
        """
        # Check termination
        if self.count == len(self.data['Date']):
            self.env_log = self._get_episode_info()
            return self.obs, True

        # Access the data for the current timestep from the dictionary of arrays
        obs = {
            'demand': np.array([self.data['demand'][self.count]], dtype=self.precision["float"]),
            'gt_states': np.array([gt.GT_state for gt in self.gts], dtype=self.precision['int']),
        }
        for i in self.state_vars:
            obs[i] = np.array([self.data[i][self.count]], dtype=self.precision["float"])

        self.obs = obs
        return obs, False


class GasTurbineBatteryRenewablesDemandEnv(GasTurbineAndBatteryBaseEnv):
    """
    An environment class based on gym, with one or more gas turbines (GTs) and a battery as dispatchable components.

    This is an environment with additional renewable energy and industrial demand.

    :param env_name: A string that represents the name of the environment.
    :param data_file: A string that represents the path to the data file.
    :param demand_file: A string that represents the path to the demand file.
    :param state_vars: A list that represents the state variables.
    :param gt: A list of dictionaries (keys: gt_class, num_gts, gt_specs) that represents the
                gas turbine configuration of the environment.
    :param storage: A dictionary that represents the storage configuration of the environment.
    :param grid: A dictionary that represents the grid configuration of the environment.
    :param num_wt: An integer that represents the number of wind turbines.
    :param pv_cap_mw: An integer that represents the photovoltaic (PV) capacity (in MW).
    :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
    :param modeling_period_h: An integer that represents the modeling period (in hours). Default is 8760.
    :param tracking: A boolean that indicates whether to track the variables. Default is True.
    :param verbose: A boolean that indicates whether to print extra information. Default is False.
    :param precision_level: A string that represents the precision level of the variables. Must be either 'low',
    'medium', or 'high'. Default is 'low'.
    """

    def __init__(self,
                 env_name: str,
                 data_file: str,
                 demand_file: str,
                 # demand: int,
                 state_vars: list,
                 gt: list[dict],
                 storage: dict,
                 grid: dict,
                 num_wt: int,
                 pv_cap_mw: int,
                 resolution_h: float = 1.0,
                 modeling_period_h: int = 8760,
                 tracking: bool = True,
                 verbose: bool = False,
                 precision_level: Optional[str] = "low",
                 ):
        """
        Initializes the GasTurbineBatteryRenewablesDemandEnv class.

        :param env_name: A string that represents the name of the environment.
        :param data_file: A string that represents the path to the data file.
        :param demand_file: A string that represents the path to the demand file.
        :param state_vars: A list that represents the state variables.
        :param gt: A list of dictionaries (keys: gt_class, num_gts, gt_specs) that represents the
                    gas turbine configuration of the environment.
        :param storage: A dictionary that represents the storage configuration of the environment.
        :param grid: A dictionary that represents the grid configuration of the environment.
        :param num_wt: An integer that represents the number of wind turbines.
        :param pv_cap_mw: An integer that represents the photovoltaic (PV) capacity (in MW).
        :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
        :param modeling_period_h: An integer that represents the modeling period (in hours). Default is 8760.
        :param tracking: A boolean that indicates whether to track the variables. Default is True.
        :param verbose: A boolean that indicates whether to print extra information. Default is False.
        :param precision_level: A string that represents the precision level of the variables. Must be either 'low',
        'medium', or 'high'. Default is 'low'.
        """
        super().__init__(env_name=env_name,
                         gt=gt,
                         storage=storage,
                         grid=grid,
                         resolution_h=resolution_h,
                         modeling_period_h=modeling_period_h,
                         tracking=tracking,
                         precision_level=precision_level,
                         )

        self.data_file = data_file
        self.demand_file = demand_file
        self.state_vars = state_vars
        self.num_wt = num_wt
        self.pv_cap_mw = pv_cap_mw
        self.verbose = verbose

        # PREPARE MAIN DATA FILE
        self.data = self._init_data(data_file=data_file,
                                    state_vars=state_vars,
                                    demand_file=demand_file,
                                    num_wind_turbines=num_wt,
                                    pv_capacity_mw=pv_cap_mw)
        demand_min, demand_max = self.data['demand'].min(), self.data['demand'].max()  # get min/max of demand
        # self.demand = demand

        # DEFINE OBS SPACE
        self.observation_space = spaces.Dict(
            {
                'demand': spaces.Box(low=demand_min, high=demand_max, shape=(1,)),
                'gt_states': spaces.Box(low=0, high=2, shape=(len(self.gts),)),  # Adjusted for multiple GTs
                'soc': spaces.Box(low=0, high=1, shape=(1,)),
            }
        )
        for i in self.state_vars:  # add variables from data file to observation space
            low, high = self.data[i].min(), self.data[i].max()
            self.observation_space[i] = spaces.Box(low=low, high=high, shape=(1,))

        # DEFINE ACTION SPACE
        # GT action limits
        gt_low = np.zeros(len(self.gts), dtype=self.precision['float'])  # Lower bounds for GTs
        gt_high = np.ones(len(self.gts), dtype=self.precision['float'])  # Upper bounds for GTs

        # Battery action limits
        battery_low = np.array([-1], dtype=self.precision['float'])  # Lower bound for the battery
        battery_high = np.array([1], dtype=self.precision['float'])  # Upper bound for the battery

        # Combine GT and battery action limits
        low = np.concatenate([gt_low, battery_low])  # Combined lower bounds
        high = np.concatenate([gt_high, battery_high])  # Combined upper bounds

        # First dimension(s) = GT(s), last dimension = Battery
        self.action_space = spaces.Box(low=low, high=high, dtype=self.precision['float'])

    def step(self, action: np.array):
        """
        Runs one timestep of the environment's dynamics.

        :param action: A numpy array that represents the action to take.
        :return: A tuple of (observation, reward, terminated, truncated, info).
        """
        # Action = [GT1, GT2, ..., Battery]
        action = action.astype(self.precision["float"])
        if self.verbose:
            self._action_checker(action)  # Check for bounds and NaNs, optional since comp. expensive + redundant

        # Check on correct action dimension, don't comment -> same action could be used for GT and battery
        assert len(action) - 1 == len(self.gts), "Number of actions must match number of GTs plus one."

        gt_powers = []
        gt_fuel_costs = []
        gt_overhaul_costs = []
        gt_carbon_taxes = []

        # Conduct one step with each GT model
        for i, gt in enumerate(self.gts):
            gt_power, gt_fuel_cost, gt_overhaul_cost, gt_carbon_tax = gt.step(action=action[i],
                                                                              idx=self.count)

            gt_powers.append(gt_power)
            gt_fuel_costs.append(gt_fuel_cost)
            gt_overhaul_costs.append(gt_overhaul_cost)
            gt_carbon_taxes.append(gt_carbon_tax)

        # Compute power available for charging battery
        available_power = sum(gt_powers) + self.obs['re_power'].item()

        # Conduct one step with the storage model
        self.storage_power, bat_degr_cost = self.storage.step(action=action[-1], avail_power=available_power)

        # Compute net power
        net_e_power = available_power + self.storage_power

        # Compute sales of electricity
        e_sales = self.grid.get_grid_interaction(power_flow=net_e_power, demand=self.obs['demand'].item())

        # Subtract fuel price, overhaul_cost, carbon tax, and battery degradation cost
        reward = e_sales - sum(gt_fuel_costs) - sum(gt_overhaul_costs) - sum(gt_carbon_taxes) - bat_degr_cost

        if np.isnan(reward):
            raise ValueError('Reward is NAN!')

        if self.verbose:
            print('#####################################')
            print(f'Time-step: {self.count}')
            print(f'\tObservation: {self.obs}')
            print(f'\tAction: {action} (GT, Battery)')
            print(f'\tGT powers: {[round(gt_power, 3) for gt_power in gt_powers]} | '
                  f'\tStorage power flow: {round(self.storage_power, 3)} | '
                  f'\tTotal power: {round(net_e_power, 3)} | '
                  f'\te-Balance: {round(net_e_power - self.obs["demand"].item(), 3)}')
            print(f'\tFuel cost: {[round(gt_fuel_cost, 3) for gt_fuel_cost in gt_fuel_costs]} | '
                  f'\tMaintenance cost: {[round(gt_overhaul_cost, 3) for gt_overhaul_cost in gt_overhaul_costs]} | '
                  f'\tCarbon tax: {[round(gt_carbon_tax, 3) for gt_carbon_tax in gt_carbon_taxes]} | '
                  f'\tDegradation cost: {round(bat_degr_cost, 3)}')
            print(f'\tGrid sales: {round(e_sales, 3)} | '
                  f'\tReward: {round(reward, 3)}')
            print()

        if self.tracking:
            self._tracking(
                action=list(map(lambda x: round(float(x), 3), action.tolist())),  # Convert to list of rounded floats
                reward=round(reward, 3),
                total_prod_power=round(net_e_power, 3),
                gt_power=[round(gt_power, 3) for gt_power in gt_powers],
                fuel_cost=[round(gt_fuel_cost, 3) for gt_fuel_cost in gt_fuel_costs],
                carbon_tax=[round(gt_carbon_tax, 3) for gt_carbon_tax in gt_carbon_taxes],
                maintenance=[round(gt_overhaul_cost, 3) for gt_overhaul_cost in gt_overhaul_costs],
                e_balance=round(net_e_power - self.obs['demand'].item(), 3),
            )

        self.count += 1
        next_obs, done = self._get_obs()

        return next_obs, reward, done, False, self._get_info()

    def _get_obs(self):
        """
        Returns the observation from the current timestep.

        :return: A tuple of (observation, done).
        """
        # Check termination
        if self.count == len(self.data['Date']):
            self.env_log = self._get_episode_info()
            return self.obs, True

        # Access the data for the current timestep from the dictionary of arrays
        obs = {
            'demand': np.array([self.data['demand'][self.count]], dtype=self.precision["float"]),
            'gt_states': np.array([gt.GT_state for gt in self.gts], dtype=self.precision['int']),
            'soc': np.array([self.storage.soc], dtype=self.precision["float"])
        }
        for i in self.state_vars:
            obs[i] = np.array([self.data[i][self.count]], dtype=self.precision["float"])

        self.obs = obs
        return obs, False


