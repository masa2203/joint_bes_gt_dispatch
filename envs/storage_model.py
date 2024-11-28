import warnings
import numpy as np
from typing import Optional, Tuple, Union, List
from numpy.random import Generator


class BESS:
    """
    A class to model a battery energy storage system (BESS).

    :param total_cap: A float or integer that represents the total capacity of the storage (MWh).
    :param max_soc: A float that represents the maximum allowed state of charge of total capacity (fraction).
    :param min_soc: A float that represents the minimum allowed state of charge of total capacity (fraction).
    :param max_charge_rate: A float or integer that represents the maximum possible rate of charge (MW).
    :param max_discharge_rate: A float or integer that represents the maximum possible rate of discharge (MW) NEGATIVE!
    :param charge_eff: A float that represents the charging efficiency (fraction).
    :param discharge_eff: A float that represents the discharging efficiency (fraction).
    :param aux_equip_eff: An optional float that represents the efficiency of auxiliary equipment applied to charge
    and discharge cycles (fraction). Default is 1.0.
    :param self_discharge: An optional float that represents the discharge of storage, applied at every time step
    (fraction). Default is 0.0.
    :param init_strategy: An optional string that determines the initial state of charge. Must be 'min', 'max',
    'half', or 'random'. Default is 'half'.
    :param resolution_h: An optional float that represents the resolution in hours. Default is 1.0.
    :param degradation: An optional boolean that represents whether ageing is considered. Default is None.
    :param tracking: An optional boolean that represents whether to track variables in step function. Default is True.
    """
    def __init__(self,
                 total_cap: Union[int, float],
                 max_soc: float,
                 min_soc: float,
                 max_charge_rate: Union[int, float],
                 max_discharge_rate: Union[int, float],
                 charge_eff: float,
                 discharge_eff: float,
                 aux_equip_eff: float = 1.0,
                 self_discharge: float = 0.0,
                 init_strategy: str = 'half',
                 resolution_h: float = 1.0,
                 degradation=None,
                 tracking: Optional[bool] = True,
                 ):
        """
        Constructs a new BESS object.

        :param total_cap: A float or integer that represents the total capacity of the storage (MWh).
        :param max_soc: A float that represents the maximum allowed state of charge of total capacity (fraction).
        :param min_soc: A float that represents the minimum allowed state of charge of total capacity (fraction).
        :param max_charge_rate: A float or integer that represents the maximum possible rate of charge (MW).
        :param max_discharge_rate: A float or integer that represents the maximum possible rate of discharge (MW).
        :param charge_eff: A float that represents the charging efficiency (fraction).
        :param discharge_eff: A float that represents the discharging efficiency (fraction).
        :param aux_equip_eff: An optional float that represents the efficiency of auxiliary equipment applied to charge
        and discharge cycles (fraction). Default is 1.0.
        :param self_discharge: An optional float that represents the discharge of storage, applied at every time step
        (fraction). Default is 0.0.
        :param init_strategy: An optional string that determines the initial state of charge. Must be 'min', 'max',
        'half', or 'random'. Default is 'half'.
        :param resolution_h: An optional float that represents the resolution in hours. Default is 1.0.
        :param degradation: An optional boolean that represents whether ageing is considered. Default is None.
        :param tracking: An optional boolean that represents whether to track variables in step function. Default is
        True.
        """
        assert init_strategy in ['min', 'max', 'half', 'random'], \
            "init_strategy must be 'min', 'max', 'half', or 'random'."
        assert 0 <= charge_eff <= 1, "Charge efficiency must be between 0 and 1."
        assert 0 <= discharge_eff <= 1, "Discharge efficiency must be between 0 and 1."
        assert 0 <= aux_equip_eff <= 1, "Auxiliary equipment efficiency must be between 0 and 1."
        assert 0 <= self_discharge <= 1, "Self-discharge must be between 0 and 1."
        assert 0 <= min_soc <= 1, "Min SOC must be between 0 and 1."
        assert 0 <= max_soc <= 1, "Max SOC must be between 0 and 1."
        assert degradation is None, "BESS class cannot handle battery degradation."
        assert max_soc > min_soc, "Max SOC must be greater than min SOC."
        assert max_charge_rate > 0, "max_charge_rate must be greater than zero."
        assert max_discharge_rate > 0, "max_discharge_rate must be greater than zero."

        # ARGUMENTS
        self.total_cap = total_cap  # MWh
        self.max_soc = max_soc  # fraction of total capacity
        self.min_soc = min_soc  # fraction of total capacity
        self.max_charge_rate = max_charge_rate  # MW
        self.max_discharge_rate = max_discharge_rate  # MW
        self.charge_eff = charge_eff  # fraction
        self.discharge_eff = discharge_eff  # fraction
        self.aux_equip_eff = aux_equip_eff  # fraction, applied to charge & discharge
        self.self_discharge_factor = 1 - self_discharge  # fraction, applied to every step
        self.init_strategy = init_strategy
        self.resolution_h = resolution_h  # resolution in hours
        self.degradation = degradation
        self.tracking = tracking

        self.init_soc = None
        self.soc = None

        # TRACKERS
        self.count = 0
        self.socs = []  # tracks SOCs
        self.power_flows = []  # tracks power flows from plant view
        self.degr_costs = []  # tracks degradation cost

    def step(self, action: float, avail_power: float) -> Tuple[float, float]:
        """
        Conducts one step with the storage.

        :param action: A float that represents the range(-1,1) used to charge or discharge the storage, negative sign
        for charge.
        :param avail_power: A float that represents the max. power available for charging (MW).
        :return: A tuple of two floats that represent the power flow from plant view and the degradation cost.
        """
        assert avail_power >= 0, "Available power must be non-negative!"
        assert -1 <= action <= 1, f"Action out of bounds [-1, 1]. Action passed: {action}"
        assert self.soc is not None, "Must call .reset() before .step() to properly initialize the storage."

        energy_flow = self._soc_change(action, avail_power)  # energy flow from plant perspective
        power_flow = energy_flow / self.resolution_h  # power flow from plant perspective
        degr_cost = 0

        if self.tracking:
            self._tracking(power_flow, degr_cost)

        self.count += 1

        return power_flow, degr_cost

    def _soc_change(self, action: float, avail_power: float) -> float:
        """
        Changes SOC according to action, i.e., either charging, discharging, or keeping unchanged.

        :param action: A float that represents the range(-1,1) used to charge or discharge the storage, negative sign
        for charge.
        :param avail_power: A float that represents the max. power available for charging (MW).
        :return: A float that represents the energy flow (negative value for charging / positive value for discharging).
        """
        # if action > 1 or action < -1:
        #     warnings.warn('WARNING: Storage charge rates outside interval (-1,1) !')

        # Self discharge of storage
        self.soc *= self.self_discharge_factor

        # Convert action into charge/discharge rate and get energy flow (variable for actual energy exchange)
        if action > 0:  # discharge, battery gives energy
            rate = self.max_discharge_rate * action
            energy_flow = self._discharge(rate)
        elif action < 0:  # charge, battery takes energy
            rate = self.max_charge_rate * action
            energy_flow = self._charge(rate, avail_power)
        else:
            energy_flow = 0

        return energy_flow  # negative value for charging / positive value for discharging

    def _charge(self, rate: float, avail_power: float) -> float:
        """
        Charges the storage.

        Notes:
        - Charges with desired rate unless less power is available
        - Charges with desired rate unless  max SOC is reached
        - Efficiencies are reflected through lower SOC values after charging

        :param rate: A float that represents the charge rate in MW.
        :param avail_power: A float that represents the available power for charging in MW.
        :return: A float that represents the energy used to charge the storage.
        """
        # assert rate <= 0, "Charge rate must not be positive!"  # Redundant with assert in init.
        rate = -rate  # Charge rate is negative -> make positive
        # Pick min of desired charge rate and available power, multiply by time to obtain energy
        available_energy = min(rate, avail_power) * self.resolution_h

        # Update SOC
        old_soc = self.soc
        new_soc = self.soc + (available_energy * self.charge_eff * self.aux_equip_eff) / self.total_cap
        self.soc = min(new_soc, self.max_soc)  # avoids overcharging

        # Compute effective energy flow
        energy_flow = (old_soc - self.soc) * self.total_cap / (self.charge_eff * self.aux_equip_eff)

        return energy_flow

    def _discharge(self, rate: float) -> float:
        """
        Discharges the storage.

        Notes:
        - Charges with desired rate unless min SOC is reached
        - Efficiencies are reflected through less energy flow compared to SOC reduction

        :param rate: A float that represents the discharge rate in MW (negative).
        :return: A float that represents the energy drawn from storage (as available for use).
        """
        # assert rate >= 0, "Charge rate must not be negative!"  # Redundant with assert in init.

        # Update SOC, no efficiencies here
        old_soc = self.soc
        new_soc = self.soc - (rate * self.resolution_h) / self.total_cap
        self.soc = max(new_soc, self.min_soc)

        # Compute effective energy flow, factor in losses during discharge
        energy_flow = (old_soc - self.soc) * self.discharge_eff * self.aux_equip_eff * self.total_cap

        return energy_flow

    def _tracking(self, power_flow: float, degr_cost: float):
        """
        Keeps track of storage behavior over time.

        :param power_flow: A float that represents the energy flow.
        :param degr_cost: A float that represents the degradation cost.
        """
        self.socs.append(self.soc)
        self.power_flows.append(round(power_flow, 2))
        self.degr_costs.append(round(degr_cost, 2))

    def _init_state(self, rng: np.random.Generator):
        """
        Initializes the state of charge (SOC) of the storage.

        :param rng: An instance of np.random.Generator used for random number generation.
        """
        if self.init_strategy == 'min':
            self.init_soc = self.min_soc
        elif self.init_strategy == 'max':
            self.init_soc = self.max_soc
        elif self.init_strategy == 'half':
            self.init_soc = 0.5
        elif self.init_strategy == 'random':
            self.init_soc = rng.integers(0, self.total_cap) / self.total_cap

    def reset(self, options: Optional[dict] = None, rng: Optional[Generator] = None):
        """
        Resets the storage.

        :param options: An optional dictionary that determines the reset options. Can be used to change tracking
                        behavior (key: 'tracking') and to reset storage to random SOC (key: 'full_reset').
        :param rng: An optional instance of np.random.Generator used for random number generation. Default is None.
        """
        # Define default options and overwrite if options are passed
        default_options = {'tracking': self.tracking, 'full_reset': False}

        if options is not None:
            default_options.update(options)

        # Update tracking parameter
        self.tracking = default_options['tracking']

        rng: Generator = np.random.default_rng(None) if rng is None else rng
        self.count = 0
        self.socs = []
        self.power_flows = []
        self.degr_costs = []

        # Initialize state if not done before or if full reset desired for random initialization
        if self.init_soc is None or (default_options['full_reset'] and self.init_strategy == 'random'):
            self._init_state(rng=rng)

        self.soc = self.init_soc

    def partial_reset(self, n):
        """
        Resets the storage partially.

        :param n: An integer that represents the number of steps to reset.
        """
        if self.count > n:
            self.count -= n
            self.socs = self.socs[:-n]
            self.power_flows = self.power_flows[:-n]
            self.degr_costs = self.degr_costs[:-n]

            self.soc = self.socs[-1]
        else:
            self.reset()


class DODDegradingBESS(BESS):
    """
    A subclass of BESS that adds battery ageing cost based on DOD changes.

    :param total_cap: A float or integer that represents the total capacity of the storage (MWh).
    :param max_soc: A float that represents the maximum allowed state of charge of total capacity (fraction).
    :param min_soc: A float that represents the minimum allowed state of charge of total capacity (fraction).
    :param max_charge_rate: A float or integer that represents the maximum possible rate of charge (MW).
    :param max_discharge_rate: A float or integer that represents the maximum possible rate of discharge (MW).
    :param charge_eff: A float that represents the charging efficiency (fraction).
    :param discharge_eff: A float that represents the discharging efficiency (fraction).
    :param degradation: A dictionary that contains the degradation parameters. The dictionary must include the
    following keys: 'battery_capex' (battery capital expenditure in USD/kWh), 'k_p' (a constant that determines
    the rate of degradation), 'N_fail_100' (the number of cycles at which the battery fails with 100% depth of
    discharge), 'add_cal_age' (a boolean that determines whether calendar ageing is considered), and 'battery_life'
    (the expected battery life in years).
    :param aux_equip_eff: An optional float that represents the efficiency of auxiliary equipment applied to charge
    and discharge cycles (fraction). Default is 1.0.
    :param self_discharge: An optional float that represents the discharge of storage, applied at every time step
    (fraction). Default is 0.0.
    :param init_strategy: An optional string that determines the initial state of charge. Must be 'min', 'max',
    'half', or 'random'. Default is 'half'.
    :param resolution_h: An optional float that represents the resolution in hours. Default is 1.0.
    :param tracking: An optional boolean that represents whether to track variables in step function. Default is True.
    """
    def __init__(self,
                 total_cap: Union[int,float],
                 max_soc: float,
                 min_soc: float,
                 max_charge_rate: Union[int,float],
                 max_discharge_rate: Union[int,float],
                 charge_eff: float,
                 discharge_eff: float,
                 degradation: dict,
                 aux_equip_eff: float = 1.0,
                 self_discharge: float = 0.0,
                 init_strategy: str = 'half',
                 resolution_h: float = 1.0,
                 tracking: bool = True,
                 ):
        super().__init__(total_cap=total_cap,
                         max_soc=max_soc,
                         min_soc=min_soc,
                         max_charge_rate=max_charge_rate,
                         max_discharge_rate=max_discharge_rate,
                         charge_eff=charge_eff,
                         discharge_eff=discharge_eff,
                         aux_equip_eff=aux_equip_eff,
                         self_discharge=self_discharge,
                         init_strategy=init_strategy,
                         resolution_h=resolution_h,
                         tracking=tracking)

        self.investment_cost = self.total_cap * degradation['battery_capex']
        self.k_p = degradation['k_p']
        self.N_fail_100 = degradation['N_fail_100']
        self.add_cal_age = degradation['add_cal_age']
        self.battery_life = degradation['battery_life']

    def step(self, action: float, avail_power: float) -> Tuple[float, float]:
        """
        Conducts one step with the storage.

        :param action: A float that represents the range(-1,1) used to charge or discharge the storage, negative sign
        for charge.
        :param avail_power: A float that represents the max. power available for charging (MW).
        :return: A tuple of two floats that represent the power flow from plant view and the degradation cost.
        """
        assert avail_power >= 0, "Available power must be non-negative!"
        assert -1 <= action <= 1, f"Action out of bounds [-1, 1]. Action passed: {action}"
        assert self.soc is not None, "Must call .reset() before .step() to properly initialize the storage."

        soc_old = self.soc
        energy_flow = self._soc_change(action, avail_power)  # energy flow from plant perspective
        power_flow = energy_flow / self.resolution_h  # power flow from plant perspective
        degr_cost = self._get_degr_cost(self.soc, soc_old)

        if self.tracking:
            self._tracking(power_flow, degr_cost)

        self.count += 1

        return power_flow, degr_cost

    def _get_degr_cost(self, soc: float, soc_old: float) -> float:
        """
        Calculates the degradation cost based on the change in state of charge (SOC) of the battery.

        :param soc: A float that represents the current state of charge (SOC) of the battery.
        :param soc_old: A float that represents the previous state of charge (SOC) of the battery.
        :return: A float that represents the degradation cost.
        """
        denominator = 2 * self.N_fail_100
        numerator = np.abs((1 - soc) ** self.k_p - (1 - soc_old) ** self.k_p)
        degr_cost = self.investment_cost * numerator / denominator

        # Add calendar age if desired
        if self.add_cal_age:
            degr_cost = max(degr_cost, self.investment_cost / self.battery_life / 8760)
        return degr_cost

