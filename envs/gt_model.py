import json
from abc import abstractmethod
from typing import Optional, Dict, Union, Tuple
from numpy.random import Generator

import numpy as np

from envs.constants import const

import warnings

warnings.simplefilter('ignore', np.RankWarning)  # for np.polyfit()


class BaseGTModel:
    """
    Base model for gas turbines providing fundamental attributes and methods for cost calculations.

    :param price_natural_gas: A float that represents the price of natural gas per GJ.
    """

    def __init__(self,
                 price_natural_gas: float,
                 ):
        """
        Base model for gas turbines providing fundamental attributes and methods for cost calculations.

        :param price_natural_gas: A float that represents the price of natural gas per GJ.
        """
        self.price_natural_gas = price_natural_gas

        # Precompute the constant conversion factor from pound to kg to GJ and to cubic meters
        self.pound_to_gj = const['pound_to_kg'] * const['ng_lhv'] / 1000
        self.pound_to_m3 = self.pound_to_gj * const['ng_GJ_to_cubic_meters']

    def _get_fuel_cost(self, consumption: float) -> float:
        """
        Computes the cost associated with fuel consumption.

        :param consumption: A float that represents the fuel consumption in pounds.
        :return: A float that represents the cost ($).
        """
        assert consumption >= 0, "Fuel consumption cannot be negative!"

        if consumption == 0:
            return 0

        energy = consumption * self.pound_to_gj  # from pounds to GJ
        costs = energy * self.price_natural_gas  # GJ * $/GJ
        return costs

    def _get_carbon_tax(self,
                        rate: float,
                        style: str = 'energy',
                        fuel_consumption: Optional[float] = None,
                        energy_produced: Optional[float] = None
                        ) -> float:
        """
        Computes the carbon tax due to consumed fuel.

        :param rate: A float that represents the carbon tax rate in $/m3 (fuel) or $/MWh (energy).
        :param style: A string that represents the style of carbon tax calculation. Must be 'energy' or 'fuel'.
        :param fuel_consumption: A float that represents the fuel consumption in pounds if style is 'fuel'.
        :param energy_produced: A float that represents the energy produced in MWh if style is 'energy'.
        :return: A float that represents the carbon tax in $.
        """
        assert style in ['energy', 'fuel'], "Style must be 'energy' or 'fuel'."

        if style == 'fuel':
            volume = fuel_consumption * self.pound_to_m3  # from pounds to m3
            tax = volume * rate  # m3 * $/m3
            return tax

        else:  # style == 'energy':
            tax = energy_produced * rate  # MWh * $/MWh
            return tax


class AdvancedGTModel(BaseGTModel):
    """
    Advanced model for GTs with additional features over the BaseGTModel.

    :param price_natural_gas: A float that represents the price of natural gas per GJ.
    :param fixed_hourly_insp_cost: A float that represents the fixed hourly inspection cost ($).
    :param cost_overhaul: An integer that represents total/lifetime cost of overhaul ($).
    :param total_cycles: An integer that represents the total number of cycles.
    :param total_hours: An integer that represents the total number of hours.
    :param mech_idle_fuel_flow: A float that represents the mechanical idle fuel flow in pph.
    :param start_reg_h: A float that represents the start time for regular starts in hours.
    :param start_long_h: An optional float that represents the start time for long starts in hours.
    :param init_strategy: A string that specifies the initialization strategy. Valid values are 'zero' or 'random'.
    :param operating_threshold: A float that represents the operating threshold below which the GT is considered off.
    :param carbon_tax: An optional dictionary that represents the carbon tax configuration.
    :param use_ramp_rates: Boolean (False is default) if ramping constraints are ignored,
                        otherwise tuple with (max ramp up, max ramp down) in MW/min
    :param resolution_h: A float that represents the resolution in hours for the simulation steps. Default is 1.0.
    :param tracking: An optional boolean that specifies whether to enable tracking.
    """

    # Set the GT-specific upper bounds (for random initialization)
    max_power = None  # in MW
    max_fuel_flow = None  # in pph

    def __init__(self,
                 price_natural_gas: float,
                 fixed_hourly_insp_cost: float,
                 cost_overhaul: int,
                 total_cycles: int,
                 total_hours: int,
                 mech_idle_fuel_flow: float,
                 start_reg_h: float,
                 start_long_h: Optional[float] = None,
                 init_strategy: str = 'zero',
                 operating_threshold: float = 0,
                 carbon_tax: Optional[dict] = None,
                 use_ramp_rates: bool | Tuple[float, float] = False,
                 resolution_h: float = 1.0,
                 tracking: Optional[bool] = True,
                 ):
        """
        Advanced model for GTs with additional features over the BaseGTModel.

        :param price_natural_gas: A float that represents the price of natural gas per GJ.
        :param fixed_hourly_insp_cost: A float that represents the fixed hourly inspection cost.
        :param cost_overhaul: An integer that represents total/lifetime cost of overhaul ($).
        :param total_cycles: An integer that represents the total number of cycles.
        :param total_hours: An integer that represents the total number of hours.
        :param mech_idle_fuel_flow: A float that represents the mechanical idle fuel flow in pph.
        :param start_reg_h: A float that represents the start time for regular starts in hours.
        :param start_long_h: An optional float that represents the start time for long starts in hours.
        :param init_strategy: A string that specifies the initialization strategy. Valid values are 'zero' or 'random'.
        :param operating_threshold: A float that represents the operating threshold below which the GT is considered
                                    off.
        :param carbon_tax: An optional dictionary that represents the carbon tax configuration.
        :param use_ramp_rates: Boolean (False is default) if ramping constraints are ignored,
                                otherwise tuple with (max ramp up, max ramp down) in MW/min
        :param resolution_h: A float that represents the resolution in hours for the simulation steps. Default is 1.0.
        :param tracking: An optional boolean that specifies whether to enable tracking.
        """
        super().__init__(price_natural_gas=price_natural_gas)

        assert init_strategy in ['zero', 'random', 'continuous'], \
            "init_strategy must be 'zero' or 'random', or 'continuous."

        if not (0 <= start_reg_h <= 1):
            warnings.warn("Passed variable 'start_reg_h' is outside [0, 1] hour range.", UserWarning)

        if start_long_h is not None:
            assert start_long_h > start_reg_h, "start_long_h > start_reg_h"
            if not (0 <= start_long_h <= 1):
                warnings.warn("Passed variable 'start_long_h' is outside [0, 1] hour range.", UserWarning)
        else:
            start_long_h = start_reg_h

        self.carbon_tax = carbon_tax
        self.fixed_hourly_insp_cost = fixed_hourly_insp_cost
        self.cost_overhaul = cost_overhaul
        self.total_cycles = total_cycles
        self.total_hours = total_hours
        self.mech_idle_fuel_flow = mech_idle_fuel_flow
        self.start_reg_h = start_reg_h
        self.start_long_h = start_long_h
        self.init_strategy = init_strategy
        self.operating_threshold = operating_threshold
        self.use_ramp_rates = use_ramp_rates
        self.resolution_h = resolution_h
        self.tracking = tracking

        self.n_GT_states = 3  # 0 = off, 1 = started recently, 2 = started a while ago.
        self.hour_cycle_ratio = int(round(self.total_hours / self.total_cycles, 0))
        self.overhaul = 0  # store overhaul cost
        self.start_up_remaining_h = 0  # Tracks remaining duration of start-up in hours

        self.init_GT_state = None
        self.init_GT_h_count = None
        self.GT_state = None
        self.GT_h_count = None
        self.gt_power_old = None
        self.fuel_flow_old = None

        # TRACKERS
        self.count = 0
        self.GT_states = []
        self.GT_h_counts = []
        self.GT_start_up_times_remaining = []

        # PRECOMPUTE
        self.overhaul_cycle_cost = self.cost_overhaul / self.total_cycles
        self.overhaul_timestep_cost = (self.cost_overhaul / self.total_hours) * self.resolution_h
        self.overhaul_timestep_cost_fixed = self.fixed_hourly_insp_cost * self.resolution_h

        if self.use_ramp_rates is not False:  # Precompute max possible ramp up and ramp down
            # MW = MW/min * 60 min/h * h
            self.ramp_up_max = self.use_ramp_rates[0] * 60 * self.resolution_h
            self.ramp_down_max = self.use_ramp_rates[1] * 60 * self.resolution_h

    def _get_gt_start_up_correction(self, output, temp) -> Dict[str, float]:
        """
        Computes the effect of GT start-up time on delivered power and consumed fuel.

        This method should only be called if the GT is started.

        :param output: A dictionary that represents the results of the DLL computation, including 'net_e_power' (kW)
        and 'fuel_flow' (pph).
        :param temp: A float that represents the ambient temperature at the surface in Kelvin.
        :return: A corrected dictionary of GT results.
        """
        if self.GT_state == 0:  # GT starting up
            # Get start time in h, depending on ambient temperature
            start_h = self.start_reg_h
            if temp < 273.15 and self.start_long_h is not None:
                start_h = self.start_long_h
            self.start_up_remaining_h = start_h  # in hours

        if self.start_up_remaining_h - self.resolution_h <= 0:  # Start-up finishes in current time-step
            fraction = self.start_up_remaining_h / self.resolution_h  # fraction of time-step during which GT starts
            # Adjust max possible ramp up using the time left after GT start finished
            # Undone in _get_ramping_correction() after usage
            if self.use_ramp_rates is not False:
                self.ramp_up_max *= (1-fraction)
            output['net_e_power'] *= (1 - fraction)  # reduce electricity sold
            regular_consumption = output['fuelflow'] * (1 - fraction)  # reduced consumption during regular operation
            start_consumption = self.mech_idle_fuel_flow * fraction  # gas consumed during start-up
            output['fuelflow'] = regular_consumption + start_consumption  # total corrected consumption in pph
            self.start_up_remaining_h = 0
        else:  # Start-up takes entire current time-step
            output['net_e_power'] = 0
            output['fuelflow'] = self.mech_idle_fuel_flow
            self.start_up_remaining_h -= self.resolution_h

        return output

    def _get_ramping_correction(self, output):
        """
        Imposes constraints on the ramping of the GT. Power and fuel are corrected using a simple
        proportional adjustment rather than calling the GT model again.

        :param output: A dictionary that represents the results of the DLL computation, including 'net_e_power' (kW)
        and 'fuel_flow' (pph).
        :return: A corrected dictionary of GT results.
        """
        def apply_ramping_correction():
            # Ramp-up
            if output['net_e_power'] > self.gt_power_old:
                # Get the maximum possible GT power (theoretical) from ramping up
                max_possible = self.gt_power_old + self.ramp_up_max
                # Perform correction if desired power exceeds maximum possible considering ramping rates
                if output['net_e_power'] > max_possible:
                    output['fuelflow'] *= max_possible / output['net_e_power']
                    output['net_e_power'] = max_possible
            # Ramp-down
            else:
                # Get the minimum possible GT power (theoretical) from ramping down
                min_possible = self.gt_power_old - self.ramp_down_max
                # Perform correction if desired power exceeds maximum possible considering ramping rates
                if output['net_e_power'] < min_possible:
                    if output['net_e_power'] == 0:
                        output['fuelflow'] = self.fuel_flow_old * (min_possible/self.gt_power_old)
                    else:
                        output['fuelflow'] *= min_possible / output['net_e_power']
                    output['net_e_power'] = min_possible

        # Ramping constraints is needed
        if self.gt_power_old != 0:  # if GT was used during last time step
            apply_ramping_correction()

        elif self.gt_power_old == 0 and self.start_up_remaining_h == 0:  # or has finished start-up
            apply_ramping_correction()
            # self.ramp_up_max is adjusted in _get_gt_start_up_correction to account for GT start-up finishing halfway
            # through a time-step. The line below resets self.ramp_up_max to its regular value.
            self.ramp_up_max = self.use_ramp_rates[0] * 60 * self.resolution_h

        return output

    def _get_gt_overhaul_cost(self, action) -> float:
        """
        Computes the overhaul cost caused by GT usage.

        Three kinds of overhaul cost exist:
        - Fixed: deducted at every hour.
        - Cycle: deducted if new cycle is started (i.e. ramp-up).
        - Operational: deducted if GT operation exceeds time covered by cycle cost.

        :param action: A float that represents the level of GT performance at the current hour (0-1).
        :return: A float that represents the overhaul cost of the current time-step.
        """
        overhaul = 0
        if action != 0:
            if self.GT_state == 0:
                overhaul = self.overhaul_cycle_cost  # add cycle cost
            elif self.GT_state == 2:
                overhaul = self.overhaul_timestep_cost  # add operational costs
        # Add constant inspection cost for GT
        overhaul += self.overhaul_timestep_cost_fixed

        return overhaul

    def _get_next_gt_state(self, action) -> Tuple[int, int]:
        """
        Handles the GT state tracker.

        State 0 = GT is off.
        State 1 = GT is running between 1 and 6h (depends on hours/cycle - ratio).
        State 2 = GT is running more than 6h.

        The hour-cycle-ration is the threshold for switching from cycle cost to hourly cost and computed as
        GT_lifetime_hours / GT_lifetime_cycles.
        The GT state tracker is required to associate maintenance cost to the GT during operation.

        :param action: A float that represents the level of GT performance at the current hour (0-1), clipped in the
        Environment class.
        :return: A tuple that represents the state (0, 1, or 2) and hours of operation since ramp-up.
        """
        # NEXT GT STATE
        if action == 0:
            self.GT_state = 0
            self.GT_h_count = 0
            self.start_up_remaining_h = 0
        else:
            self.GT_h_count += self.resolution_h
            if self.GT_h_count > self.hour_cycle_ratio - 1:
                self.GT_state = 2
            else:
                self.GT_state = 1

        return self.GT_state, self.GT_h_count

    def _tracking(self, GT_state, GT_h_count, start_up_remaining_h):
        """
        Keep track of GT behavior over time.

        :param GT_state: An integer that represents the state of the GT.
        :param GT_h_count: An integer or float that represents the hours of operation since ramp-up.
        :param start_up_remaining_h: A float that represents the remaining start-up time of the GT.
        """
        self.GT_states.append(GT_state)
        self.GT_h_counts.append(GT_h_count)
        self.GT_start_up_times_remaining.append(start_up_remaining_h)

    def _init_state(self, rng):
        """
        Initializes the GT.

        :param rng: A random number generator object.
        """
        # Initialize GT
        if self.init_strategy == 'zero':
            self.init_GT_state = 0
            self.init_GT_h_count = 0
            self.gt_power_old = 0
            self.fuel_flow_old = 0
        elif self.init_strategy == 'random':
            self.init_GT_state = rng.integers(0, self.n_GT_states)  # randomly pick initial GT state
            self.init_GT_h_count = 0
            if self.init_GT_state == 2:
                self.init_GT_h_count = rng.integers(self.hour_cycle_ratio, self.hour_cycle_ratio + 10)
            elif self.init_GT_state == 1:
                self.init_GT_h_count = rng.integers(1, self.hour_cycle_ratio - 1)
            self.gt_power_old = rng.integers(0, self.max_power)
            self.fuel_flow_old = rng.integers(0, self.max_fuel_flow)
        elif self.init_strategy == 'continuous':
            self.init_GT_state = 2
            self.init_GT_h_count = self.hour_cycle_ratio

    def reset(self, rng: Optional[Generator] = None, options=None):
        """
        Resets the counter and GT state.

        :param rng: A random number generator object.
        :param options: An optional string that represents the options for resetting the state.
        """
        rng: Generator = np.random.default_rng(None) if rng is None else rng
        self.start_up_remaining_h = 0  # Tracks remaining duration of start-up in terms of timesteps
        self.count = 0
        self.GT_states = []
        self.GT_h_counts = []
        self.GT_start_up_times_remaining = []

        # Initialize state if not done before or if full reset desired for random initialization
        if self.init_GT_state is None or (options == 'full' and self.init_strategy == 'random'):
            self._init_state(rng=rng)

        self.GT_state = self.init_GT_state
        self.GT_h_count = self.init_GT_h_count

    def partial_reset(self, n):
        """
        Performs a partial reset (required for oracles).

        :param n: An integer that represents the number of previous steps to reset.
        """
        if self.count > n:
            self.count -= n
            self.GT_states = self.GT_states[:-n]
            self.GT_h_counts = self.GT_h_counts[:-n]
            self.GT_start_up_times_remaining = self.GT_start_up_times_remaining[:-n]

            self.GT_state = self.GT_states[-1]
            self.GT_h_count = self.GT_h_counts[-1]
            self.start_up_remaining_h = self.GT_start_up_times_remaining[-1]
        else:
            self.reset()

    def _step_checker(self,
                      action,
                      amb_temp_k: float,
                      amb_pressure_pa: float,
                      rh_pct: float,
                      ):
        """
        Checks the arguments passed to the step-function.

        :param action: A float that represents the level of GT performance at the current hour (0-1), clipped in the
        Environment class.
        :param amb_temp_k: A float that represents the ambient temperature in Kelvin.
        :param amb_pressure_pa: A float that represents the ambient pressure in Pascal.
        :param rh_pct: A float that represents the relative humidity in percent.
        """
        assert self.GT_state is not None, "Must call .reset() before .step() to properly initialize the GT model."

        if action > 1 or action < 0:
            raise ValueError(f'Action {action} outside of bound [0,1]!')

        warnings.warn('Warning: temperature value might not be in Kelvin') if amb_temp_k < 50 else None
        warnings.warn('Warning: pressure value might not be in Pascal') if amb_pressure_pa < 70_000 else None
        warnings.warn('Warning: relative humidity value might not be in percent') if rh_pct < 1 else None

    @abstractmethod
    def step(self, action, amb_temp_k, amb_pressure_pa, rh_pct, idx):
        pass


class A35(AdvancedGTModel):
    """
    Model of SGT-A35 gas turbine.

    :param price_natural_gas: A float that represents the price of natural gas per GJ.
    :param fixed_hourly_insp_cost: A float that represents the fixed hourly inspection cost.
    :param cost_overhaul: An integer that represents total/lifetime cost of overhaul ($).
    :param total_cycles: An integer that represents the total number of cycles.
    :param total_hours: An integer that represents the total number of hours.
    :param mech_idle_fuel_flow: A float that represents the mechanical idle fuel flow in pph.
    :param start_reg_h: A float that represents the start time for regular starts in hours.
    :param start_long_h: A float that represents the start time for long starts in hours.
    :param init_strategy: A string that specifies the initialization strategy. Valid values are 'zero' or 'random'.
    :param operating_threshold: A float that represents the operating threshold.
    :param carbon_tax: A dictionary that represents the carbon tax.
    :param use_ramp_rates: Boolean (False is default) if ramping constraints are ignored,
                        otherwise tuple with (max ramp up, max ramp down) in MW/min
    :param resolution_h: An optional float that represents the resolution in hours. Default is 1.0.
    :param tracking: A boolean that specifies whether to track variables in the step function.
    """

    # Set the GT-specific upper bounds (for random initialization)
    max_power = 36  # in MW
    max_fuel_flow = 14_000  # in pph

    def __init__(self,
                 price_natural_gas: float,
                 fixed_hourly_insp_cost: float,
                 cost_overhaul: int,
                 total_cycles: int,
                 total_hours: int,
                 mech_idle_fuel_flow: float,
                 start_reg_h: float,
                 start_long_h: Optional[float] = None,
                 init_strategy: str = 'zero',
                 operating_threshold: float = 0,
                 carbon_tax: Optional[dict] = None,
                 use_ramp_rates: bool | Tuple[float, float] = False,
                 resolution_h: float = 1.0,
                 tracking: Optional[bool] = True,
                 ):
        """
        Initialize a new GTModel object.

        :param price_natural_gas: A float that represents the price of natural gas per GJ.
        :param fixed_hourly_insp_cost: A float that represents the fixed hourly inspection cost.
        :param cost_overhaul: An integer that represents total/lifetime cost of overhaul ($).
        :param total_cycles: An integer that represents the total number of cycles.
        :param total_hours: An integer that represents the total number of hours.
        :param mech_idle_fuel_flow: A float that represents the mechanical idle fuel flow in pph.
        :param start_reg_h: A float that represents the start time for regular starts in hours.
        :param start_long_h: A float that represents the start time for long starts in hours.
        :param init_strategy: A string that specifies the initialization strategy. Valid values are 'zero' or 'random'.
        :param operating_threshold: A float that represents the operating threshold.
        :param carbon_tax: A dictionary that represents the carbon tax.
        :param use_ramp_rates: Boolean (False is default) if ramping constraints are ignored,
                        otherwise tuple with (max ramp up, max ramp down) in MW/min
        :param resolution_h: An optional float that represents the resolution in hours. Default is 1.0.
        :param tracking: A boolean that specifies whether to track variables in the step function.
        """
        super().__init__(price_natural_gas=price_natural_gas,
                         fixed_hourly_insp_cost=fixed_hourly_insp_cost,
                         cost_overhaul=cost_overhaul,
                         total_cycles=total_cycles,
                         total_hours=total_hours,
                         mech_idle_fuel_flow=mech_idle_fuel_flow,
                         start_reg_h=start_reg_h,
                         start_long_h=start_long_h,
                         init_strategy=init_strategy,
                         operating_threshold=operating_threshold,
                         carbon_tax=carbon_tax,
                         use_ramp_rates=use_ramp_rates,
                         resolution_h=resolution_h,
                         tracking=tracking)

    def step(
            self,
            action,
            idx: int,
            amb_temp_k: float = 283.15,
            amb_pressure_pa: float = 101300.00,
            rh_pct: float = 75.0,
    ) -> Tuple[Dict[str, Union[int, float]], float, float, float]:
        """
        Conducts one step with the GT.

        :param action: A float that represents the level of GT performance at the current hour (0-1), clipped in the
        Environment class.
        :param idx: An integer that counts the time-steps in the current episode (required for look-up table).
        :param amb_temp_k: A float that represents the ambient temperature in Kelvin.
        :param amb_pressure_pa: A float that represents the ambient pressure in Pascal.
        :param rh_pct: A float that represents the relative humidity in percent.
        :return: A tuple that contains the output dictionary, the fuel cost, the overhaul cost, and the carbon tax.
        """
        self._step_checker(action=action, amb_temp_k=amb_temp_k, amb_pressure_pa=amb_pressure_pa, rh_pct=rh_pct)

        if action < self.operating_threshold:
            action = 0

        # MODEL RETURN
        output = self.piecewise_linear_a35(action)

        # Correct for start-up time
        # Check if action implies starting and either GT is off or in the start-up process
        if action != 0 and (self.GT_state == 0 or self.start_up_remaining_h > 0):
            output = self._get_gt_start_up_correction(output, amb_temp_k)

        # Account for ramping constraints if necessary
        # Don't call method if GT is off and will be kept off
        if self.use_ramp_rates is not False and not (self.gt_power_old == 0 and action == 0):
            output = self._get_ramping_correction(output)

        # Get and store overhaul cost
        overhaul_cost = self._get_gt_overhaul_cost(action)

        self.GT_state, self.GT_h_count = self._get_next_gt_state(action)
        # print(f'Overhaul cost = {self.overhaul} | GT State = {self.GT_state} | GT h count = {self.GT_h_count}')

        fuel_cost = self._get_fuel_cost(consumption=output['fuelflow'] * self.resolution_h)

        if self.carbon_tax is not None:
            carbon_tax = self._get_carbon_tax(**self.carbon_tax,
                                              fuel_consumption=output['fuelflow'] * self.resolution_h,
                                              energy_produced=output['net_e_power'] * self.resolution_h)
        else:
            carbon_tax = 0

        if self.tracking:
            self._tracking(self.GT_state, self.GT_h_count, self.start_up_remaining_h)

        self.count += 1
        self.gt_power_old = output['net_e_power']
        self.fuel_flow_old = output['fuelflow']

        return output['net_e_power'], fuel_cost, overhaul_cost, carbon_tax

    @staticmethod
    def piecewise_linear_a35(action):
        gt_power = action * 32.61
        approx1 = np.poly1d((700, 1550))
        approx2 = np.poly1d((360, 2200))
        if 0 < gt_power <= 1:
            fuelflow = approx1(gt_power)
        elif gt_power > 1:
            fuelflow = approx2(gt_power)
        else:
            fuelflow = 0
        return {'net_e_power': gt_power, 'fuelflow': fuelflow}


class A05(AdvancedGTModel):
    """
    Model of SGT-A05 gas turbine.

    :param price_natural_gas: A float that represents the price of natural gas per GJ.
    :param fixed_hourly_insp_cost: A float that represents the fixed hourly inspection cost.
    :param cost_overhaul: An integer that represents total/lifetime cost of overhaul ($).
    :param total_cycles: An integer that represents the total number of cycles.
    :param total_hours: An integer that represents the total number of hours.
    :param mech_idle_fuel_flow: A float that represents the mechanical idle fuel flow in pph.
    :param start_reg_h: A float that represents the start time for regular starts in hours.
    :param start_long_h: A float that represents the start time for long starts in hours. Not applicable for A05.
    :param init_strategy: A string that specifies the initialization strategy. Valid values are 'zero' or 'random'.
    :param operating_threshold: A float that represents the operating threshold.
    :param carbon_tax: A dictionary that represents the carbon tax.
    :param use_ramp_rates: Boolean (False is default) if ramping constraints are ignored,
                        otherwise tuple with (max ramp up, max ramp down) in MW/min
    :param resolution_h: An optional float that represents the resolution in hours. Default is 1.0.
    :param tracking: A boolean that specifies whether to track variables in the step function.
    """

    # Set the GT-specific upper bounds (for random initialization)
    max_power = 6  # in MW
    max_fuel_flow = 3_000  # in pph

    def __init__(self,
                 price_natural_gas: float,
                 fixed_hourly_insp_cost: float,
                 cost_overhaul: int,
                 total_cycles: int,
                 total_hours: int,
                 mech_idle_fuel_flow: float,
                 start_reg_h: float = 1 / 60,
                 start_long_h: Optional[float] = None,
                 init_strategy: str = 'zero',
                 operating_threshold: float = 0,
                 carbon_tax: Optional[dict] = None,
                 use_ramp_rates: bool | Tuple[float, float] = False,
                 resolution_h: float = 1.0,
                 tracking: Optional[bool] = True,
                 ):
        """
        Initialize a new GTModel object.

        :param price_natural_gas: A float that represents the price of natural gas per GJ.
        :param fixed_hourly_insp_cost: A float that represents the fixed hourly inspection cost.
        :param cost_overhaul: An integer that represents total/lifetime cost of overhaul ($).
        :param total_cycles: An integer that represents the total number of cycles.
        :param total_hours: An integer that represents the total number of hours.
        :param mech_idle_fuel_flow: A float that represents the mechanical idle fuel flow in pph.
        :param start_reg_h: A float that represents the start time for regular starts in hours.
        :param start_long_h: A float that represents the start time for long starts in hours. Not applicable for A05.
        :param init_strategy: A string that specifies the initialization strategy. Valid values are 'zero' or 'random'.
        :param operating_threshold: A float that represents the operating threshold.
        :param carbon_tax: A dictionary that represents the carbon tax.
        :param use_ramp_rates: Boolean (False is default) if ramping constraints are ignored,
                                otherwise tuple with (max ramp up, max ramp down) in MW/min
        :param resolution_h: An optional float that represents the resolution in hours. Default is 1.0.
        :param tracking: A boolean that specifies whether to track variables in the step function.
        """
        super().__init__(price_natural_gas=price_natural_gas,
                         fixed_hourly_insp_cost=fixed_hourly_insp_cost,
                         cost_overhaul=cost_overhaul,
                         total_cycles=total_cycles,
                         total_hours=total_hours,
                         mech_idle_fuel_flow=mech_idle_fuel_flow,
                         start_reg_h=start_reg_h,
                         start_long_h=start_long_h,
                         init_strategy=init_strategy,
                         operating_threshold=operating_threshold,
                         carbon_tax=carbon_tax,
                         use_ramp_rates=use_ramp_rates,
                         resolution_h=resolution_h,
                         tracking=tracking)

    def step(
            self,
            action,
            idx: int,
            amb_temp_k: float = 283.15,
            amb_pressure_pa: float = 101300.00,
            rh_pct: float = 75.0,
    ) -> Tuple[Dict[str, Union[int, float]], float, float, float]:
        """
        Conducts one step with the GT.

        :param action: A float that represents the level of GT performance at the current hour (0-1), clipped in the
        Environment class.
        :param idx: An integer that counts the time-steps in the current episode (required for look-up table).
        :param amb_temp_k: A float that represents the ambient temperature in Kelvin.
        :param amb_pressure_pa: A float that represents the ambient pressure in Pascal.
        :param rh_pct: A float that represents the relative humidity in percent.
        :return: A tuple that contains the output dictionary, the fuel cost, the overhaul cost, and the carbon tax.
        """
        self._step_checker(action=action, amb_temp_k=amb_temp_k, amb_pressure_pa=amb_pressure_pa, rh_pct=rh_pct)

        if action < self.operating_threshold:
            action = 0

        # MODEL RETURN
        output = self.piecewise_linear_a05(action)

        # Correct for start-up time
        # Check if action implies starting and either GT is off or in the start-up process
        if action != 0 and (self.GT_state == 0 or self.start_up_remaining_h > 0):
            output = self._get_gt_start_up_correction(output, amb_temp_k)

        # Account for ramping constraints if necessary
        # Don't call method if GT is off and will be kept off
        if self.use_ramp_rates is not False and not (self.gt_power_old == 0 and action == 0):
            output = self._get_ramping_correction(output)

        # Get and store overhaul cost
        overhaul_cost = self._get_gt_overhaul_cost(action)

        self.GT_state, self.GT_h_count = self._get_next_gt_state(action)
        # print(f'Overhaul cost = {self.overhaul} | GT State = {self.GT_state} | GT h count = {self.GT_h_count}')

        fuel_cost = self._get_fuel_cost(consumption=output['fuelflow'] * self.resolution_h)

        if self.carbon_tax is not None:
            carbon_tax = self._get_carbon_tax(**self.carbon_tax,
                                              fuel_consumption=output['fuelflow'] * self.resolution_h,
                                              energy_produced=output['net_e_power'] * self.resolution_h)
        else:
            carbon_tax = 0

        if self.tracking:
            self._tracking(self.GT_state, self.GT_h_count, self.start_up_remaining_h)

        self.count += 1
        self.gt_power_old = output['net_e_power']
        self.fuel_flow_old = output['fuelflow']

        return output['net_e_power'], fuel_cost, overhaul_cost, carbon_tax

    @staticmethod
    def piecewise_linear_a05(action):
        gt_power = action * 6
        approx1 = np.poly1d((360, 800))
        if gt_power > 0.25:
            fuelflow = approx1(gt_power)
        else:
            fuelflow = 0
        return {'net_e_power': gt_power, 'fuelflow': fuelflow}

