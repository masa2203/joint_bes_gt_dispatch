from typing import Optional
import numpy as np


class GridModel:
    """
    Class to model the grid and its interaction with the plant.

    :param demand_profile: A string that specifies the type of demand. Valid values are 'industry', 'grid', or None
    (no demand).
    :param sell_surplus: A boolean that indicates whether surplus power can be sold to the grid (only with demand).
    :param buy_deficit: A boolean that indicates whether deficient power can be bought from grid (only with demand).
    :param spread: A float that represents the amount added to the price of bought power ($).
    :param penalty: A float that represents the penalty for deficient power (only with demand, $/MWh).
    :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
    """

    def __init__(self,
                 demand_profile: Optional[str] = None,  # if None no demand is used, else specify demand type
                 sell_surplus: Optional[bool] = False,  # if set to True, surplus power is sold (only with demand)
                 buy_deficit: Optional[bool] = True,  # if True, deficit power is bought from grid (only with demand)
                 spread: float = 0.0,  # in $/MWh, added to price of bought power
                 penalty: Optional[float] = None,  # in $/MWh, added to deficient power (only with demand)
                 resolution_h: float = 1.0,
                 ):
        """
        Initialize a new GridModel object.

        :param demand_profile: A string that specifies the type of demand. Valid values are 'industry', 'grid', or None
        (no demand).
        :param sell_surplus: A boolean that indicates whether surplus power can be sold to the grid (only with demand).
        :param buy_deficit: A boolean that indicates whether deficient power can be bought from grid (only with demand).
        :param spread: A float that represents the amount added to the price of bought power ($).
        :param penalty: A float that represents the penalty for deficient power (only with demand, $/MWh).
        :param resolution_h: A float that represents the resolution of the simulation (in hours). Default is 1.0.
        """
        assert demand_profile in [None, 'industry', 'grid'], "Invalid demand_profile."
        assert spread >= 0, "Spread must be greater than or equal to 0."
        assert penalty is None or penalty >= 0, "Penalty must be greater than or equal to 0."

        self.demand_profile = demand_profile
        self.sell_surplus = sell_surplus
        self.buy_deficit = buy_deficit
        self.spread = spread
        self.penalty = penalty
        self.resolution_h = resolution_h

    def get_grid_interaction(self,
                             power_flow: float,
                             pool_price: Optional[float] = None,
                             demand: Optional[float] = None) -> float:
        """
        Models grid response to produced electricity.

        :param power_flow: A float that represents the electricity flow (MW).
        :param pool_price: A float that represents the pool price ($/MWh).
        :param demand: A float that represents the power demand (MW).
        :return: A float that represents the cash flow ($).
        """
        # Free interaction with the grid, no quantity limits
        if self.demand_profile is None:
            return self._get_free_interaction(power_flow, pool_price)
        # GRID-style demand
        # NOTE: Selling surplus and purchasing deficit makes no sense here
        elif self.demand_profile == 'grid':
            return self._get_utility_grid_interaction(power_flow, pool_price, demand)
        # INDUSTRY demand (i.e. power supplied to a company)
        elif self.demand_profile == 'industry':
            return self._get_industry_interaction(power_flow, pool_price, demand)

    def _get_free_interaction(self, power_flow: float, pool_price: float) -> float:
        """
        Calculate the cash flow for free interaction with the grid.

        :param power_flow: A float that represents the electricity flow (negative = purchase power), (MW).
        :param pool_price: A float that represents the pool price ($/MWh).
        :return: A float that represents the cash flow ($).
        """
        if power_flow < 0:  # Purchase power (e.g. energy arbitrage with a battery)
            return power_flow * self.resolution_h * (pool_price + self.spread)
        else:  # sell power
            return power_flow * self.resolution_h * pool_price

    def _get_utility_grid_interaction(self, power_flow: float, pool_price: float, demand: float) -> float:
        """
        Calculate the cash flow for grid-style demand.

        :param power_flow: A float that represents the electricity flow (MW).
        :param pool_price: A float that represents the pool price ($/MWh).
        :param demand: A float that represents the power demand (MW).
        :return: A float that represents the cash flow ($).
        """
        assert isinstance(demand, (int, float)) and not isinstance(demand, bool) and demand >= 0, \
            'Demand must be zero or a positive!'

        diff = power_flow - demand

        if diff >= 0:  # more power than demanded
            return demand * self.resolution_h * pool_price
        else:  # less power than demanded
            if power_flow >= 0:  # still some power sold to grid
                cash_flow = power_flow * self.resolution_h * pool_price
            else:  # power bought from grid
                cash_flow = power_flow * self.resolution_h * (pool_price + self.spread)
            if self.penalty is not None:  # add penalty for missing power
                cash_flow -= abs(diff * self.resolution_h * self.penalty)
            return cash_flow

    def _get_industry_interaction(self, power_flow: float, pool_price: Optional[float], demand: float) -> float:
        """
        Calculate the cash flow for industry-style demand.

        :param power_flow: A float that represents the electricity flow (MW).
        :param pool_price: A float that represents the pool price ($/MWh).
        :param demand: A float that represents the power demand (MW).
        :return: A float that represents the cash flow ($).
        """
        assert (isinstance(demand, (int, float, np.floating, np.integer))
                and not isinstance(demand, bool) and demand >= 0), 'Demand must be zero or a positive!'

        diff = (power_flow - demand) * self.resolution_h

        if diff >= 0:  # more power than demanded
            if self.sell_surplus:  # selling possible
                return diff * pool_price
            else:
                return 0
        else:  # less power than demanded
            cash_flow = 0
            if self.buy_deficit:  # purchase missing power
                cash_flow -= abs(diff * (pool_price + self.spread))
            if self.penalty:  # add penalty for missing power
                cash_flow -= abs(diff * self.penalty)

            return cash_flow
