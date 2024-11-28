import os
import copy
from config import src_dir


# DEFAULT PARAMETERS FOR A35
a35_default = dict(
    fixed_hourly_insp_cost=0,
    cost_overhaul=33_000_000,  # CAD, over 25 years lifetime value from literature
    total_cycles=26_000,  # value from literature
    total_hours=200_000,  # value from literature
    mech_idle_fuel_flow=1200,  # pph, value from literature
    start_long_h=35 / 60,  # time long start in hours (e.g. 35/60 for 35min)
    start_reg_h=15 / 60,  # time regular start in hours (e.g. 35/60 for 35min)
    init_strategy='zero',  # 'zero' or 'random'
    operating_threshold=0.005,  # if action smaller, GT is kept off
)

# DEFAULT PARAMETERS FOR A05
a05_default = dict(
    fixed_hourly_insp_cost=0,
    cost_overhaul=5_500_000,  # CAD, over 25 years lifetime value from literature
    total_cycles=26_000,  # value from literature
    total_hours=200_000,  # value from literature
    mech_idle_fuel_flow=200,  # pph, value from literature
    start_long_h=None,  # time long start in hours (e.g. 35/60 for 35min)
    start_reg_h=1 / 60,  # time regular start in hours (e.g. 35/60 for 35min)
    init_strategy='zero',  # 'zero' or 'random'
    operating_threshold=0.04,  # if action smaller, GT is kept off
)

# BATTERY DEGRADATION
dod_degr = {
    'type': 'DOD',
    'battery_capex': 300_000,  # CAD/MWh
    'k_p': 1.14,  # Peukert lifetime constant, degradation parameter
    'N_fail_100': 6_000,  # number of cycles at DOD=1 until battery is useless
    'add_cal_age': False,  # adds fixed cost for calendar ageing if True via MAX-operator
    'battery_life': 20,  # expected battery life in years
}

# BES PARAMETERS USED FOR ON1 CASE STUDY
on1_bes = dict(
    total_cap=75,  # MWh
    max_soc=0.9,  # fraction of total capacity
    min_soc=0.1,  # fraction of total capacity
    max_charge_rate=10,  # MW
    max_discharge_rate=10,  # MW
    charge_eff=0.92,  # fraction
    discharge_eff=0.92,  # fraction
    aux_equip_eff=1.0,  # fraction, applied to charge & discharge
    self_discharge=0.0,  # fraction, applied to every step (0 = no self-discharge)
    init_strategy='half',  # 'min', 'max', 'half', or 'random'
    degradation=dod_degr,
)

# GRID PARAMETERS USED FOR ON1 CASE STUDY
on1_grid = dict(
    demand_profile='industry',
    sell_surplus=False,
    buy_deficit=False,
    spread=0,  # CAD per MWh added to the pool price of bought power
    penalty=500,  # CAD per MWh for missing electricity
)

# REPRESENTATIVE DAY
cs1_gt_simple = copy.deepcopy(a35_default)
cs1_gt_simple['start_long_h'] = None
cs1_gt_simple['start_reg_h'] = 0

cs1_bes_simple = copy.deepcopy(on1_bes)
cs1_bes_simple['degradation']['k_p'] = 1

cs1 = {
    'env_name': 'cs1',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'cs1', 'all_data.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'cs1', 'demand.csv'),
    'state_vars':  # list of data columns to serve as state var
        ['re_power', 'sin_h', 'cos_h'],
    'grid': on1_grid,
    'gt': [dict(
        gt_class='A35',
        num_gts=1,
        gt_specs=dict(
            price_natural_gas=7.095,  # CAD per GJ - converted from OEB.ca using 2022 mean (Union South Rate Zone)
            carbon_tax={'rate': 0.0979, 'style': 'fuel'},  # CAD per cubic meter (m^3)
            **cs1_gt_simple
        ),
    )],
    'storage': cs1_bes_simple,
    'num_wt': 12,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 12,  # PV capacity in MW
    'resolution_h': 1.0,  # resolution in hours
    'modeling_period_h': 24,  # modeling period duration in hours
}


# PLANT - ONTARIO - HES WITH FIXED DEMAND - NO GRID - 2022
# Base env
cs2_base = {
    'env_name': 'cs2_base',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'cs2', 'all_data.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'cs2', 'demand.csv'),
    'state_vars':  # list of data columns to serve as state var
        ['re_power', 'sin_h', 'cos_h', 'sin_w', 'cos_w', 'sin_m', 'cos_m', 'workday'],
    'grid': on1_grid,
    'gt': [dict(
        gt_class='A35',
        num_gts=1,
        gt_specs=dict(
            price_natural_gas=7.095,  # CAD per GJ - converted from OEB.ca using 2022 mean (Union South Rate Zone)
            carbon_tax={'rate': 0.0979, 'style': 'fuel'},  # CAD per cubic meter (m^3)
            **a35_default
        ),
    )],
    'storage': on1_bes,
    'num_wt': 12,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 12,  # PV capacity in MW
    'resolution_h': 1.0,  # resolution in hours
    'modeling_period_h': 8760,  # modeling period duration in hours
}

# Version with A35+A05
cs2_2gt = {
    'env_name': 'cs2_2gt',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'cs2', 'all_data.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'cs2', 'demand.csv'),
    'state_vars':  # list of data columns to serve as state var
        ['re_power', 'sin_h', 'cos_h', 'sin_w', 'cos_w', 'sin_m', 'cos_m', 'workday'],
    'grid': on1_grid,
    'gt': [dict(
        gt_class='A35',
        num_gts=1,
        gt_specs=dict(
            price_natural_gas=7.095,  # CAD per GJ - converted from OEB.ca using 2022 mean (Union South Rate Zone)
            carbon_tax={'rate': 0.0979, 'style': 'fuel'},  # CAD per cubic meter (m^3)
            **a35_default
        ),
    ), dict(
        gt_class='A05',
        num_gts=1,
        gt_specs=dict(
            price_natural_gas=7.095,  # CAD per GJ - converted from OEB.ca using 2022 mean (Union South Rate Zone)
            carbon_tax={'rate': 0.0979, 'style': 'fuel'},  # CAD per cubic meter (m^3)
            **a05_default
        ), )],
    'num_wt': 12,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 12,  # PV capacity in MW
    'resolution_h': 1.0,  # resolution in hours
    'modeling_period_h': 8760,  # modeling period duration in hours
}

# Version with A35+A05+BES
cs2_2gt_bes = {
    'env_name': 'cs2_2gt_bes',  # used for saving path
    'data_file': os.path.join(src_dir, 'data', 'cs2', 'all_data.csv'),
    'demand_file': os.path.join(src_dir, 'data', 'cs2', 'demand.csv'),
    'state_vars':  # list of data columns to serve as state var
        ['re_power', 't2m', 'sp', 'rh', 'sin_h', 'cos_h', 'sin_w', 'cos_w', 'sin_m', 'cos_m', 'workday'],
    'grid': on1_grid,
    'gt': [dict(
        gt_class='A35',
        num_gts=1,
        gt_specs=dict(
            price_natural_gas=7.095,  # CAD per GJ - converted from OEB.ca using 2022 mean (Union South Rate Zone)
            carbon_tax={'rate': 0.0979, 'style': 'fuel'},  # CAD per cubic meter (m^3)
            **a35_default
        ),
    ), dict(
        gt_class='A05',
        num_gts=1,
        gt_specs=dict(
            price_natural_gas=7.095,  # CAD per GJ - converted from OEB.ca using 2022 mean (Union South Rate Zone)
            carbon_tax={'rate': 0.0979, 'style': 'fuel'},  # CAD per cubic meter (m^3)
            **a05_default
        ), )],
    'storage': on1_bes,
    'num_wt': 12,  # number of wind turbines, 1 WT = 2.3 MW
    'pv_cap_mw': 12,  # PV capacity in MW
    'resolution_h': 1.0,  # resolution in hours
    'modeling_period_h': 8760,  # modeling period duration in hours
}