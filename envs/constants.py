# Save physical constants and conversion rates
const = {
    'pound_to_kg': 0.45359237,
    'pascal_to_psi': 0.0001450377,
    'ng_lhv': 49.1098,  # MJ/kg
    'ng_GJ_to_cubic_meters': 26.856,  # m^3/GJ
    'meter_to_feet': 3.28084,
    'kilowatt_to_hp': 1.3410220888,  # mechanical/imperial horsepower
}


def kelvin_to_fahrenheit(temp_k: float) -> float:
    """Convert temperature in Kelvin to temperature in degree Fahrenheit"""
    return (temp_k - 273.15) * 9/5 + 32


def celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert temperature in degree Celsius to temperature in degree Fahrenheit"""
    return temp_c * 9/5 + 32


def psi_to_altitude(psi: float) -> float:
    """Convert atmospheric pressure to altitude in feet (required for A05 model)"""
    # Convert PSI to Pascal
    pascals = psi / const['pascal_to_psi']

    # Calculate altitude using the simplified barometric formula
    altitude_meters = 44330.77 * (1 - (pascals / 101325) ** 0.1902632)

    # Convert to feet
    altitude_feet = altitude_meters * const['meter_to_feet']

    return altitude_feet
