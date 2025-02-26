
from .material_library import MATERIAL_LIBRARY, list_available_materials
from .lorentz_sum import lorentz_sum_model
from .lorentz_product import lorentz_product_model
from .meneses_voigt_model import meneses_voigt_model

def compute_permittivity(omega, material_key, temperature=None):
    """
    Compute the complex permittivity for a given material and temperature.

    Parameters
    ----------
    omega : float or array
        Frequency in the units expected by the material's entry 
        (e.g., cm^-1 if the 'units' is 'cm^-1', rad/s if 'units' is 'rad/s').
    material_key : str
        Key in the MATERIAL_LIBRARY dictionary (e.g. "LaAlO3").
    temperature : int, str, or None
        Temperature at which to compute permittivity. If None, 
        it will look for the default or single data set in the library.

    Returns
    -------
    permittivity : complex or array
        The computed complex permittivity at frequency omega.
    """
    if material_key not in MATERIAL_LIBRARY:
        list_available_materials()
        raise ValueError(f"Unknown material key '{material_key}'. Check MATERIAL_LIBRARY.")
    
    entry = MATERIAL_LIBRARY[material_key]
    model = entry["model"]         # "lorentz_sum" or "lorentz_product"
    # units = entry["units"]         # "cm^-1" or "rad/s"
    data_dict = entry["data"]
    
    # If temperature is None, assume there's only one dataset
    # or a default dataset for that material.
    if temperature is None:
        if len(data_dict) == 1:
            # get the sole key
            temperature = list(data_dict.keys())[0]
        else:
            raise ValueError(f"Must specify temperature for material '{material_key}'.")

    if temperature not in data_dict:
        raise ValueError(f"Temperature '{temperature}' not in data for '{material_key}'.")
    
    params = data_dict[temperature]

    # Dispatch to the correct model function
    if model == "lorentz_sum":
        return lorentz_sum_model(omega, params)
    elif model == "lorentz_product":
        return lorentz_product_model(omega, params)
    elif model == "meneses_voigt_model":
        return meneses_voigt_model(omega, params)
    else:
        raise ValueError(f"Unknown model '{model}' in MATERIAL_LIBRARY.")
