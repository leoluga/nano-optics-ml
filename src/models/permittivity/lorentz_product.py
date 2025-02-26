import numpy as np

def lorentz_product_model(omega, params):
    """
    Lorentz product model for complex permittivity:
        epsilon(omega) = epsilon_inf * product_m [ (omega_L^2 - omega^2 + i gamma_L omega)
                                                  / (omega_T^2 - omega^2 + i gamma_T omega) ]
    """
    omega = np.asarray(omega, dtype=np.complex128)
    epsilon_inf = params["epsilon_inf"]
    modes = params["modes"]

    product_term = np.ones_like(omega, dtype=np.complex128)
    for mode in modes:
        omega_L = mode["omega_L"]
        gamma_L = mode["gamma_L"]
        omega_T = mode["omega_T"]
        gamma_T = mode["gamma_T"]

        numerator = (omega_L**2 - omega**2) + 1j * gamma_L * omega
        denominator = (omega_T**2 - omega**2) + 1j * gamma_T * omega
        product_term *= (numerator / denominator)

    epsilon = epsilon_inf * product_term
    epsilon = epsilon.real - 1j * epsilon.imag

    return epsilon
