import numpy as np

def lorentz_sum_model(omega, params):
    """
    Lorentz sum model for complex permittivity:
        epsilon(omega) = epsilon_inf + sum_j [ (S_j * omega_j^2) / (omega_j^2 - omega^2 - i * gamma_multiplicator * gamma_j * omega) ]

    If an oscillator has 'A' instead of 'S', we use that (Nunley2016 style).
    """
    omega = np.asarray(omega, dtype=np.complex128)
    epsilon_inf = params["epsilon_inf"]
    oscillators = params["oscillators"]
    try:
        gamma_multiplicator = params["gamma_multiplicator"]
    except KeyError:
        gamma_multiplicator = 1

    epsilon = epsilon_inf * np.ones_like(omega, dtype=np.complex128)
    for osc in oscillators:
        # Either S or A (some references call it S, others call it A)
        strength = osc.get("S", osc.get("A", 0.0))
        omega_j = osc["omega"]
        gamma_j = osc["gamma"]

        numerator = strength * (omega_j**2)
        denominator = (omega_j**2 - omega**2) - 1j * gamma_multiplicator * gamma_j * omega
        epsilon += numerator / denominator

    return epsilon
