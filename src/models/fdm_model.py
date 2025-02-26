#defining the finite dipole model here for later use

from numba import njit
import numpy as np

@njit
def calculate_integrand_for_scattered_field(
    time,
    epsilon,
    order_index,
    incidence_angle=np.deg2rad(45),
    complex_factor=0.7 * (np.cos(0.06) + 1.0j * np.sin(0.06)),
    oscillation_amplitude=68e-9,
    tip_radius=20e-9,
    tip_length=300e-9,
    angular_frequency=2 * np.pi * (250e3),
    initial_height=0.0
):
    """
    Calculate the integrand for the scattered field in a near-field setup.

    This function computes the integrand of the scattered field by incorporating
    reflection coefficients, tip oscillation, and effective polarizability.

    Parameters
    ----------
    time : float or ndarray
        Time value(s) (in seconds) at which to evaluate the integrand.
    dielectric_constant : complex
        Complex dielectric constant (epsilon) of the sample.
    order_index : int or float
        Index used in the exponential factor e^(-i * angular_frequency * order_index * time).
    incidence_angle : float, optional
        Angle of incidence in radians. Defaults to π/4 (45°).
    complex_factor : complex, optional
        Complex dimensionless parameter (G) in the near-field model.
        Defaults to 0.7*(cos(0.06) + i*sin(0.06)).
    oscillation_amplitude : float, optional
        Amplitude of tip oscillation in meters. Defaults to 18e-9.
    tip_radius : float, optional
        Tip radius in meters. Defaults to 20e-9.
    tip_length : float, optional
        Length scale (e.g., near-field region or tip length) in meters. Defaults to 300e-9.
    angular_frequency : float, optional
        Angular frequency of the tip oscillation in rad/s. Defaults to 2π × 250 kHz.
    initial_height : float, optional
        Initial height offset of the tip in meters. Defaults to 0.0.

    Returns
    -------
    complex
        The integrand of the scattered field at the specified time(s).

    Notes
    -----
    - Uses `calculate_beta(dielectric_constant)` to compute beta.
    - Uses `calculate_rp(dielectric_constant, incidence_angle)` to compute reflection coefficient.
    - Uses `calculate_H_t(time, oscillation_amplitude, angular_frequency, initial_height)` 
      to compute the instantaneous tip height H(t).
    - Uses `calculate_f_parameter(...)` to compute f-parameters.
    - Uses `calculate_alpha_eff(beta, f0, f1)` to compute the effective polarizability.
    """

    # Compute beta (ratio of (epsilon-1)/(epsilon+1))
    beta = calculate_beta(epsilon)

    # Fresnel reflection coefficient at the sample interface
    reflection_coeff = calculate_rp(epsilon, incidence_angle)
    
    # Tip position at time t
    tip_height = calculate_H_t(time, oscillation_amplitude, angular_frequency, initial_height)

    # Define widths for near-field geometry
    width_0 = 1.31 * tip_radius
    width_1 = 0.5  * tip_radius

    # Calculate f-parameters for each width
    f0 = calculate_f_parameter(complex_factor, tip_radius, tip_height, width_0, tip_length)
    f1 = calculate_f_parameter(complex_factor, tip_radius, tip_height, width_1, tip_length)

    # Effective polarizability
    alpha_eff = calculate_alpha_eff(beta, f0, f1)

    # Compute the scattered field integrand
    scattered_field_integrand = (
        (1 + reflection_coeff)**2
        * alpha_eff
        * np.exp(-1.0j * angular_frequency * order_index * time)
    )
    return scattered_field_integrand

@njit
def calculate_beta(epsilon):
    return (epsilon-1)/(epsilon+1)

@njit
def calculate_rp(epsilon_mat, theta_i):
    epsilon_air = 1.0 
    
    n1 = np.sqrt(epsilon_air)   
    n2 = np.sqrt(epsilon_mat)      
    # snell law
    sin_theta_t = (n1 / n2) * np.sin(theta_i)
    cos_theta_t = np.sqrt(1 - sin_theta_t**2)  # non absorvent medium

    rp = (n2 * np.cos(theta_i) - n1 * cos_theta_t) / (n2 * np.cos(theta_i) + n1 * cos_theta_t)
    return rp

@njit
def calculate_f_parameter(g, radius, H, W, L):
    f = (g - (radius +2*H + W)/(2*L))*(np.log(4*L/(radius + 4*H +2*W)))/np.log(4*L/radius)
    return f

@njit
def calculate_H_t(t, A, Omega, H0):
    return H0 + A*(1 - np.cos(Omega*t))

@njit
def calculate_alpha_eff(beta, f0, f1):
    return 1 + 0.5 * (beta*f0)/(1 - beta*f1)
