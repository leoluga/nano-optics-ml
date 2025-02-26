import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
from numba import njit

from models.fdm_model import calculate_integrand_for_scattered_field

# ----------------------------
# Finite Dipole Model Functions
# ----------------------------
@njit
def calculate_beta(epsilon):
    return (epsilon - 1) / (epsilon + 1)

@njit
def calculate_rp(epsilon_mat, theta_i=np.deg2rad(45)):
    n1 = 1.0  # Air refractive index
    n2 = np.sqrt(epsilon_mat)
    sin_theta_t = (n1/n2) * np.sin(theta_i)
    cos_theta_t = np.sqrt(1 - sin_theta_t**2)
    return (n2*np.cos(theta_i) - n1*cos_theta_t) / (n2*np.cos(theta_i) + n1*cos_theta_t)

@njit
def calculate_f_parameter(g, radius, H, W, L):
    return (g - (radius + 2*H + W)/(2*L)) * (np.log(4*L/(radius + 4*H + 2*W)))/np.log(4*L/radius)

@njit
def calculate_H_t(t, A, Omega, H0):
    return H0 + A*(1 - np.cos(Omega*t))

@njit
def calculate_alpha_eff(beta, f0, f1):
    return 1 + 0.5 * (beta*f0)/(1 - beta*f1)

@njit
def calculate_integrand_for_scattered_field(
    time, epsilon, order_index,
    incidence_angle=np.deg2rad(45),
    complex_factor=0.7*(np.cos(0.06) + 1j*np.sin(0.06)),
    oscillation_amplitude=18e-9,
    tip_radius=20e-9,
    tip_length=300e-9,
    angular_frequency=2*np.pi*250e3,
    initial_height=0.0
):
    beta = calculate_beta(epsilon)
    rp = calculate_rp(epsilon, incidence_angle)
    H = calculate_H_t(time, oscillation_amplitude, angular_frequency, initial_height)
    
    W0 = 1.31 * tip_radius
    W1 = 0.5 * tip_radius
    
    f0 = calculate_f_parameter(complex_factor, tip_radius, H, W0, tip_length)
    f1 = calculate_f_parameter(complex_factor, tip_radius, H, W1, tip_length)
    
    alpha_eff = calculate_alpha_eff(beta, f0, f1)
    return (1 + rp)**2 * alpha_eff * np.exp(-1j*angular_frequency*order_index*time)

# ----------------------------
# Plotting Function
# ----------------------------
def plot_material_results(material, f_cm, epsilon, Sn_fd, Sn_pd, Sn_meas=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot permittivity
    ax1.plot(f_cm, epsilon.real, 'b-', label='Real (ε₁)')
    ax1.plot(f_cm, epsilon.imag, 'r-', label='Imaginary (ε₂)')
    ax1.set_title(f'Complex Permittivity: {material}', fontsize=14)
    ax1.set_ylabel('ε')
    ax1.grid(True)
    ax1.legend()
    
    # # Plot scattered field components
    # ax2.plot(f_cm, Sn_fd.real, 'g-', label='Sn_fd Real (n=2)')
    # ax2.plot(f_cm, Sn_fd.imag, 'g--', label='Sn_fd Imag')
    # ax2.plot(f_cm, Sn_pd.real, 'm-', label='Sn_pd Real (n=3)')
    # ax2.plot(f_cm, Sn_pd.imag, 'm--', label='Sn_pd Imag')
    
    if Sn_meas is not None:
        ax2.plot(f_cm, Sn_meas.real, 'k:', label='Measured Real')
        ax2.plot(f_cm, Sn_meas.imag, 'k-.', label='Measured Imag')
    
    ax2.set_title(f'Scattered Field Components: {material}', fontsize=14)
    ax2.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('Scattered Field (a.u.)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{material}_results.png', dpi=300)
    plt.close()

# ----------------------------
# Main Processing Function
# ----------------------------
def process_materials_with_plots():
    # FDM parameters (converted to meters)
    params = {
        'oscillation_amplitude': 68e-9,    # 68 nm
        'tip_radius': 50e-9,              # 50 nm
        'tip_length': 500e-9,             # 500 nm
        'initial_height': 0.0,            # 0 nm
        'Omega': 2*np.pi*250e3,           # 250 kHz
        'complex_factor': 0.7*(np.cos(0.07) + 1j*np.sin(0.07)),
        'n_values': [2, 3]                # Calculate for n=2 (Sn_fd) and n=3 (Sn_pd)
    }

    materials = ['SiO2', 'STO', 'GaAs', 'LSAT', 'NGO', 'CaF2']
    
    for material in materials:
        print(f"Processing {material}...")
        
        # Load material data (frequency in cm⁻¹)
        eff_data = np.genfromtxt(rf'C:\nano_optics_ml_data\raw\{material}_eff.csv', delimiter=',')
        f_cm = eff_data[:, 4]  # Frequency in cm⁻¹
        epsilon = eff_data[:, 0] + 1j*eff_data[:, 1]
        Sn_meas = eff_data[:, 2] + 1j*eff_data[:, 3]

        # Convert frequency to Hz for calculations
        f_Hz = f_cm * 3e10  # 1 cm⁻¹ = 3e10 Hz

        # Initialize results arrays
        Sn_fd = np.zeros_like(f_Hz, dtype=np.complex128)
        Sn_pd = np.zeros_like(f_Hz, dtype=np.complex128)

        # Calculate scattered fields for each frequency
        for i, freq_hz in enumerate(f_Hz):
            eps = epsilon[i]
            
            # Calculate for n=2 (Sn_fd)
            res_fd, _ = quad_vec(
                lambda t: calculate_integrand_for_scattered_field(
                    t, eps, 2,
                    oscillation_amplitude=params['oscillation_amplitude'],
                    tip_radius=params['tip_radius'],
                    tip_length=params['tip_length'],
                    initial_height=params['initial_height'],
                    angular_frequency=params['Omega'],
                    complex_factor=params['complex_factor']
                ),
                0, 2*np.pi/params['Omega']
            )
            # Calculate for n=3 (Sn_pd)
            res_pd, _ = quad_vec(
                    lambda t: calculate_integrand_for_scattered_field(
                    t, eps, 3,
                    oscillation_amplitude=params['oscillation_amplitude'],
                    tip_radius=params['tip_radius'],
                    tip_length=params['tip_length'],
                    initial_height=params['initial_height'],
                    angular_frequency=params['Omega'],
                    complex_factor=params['complex_factor']
                ),
                0, 2*np.pi/params['Omega']
            )
            # Apply farfield factor
            c = 3e8  # Speed of light (m/s)
            k0 = 2*np.pi*freq_hz/c
            ff_factor = 1j*k0*np.sqrt(eps)
            
            Sn_fd[i] = 0.25 * ff_factor * res_fd
            Sn_pd[i] = 0.25 * ff_factor * res_pd

        # Save data
        output = np.column_stack((
            f_cm,
            np.real(Sn_fd), np.imag(Sn_fd),
            np.real(Sn_pd), np.imag(Sn_pd),
            np.real(epsilon), np.imag(epsilon),
            np.real(Sn_meas), np.imag(Sn_meas)
        ))
        # np.savetxt(f'{material}_results.csv', output, delimiter=',',
        #          header=('Frequency(cm-1), Re(Sn_fd), Im(Sn_fd), Re(Sn_pd), Im(Sn_pd), '
        #                  'Re(epsilon), Im(epsilon), Re(Sn_meas), Im(Sn_meas)'))

        # Generate plots
        plot_material_results(material, f_cm, epsilon, Sn_fd, Sn_pd, Sn_meas)

# ----------------------------
# Run the Script
# ----------------------------
if __name__ == "__main__":
    process_materials_with_plots()