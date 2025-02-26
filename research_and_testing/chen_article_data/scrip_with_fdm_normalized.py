import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec

from models.fdm_model import calculate_integrand_for_scattered_field
# ----------------------------
# Finite Dipole Model Functions
# ----------------------------
# [Keep all the FDM functions from previous implementation]

# ----------------------------
# Plotting Function with Normalization
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
    
    # Plot normalized scattered field components
    ax2.plot(f_cm, Sn_fd.real, 'g-', label='Sn_fd Real (n=2)')
    ax2.plot(f_cm, Sn_fd.imag, 'g--', label='Sn_fd Imag')
    ax2.plot(f_cm, Sn_pd.real, 'm-', label='Sn_pd Real (n=3)')
    ax2.plot(f_cm, Sn_pd.imag, 'm--', label='Sn_pd Imag')
    
    if Sn_meas is not None:
        Sn_meas_normalized = Sn_meas / np.abs(au_Sn_fd)  # Normalize measurements too if needed
        ax2.plot(f_cm, Sn_meas_normalized.real, 'k:', label='Measured Real (Norm)')
        ax2.plot(f_cm, Sn_meas_normalized.imag, 'k-.', label='Measured Imag (Norm)')
    
    ax2.set_title(f'Normalized Scattered Field: {material}', fontsize=14)
    ax2.set_xlabel('Frequency (cm⁻¹)', fontsize=12)
    ax2.set_ylabel('Sn/Sn_Au')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{material}_normalized.png', dpi=300)
    plt.close()


def nk_to_eps(n,k):
        eps1 = n**2-k**2
        eps2 = 2*n*k
        return eps1, eps2

# ----------------------------
# Main Processing Function with Normalization
# ----------------------------
def process_materials_with_normalization():
    params = {
        'oscillation_amplitude': 68e-9,
        'tip_radius': 50e-9,
        'tip_length': 500e-9,
        'initial_height': 0.0,
        'angular_frequency': 2*np.pi*250e3,
        'complex_factor': 0.7*(np.cos(0.07) + 1j*np.sin(0.07))
    }

    # Process gold first
    materials = ['Au', 'SiO2', 'STO', 'GaAs', 'LSAT', 'NGO', 'CaF2']
    global au_Sn_fd, au_Sn_pd  # Store gold values for normalization
    

    for idx, material in enumerate(materials):
        # Store gold values for normalization
        if material == 'Au':
            au_Sn_fd = Sn_fd.copy()
            au_Sn_pd = Sn_pd.copy()
            # Save gold results without normalization
            output = np.column_stack((f_cm, Sn_fd.real, Sn_fd.imag, 
                                    Sn_pd.real, Sn_pd.imag,
                                    epsilon.real, epsilon.imag))
            np.savetxt(f'{material}_raw.csv', output, delimiter=',',
                     header='Frequency(cm⁻¹), Re(Sn_fd), Im(Sn_fd), Re(Sn_pd), Im(Sn_pd), Re(epsilon), Im(epsilon)')
            continue  # Skip normalization for gold
        
        print(f"Processing {material}...")
        
        # Load material data
        eff_data = np.genfromtxt(f'{material}_eff.csv', delimiter=',')
        f_cm = eff_data[:, 4]
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
                0, 2*np.pi/params['angular_frequency']
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
                0, 2*np.pi/params['angular_frequency']
            )
            
            # Apply farfield factor
            c = 3e8  # Speed of light (m/s)
            k0 = 2*np.pi*freq_hz/c
            ff_factor = 1j*k0*np.sqrt(eps)
            
            Sn_fd[i] = 0.25 * ff_factor * res_fd
            Sn_pd[i] = 0.25 * ff_factor * res_pd


        # Normalize with gold values
        Sn_fd_normalized = Sn_fd / np.abs(au_Sn_fd)
        Sn_pd_normalized = Sn_pd / np.abs(au_Sn_pd)

        # Save normalized data
        output = np.column_stack((f_cm, 
                                Sn_fd_normalized.real, Sn_fd_normalized.imag,
                                Sn_pd_normalized.real, Sn_pd_normalized.imag,
                                epsilon.real, epsilon.imag,
                                Sn_meas.real, Sn_meas.imag))
        
        np.savetxt(f'{material}_normalized.csv', output, delimiter=',',
                 header=('Frequency(cm⁻¹), Re(Sn_fd_norm), Im(Sn_fd_norm), Re(Sn_pd_norm), Im(Sn_pd_norm), '
                        'Re(epsilon), Im(epsilon), Re(Sn_meas), Im(Sn_meas)'))

        # Generate plots with normalized values
        plot_material_results(material, f_cm, epsilon, 
                            Sn_fd_normalized, Sn_pd_normalized, Sn_meas)

# ----------------------------
# Run the Script
# ----------------------------
if __name__ == "__main__":
    process_materials_with_normalization()