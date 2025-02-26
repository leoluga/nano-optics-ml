import numpy as np
import matplotlib.pyplot as plt

def nk_to_eps(n, k):
    """
    Convert refractive index (n) and extinction coefficient (k)
    to complex dielectric permittivity: eps = (n^2 - k^2) + i(2*n*k)
    """
    eps1 = n**2 - k**2
    eps2 = 2 * n * k
    return eps1, eps2

def load_eff_data(filename, use_nk=False, nk_filename=None):
    """
    Load effective data from CSV.
    Expected columns in the eff file:
      Column 0: Real part of epsilon (or other effective property)
      Column 1: Imaginary part of epsilon
      Column 2: Real part of the measured far-field signal (Sn)
      Column 3: Imaginary part of the measured far-field signal (Sn)
      Column 4: Frequency
    If use_nk is True, epsilon is computed via nk_to_eps by interpolating
    the n and k values from the provided nk file.
    """
    data = np.genfromtxt(filename, delimiter=',')
    f = data[:, 4]
    Sn = data[:, 2] + 1j * data[:, 3]
    
    if use_nk and nk_filename is not None:
        nk_data = np.genfromtxt(nk_filename, delimiter=',')
        f_nk = nk_data[:, 0]
        n_interp = np.interp(f, f_nk, nk_data[:, 1])
        k_interp = np.interp(f, f_nk, nk_data[:, 2])
        eps1, eps2 = nk_to_eps(n_interp, k_interp)
        eps = eps1 + 1j * eps2
    else:
        eps = data[:, 0] + 1j * data[:, 1]
    return f, eps, Sn

# Dictionary to hold data for each material
materials = {}

# Materials that use the standard effective data file
for mat in ['SiO2', 'STO', 'GaAs', 'LSAT', 'NGO']:
    filename = rf'C:\nano_optics_ml_data\raw\{mat}_eff.csv'
    f, eps, Sn = load_eff_data(filename)
    materials[mat] = {'f': f, 'eps': eps, 'Sn': Sn}

# For CaF2, compute epsilon using the nk file
f, eps, Sn = load_eff_data(r'C:\nano_optics_ml_data\raw\CaF2_eff.csv', use_nk=True, nk_filename=r'C:\nano_optics_ml_data\raw\CaF2_nk.csv')
materials['CaF2'] = {'f': f, 'eps': eps, 'Sn': Sn}

# For Au, the available data is in the nk file only.
# Compute epsilon from Au_nk.csv and assign a dummy Sn (ones) since no measured Sn exists.
data_Au = np.genfromtxt(r'C:\nano_optics_ml_data\raw\Au_nk.csv', delimiter=',')
f_Au = data_Au[:, 0]
n_Au = data_Au[:, 1]
k_Au = data_Au[:, 2]
eps1_Au, eps2_Au = nk_to_eps(n_Au, k_Au)
eps_Au = eps1_Au + 1j * eps2_Au
Sn_Au = np.ones_like(eps_Au)
materials['Au'] = {'f': f_Au, 'eps': eps_Au, 'Sn': Sn_Au}

# Create separate plots for each material
for mat, data in materials.items():
    f = data['f']
    eps = data['eps']
    Sn = data['Sn']
    
    # Create a new figure with two subplots: one for epsilon and one for Sn.
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))
    
    # Plot dielectric permittivity (ε)
    axs[0].plot(f, np.real(eps), label='Re(ε)')
    axs[0].plot(f, np.imag(eps), '--', label='Im(ε)')
    axs[0].set_title(f'{mat}: Dielectric Permittivity')
    axs[0].set_xlabel('Frequency')
    axs[0].set_ylabel('ε')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot the measured far-field signal (Sn)
    axs[1].plot(f, np.real(Sn), label='Re(Sn)')
    axs[1].plot(f, np.imag(Sn), '--', label='Im(Sn)')
    axs[1].set_title(f'{mat}: Measured Far-field Signal')
    axs[1].set_xlabel('Frequency')
    axs[1].set_ylabel('Sn')
    axs[1].legend()
    axs[1].grid(True)
    
    fig.tight_layout()
    plt.show()
