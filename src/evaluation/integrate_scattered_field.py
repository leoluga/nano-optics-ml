# src/evaluation/integrate_scattered_field.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import quad_vec
from models.fdm_model import calculate_integrand_for_scattered_field
from models.permittivity import compute_permittivity

# -------------------------
# Main Function
# -------------------------
def integrate_and_plot_scattered_field(material, temperature, omega_cm, n_values, Omega):
    """
    Compute and plot both the complex permittivity and the integrated scattered field.
    
    The output figure is divided into three sections:
      1. Top panel: Plot of complex permittivity vs. frequency.
      2. Middle panel: Title for the integration subplots.
      3. Bottom panel: Grid of integration subplots (number adjusts based on n_values).
    
    Parameters
    ----------
    material : str
        Material key for compute_permittivity (e.g., 'LaAlO3_Zhang1994', 'LSAT').
    temperature : int or str
        Temperature value (e.g., 300).
    omega_cm : array-like
        Frequencies (in cm⁻¹) for both permittivity and integration.
    n_values : list or array-like
        List of order indices for the scattered field integration.
    Omega : float
        Angular frequency (rad/s) used in the integration limit.
    
    Returns
    -------
    integrated_values : np.ndarray
        Array with integrated scattered field values.
    fig : matplotlib.figure.Figure
        The combined plot figure.
    """
    # Compute epsilon values once over the frequency range.
    epsilon_values = compute_permittivity(omega_cm, material, temperature)
    # Compute integrated scattered field values.
    integrated_values = compute_integrated_values(omega_cm, n_values, Omega, epsilon_values)
    
    # Determine layout for integration subplots.
    n_plots = len(n_values)
    if n_plots == 1:
        nrows_integ, ncols_integ = 1, 1
    else:
        ncols_integ = 2
        nrows_integ = int(np.ceil(n_plots / ncols_integ))
    
    # Create figure with GridSpec layout.
    fig = plt.figure(figsize=(10, 6 + nrows_integ * 2.5))
    # Use three rows: Top for epsilon, Middle for integration title, Bottom for integration plots.
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0, nrows_integ])
    
    # Top panel: Epsilon plot.
    ax_epsilon = fig.add_subplot(gs[0])
    plot_epsilon( omega_cm, epsilon_values, material, temperature, ax_epsilon)
    
    # Middle panel: Title for integration subplots.
    ax_integration_title = fig.add_subplot(gs[1])
    ax_integration_title.set_title(
        f'Integrated Scattered Field for {material} at {temperature} K', 
        fontsize=14
    )
    ax_integration_title.axis("off")
    
    # Bottom panel: Integration subplots.
    gs_integ = gridspec.GridSpecFromSubplotSpec(nrows_integ, ncols_integ, subplot_spec=gs[2])
    integ_axes = [fig.add_subplot(gs_integ[i, j]) for i in range(nrows_integ) for j in range(ncols_integ)]
    
    plot_integration_subplots(integ_axes, omega_cm, integrated_values, n_values)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    return integrated_values, fig

# -------------------------
# Helper functions
# -------------------------

def compute_integrated_values(omega_cm, n_values, Omega, epsilon_values):
    """
    Compute the integrated scattered field values over omega_cm and n_values.
    
    Parameters
    ----------
    omega_cm : array-like
        Frequencies (in cm⁻¹).
    n_values : list or array-like
        Order indices for the scattered field calculation.
    Omega : float
        Angular frequency (rad/s) for integration limit.
    epsilon_values : np.ndarray
        Pre-computed complex permittivity values.
    
    Returns
    -------
    integrated_values : np.ndarray
        Array of shape (len(omega_cm), len(n_values)) with integrated results.
    """
    integrated_values = np.empty((len(omega_cm), len(n_values)), dtype=np.complex128)
    for i, _ in enumerate(omega_cm):
        eps = epsilon_values[i]
        for j, n_value in enumerate(n_values):
            result, _ = quad_vec(
                lambda x: calculate_integrand_for_scattered_field(x, eps, n_value),
                0, 2 * np.pi / Omega
            )
            integrated_values[i, j] = result
    return integrated_values


def plot_epsilon(omega_cm, epsilon_values, material, temperature, ax=None):
    """
    Plot the complex permittivity on the given axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    omega_cm : array-like
        Frequencies (in cm⁻¹).
    epsilon_values : np.ndarray
        Complex permittivity values.
    material : str
        Material key.
    temperature : int or str
        Temperature value.
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(omega_cm, epsilon_values.real, 'b-', label='Real Part (ε₁)')
    ax.plot(omega_cm, epsilon_values.imag, 'r-', label='Imaginary Part (ε₂)')
    if temperature is None or temperature == '':
        title = f'Dielectric Function of {material}'
    else:
        title = f'Dielectric Function of {material} at {temperature} K'
    ax.set_title(title)
    ax.set_xlabel('Frequency (cm⁻¹)')
    ax.set_ylabel('ε')
    ax.grid(True)
    ax.legend()


def plot_integration_subplots(ax_array, omega_cm, integrated_values, n_values):
    """
    Plot integration subplots (one per n value) into the provided axes.
    
    Parameters
    ----------
    ax_array : list of matplotlib.axes.Axes
        List of axes for the integration subplots.
    omega_cm : array-like
        Frequencies (in cm⁻¹).
    integrated_values : np.ndarray
        Integrated scattered field values.
    n_values : list or array-like
        Order indices used in the integration.
    """
    for idx, n in enumerate(n_values):
        ax = ax_array[idx]
        ax.plot(omega_cm, integrated_values[:, idx].real, label='Real', color='blue')
        ax.plot(omega_cm, integrated_values[:, idx].imag, label='Imag', color='red')
        ax.set_xlabel('Frequency (cm⁻¹)')
        ax.set_title(f'$S_{{{n}}}(\\omega)$')
        ax.legend()
        ax.grid(True)
    # Hide extra axes if any.
    for ax in ax_array[len(n_values):]:
        ax.set_visible(False)

