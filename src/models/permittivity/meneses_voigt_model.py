import numpy as np
from scipy import special

def meneses_voigt_model(eta, params):
    r"""
    Compute the complex dielectric function based on the causal Voigt/Gaussian model.
    
    The dielectric constant is given by:
    
    .. math::
        \epsilon(\eta) = \epsilon_{\infty} + \sum_j \Big[ g_{ej}(\eta) 
        + i\,g_{cj}^{kkg}(\eta) \Big],
    
    where the Gaussian functions are defined as:
    
    .. math::
        g_{ej}(\eta) = \alpha_j \exp\Big[-4\ln 2\Big(\frac{\eta - \eta_{0j}}{\sigma_j}\Big)^2\Big]
        - \alpha_j \exp\Big[-4\ln 2\Big(\frac{\eta + \eta_{0j}}{\sigma_j}\Big)^2\Big],
    
    and
    
    .. math::
        g_{cj}^{kkg}(\eta) = \frac{2\alpha_j}{\pi} \Bigg[
            D\Big(2\sqrt{\ln2}\frac{\eta + \eta_{0j}}{\sigma_j}\Big)
          - D\Big(2\sqrt{\ln2}\frac{\eta - \eta_{0j}}{\sigma_j}\Big)
        \Bigg],
    
    with the Dawson function defined by
    
    .. math::
        D(x) = e^{-x^2}\int_0^x e^{t^2}\,dt.
    
    Parameters
    ----------
    eta : float or np.ndarray
        The spectral variable (e.g., in cm⁻¹) at which to evaluate the dielectric function.
    epsilon_inf : float
        The high-frequency dielectric constant, \(\epsilon_{\infty}\).
    oscillator_params : list of dict
        A list where each dictionary corresponds to an oscillator and must contain:
            - 'alpha': Amplitude \(\alpha_j\)
            - 'eta0': Peak position \(\eta_{0j}\)
            - 'sigma': Full width at half-maximum \(\sigma_j\)
    
    Returns
    -------
    epsilon : complex or np.ndarray of complex
        The complex dielectric function \(\epsilon(\eta)\).
    
    Example
    -------
    >>> osc_params = [
    ...     {'alpha': 3.7998, 'eta0': 1089.7, 'sigma': 31.454},
    ...     {'alpha': 0.46089, 'eta0': 1187.7, 'sigma': 100.46},
    ...     # ... additional oscillators ...
    ... ]
    >>> eta_values = np.linspace(800, 1200, 500)
    >>> eps = dielectric_function(eta_values, epsilon_inf=2.1232, oscillator_params=osc_params)
    """
    eta = np.array(eta, dtype=np.float64)
    epsilon_inf = params["epsilon_inf"]

    epsilon_real = epsilon_inf * np.ones_like(eta, dtype=np.float64)
    epsilon_imag = np.zeros_like(eta, dtype=np.float64)

    oscillators = params["oscillators"]
    for osc in oscillators:
        alpha = osc['alpha']
        eta0  = osc['eta0']
        sigma = osc['sigma']
        
        gej = (
            alpha * np.exp(-4 * np.log(2) * ((eta - eta0)/sigma)**2) -
            alpha * np.exp(-4 * np.log(2) * ((eta + eta0)/sigma)**2)
        )
        epsilon_real += gej

        x1 = 2 * np.sqrt(np.log(2)) * (eta + eta0) / sigma
        x2 = 2 * np.sqrt(np.log(2)) * (eta - eta0) / sigma
        gcj = (2 * alpha / np.pi) * (special.dawsn(x1) - special.dawsn(x2))
        epsilon_imag += gcj
        
    epsilon = epsilon_real + 1j * epsilon_imag
    return np.sqrt(epsilon)

