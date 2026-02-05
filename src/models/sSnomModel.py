import numpy as np
import math

class sSnomModel:
    def __init__(self, radius, length, g_factor):
        """
        Initialize the SnomModel with probe geometry.
        
        Args:
            radius (float): Tip radius (a) in cm.
            length (float): Tip length (L) in cm.
            g_factor (float): Geometric factor (g), typically around 0.7 - 0.9.
        """
        self.a = radius
        self.L = length
        self.g = g_factor

    def _get_beta(self, eps):
        """Calculates the reflection coefficient beta."""
        return (eps - 1) / (eps + 1)

    # ==========================
    #     DIPOLE MODELS
    # ==========================

    def model_finite_1(self, w, z, eps):
        """
        Finite Dipole Model 1 (Standard).
        Vectorized for performance.
        """
        beta = self._get_beta(eps)
        # Ensure z is a row vector (1, Nt) and beta is a column vector (Nw, 1)
        z_grid = z[np.newaxis, :]  
        beta_grid = beta[:, np.newaxis]

        # Geometric terms dependent only on z
        term1 = self.g - (self.a + z_grid / self.L) * np.log(4 * self.L / (4 * z_grid + 3 * self.a))
        term2 = np.log(4 * self.L / self.a) - beta_grid * (self.g - (3 * self.a + 4 * z_grid) / (4 * self.L)) * np.log(2 * self.L / (2 * z_grid + self.a))

        alpha_eff = beta_grid * term1 / term2
        return alpha_eff.T # Return shape (Time, Freq)

    def model_finite_2(self, w, z, eps):
        """
        Finite Dipole Model 2 (Alternative/Lightning Rod approximation).
        Vectorized for performance.
        """
        beta = self._get_beta(eps)
        z_grid = z[np.newaxis, :]
        beta_grid = beta[:, np.newaxis]

        W_0 = z_grid + 1.31 * self.a
        W_1 = z_grid + self.a / 2

        f_0 = (self.g - (self.a + z_grid + W_0) / (2 * self.L)) * np.log(4 * self.L / (self.a + 2 * z_grid + 2 * W_0)) / np.log(4 * self.L / self.a)
        f_1 = (self.g - (self.a + z_grid + W_1) / (2 * self.L)) * np.log(4 * self.L / (self.a + 2 * z_grid + 2 * W_1)) / np.log(4 * self.L / self.a)

        alpha_eff = beta_grid * f_0 / (1 - beta_grid * f_1)
        return alpha_eff.T

    def model_point(self, w, z, eps):
        """
        Point Dipole Model.
        """
        beta = self._get_beta(eps)
        z_eff = z + self.a # Point dipole center shift
        
        z_grid = z_eff[np.newaxis, :]
        beta_grid = beta[:, np.newaxis]
        
        alpha_0 = 4 * math.pi * self.a**3
        alpha_eff = alpha_0 / (1 - alpha_0 * beta_grid / (16 * math.pi * z_grid**3))
        
        return alpha_eff.T

    # ==========================
    #     SIGNAL PROCESSING
    # ==========================

    def demodulate(self, alpha_eff, t, n):
        """
        Demodulates the time-varying polarizability at harmonic n.
        Uses np.trapezoid (NumPy 2.0+ safe).
        """
        # Create Fourier kernel
        fourier = np.cos(n * t)
        # Broadcast fourier to match alpha_eff shape: (Time, Freq)
        fourier = fourier[:, np.newaxis] 
        
        # Integrate over time (axis 0)
        Sn = np.trapezoid(alpha_eff * fourier, axis=0)
        return Sn

    def calculate_signal(self, w, z, t, eps_sample, n, model='finite_2', eps_ref=None):
        """
        Main function to calculate normalized signal Sn = S_sample / S_ref.
        
        Args:
            w (array): Frequency array.
            z (array): Tip height array z(t).
            t (array): Time array.
            eps_sample (array): Complex dielectric function of sample.
            n (int): Demodulation harmonic order.
            model (str): 'finite_1', 'finite_2', or 'point'.
            eps_ref (complex): Reference epsilon. Defaults to Gold (-1e4 + 1e4j).
        """
        # 1. Select Model
        if model == 'finite_1':
            func = self.model_finite_1
        elif model == 'point':
            func = self.model_point
        else:
            func = self.model_finite_2 # Default
        
        # 2. Calculate Sample Signal
        alpha_samp = func(w, z, eps_sample)
        s_samp = self.demodulate(alpha_samp, t, n)
        
        # 3. Calculate Reference Signal
        # Default reference is a perfect conductor/Gold
        if eps_ref is None:
            eps_ref_val = -10000 + 10000j 
        else:
            eps_ref_val = eps_ref
            
        # Create array of constant reference epsilon
        eps_ref_arr = np.ones_like(w, dtype=complex) * eps_ref_val
        
        alpha_ref = func(w, z, eps_ref_arr)
        s_ref = self.demodulate(alpha_ref, t, n)

        # 4. Normalize
        return s_samp / s_ref

    # ==========================
    #     STATIC UTILITIES
    # ==========================
    
    @staticmethod
    def lorentz_eps(w, w0, s, gamma, eps_inf):
        """Helper to generate Lorentz oscillator dielectric function."""
        return eps_inf + s / (w0**2 - w**2 - 1j * w * gamma)
    
    @staticmethod
    def far_field_factor(f, eps):
        """Calculates Far Field Factor (FFF)."""
        n0 = np.ones(len(f))
        # Complex refractive index calculation
        n1 = np.sqrt((0.5 * np.sqrt(eps * np.conj(eps)) + np.real(eps) / 2)) + \
             np.sqrt((0.5 * np.sqrt(eps * np.conj(eps)) - np.real(eps) / 2)) * 1j
        
        theta = math.pi / 3
        cos_theta_t = np.sqrt(1 - (n0 * np.sin(theta) / n1)**2)
        
        r_p = (-n0 * cos_theta_t + n1 * np.cos(theta)) / \
              (n0 * cos_theta_t + n1 * np.cos(theta))
        
        return (1 + r_p)**2