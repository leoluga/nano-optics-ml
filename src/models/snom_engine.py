import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import math

# --- 1. Define Models ---

class SimpleNet(nn.Module):
    """Standard NN: Input (Beta) -> Output (Signal)"""
    def __init__(self, input_size=2, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.net(x)

class HybridNet(nn.Module):
    """
    Hybrid NN: Input (Beta + FDM_Baseline) -> Output (Correction)
    Final Output = FDM_Baseline + Correction
    """
    def __init__(self, input_size=4, hidden_size=16):
        super().__init__()
        # Input size is 4: [Beta_Re, Beta_Im, FDM_Re, FDM_Im]
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 2) # Output: Correction Real, Correction Imag
        )

    def forward(self, x):
        # x columns: [beta_re, beta_im, fdm_re, fdm_im]
        fdm_signal = x[:, 2:] # The last 2 columns are the physics baseline
        correction = self.net(x)
        return fdm_signal + correction # SKIP CONNECTION

# --- 2. The Master Wrapper ---

class NanoOpticsEngine:
    def __init__(self, model_type='HNN', hidden_size=16, device='cpu', physics_params=None):
        """
        model_type: 'NN' (Standard) or 'HNN' (Hybrid)
        physics_params: Dict with keys {'radius', 'length', 'g_factor', 'harmonic'} 
                        (Only needed for HNN prediction)
        """
        self.model_type = model_type
        self.device = device
        self.is_fitted = False
        self.physics_params = physics_params if physics_params else {}
        
        # Initialize Scalers
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        # Initialize Architecture
        if self.model_type == 'NN':
            self.model = SimpleNet(input_size=2, hidden_size=hidden_size).to(device)
        elif self.model_type == 'HNN':
            self.model = HybridNet(input_size=4, hidden_size=hidden_size).to(device)
        else:
            raise ValueError("model_type must be 'NN' or 'HNN'")

    def _epsilon_to_beta(self, eps):
        """Physics transformation: beta = (eps - 1) / (eps + 1)"""
        return (eps - 1) / (eps + 1)

    def _compute_fdm_baseline(self, eps_complex, frequencies):
        """
        Internal FDM Calculator.
        Replicates your sSnomModel logic purely for prediction generation.
        """
        if not self.physics_params:
            raise ValueError("Physics params (radius, length, g) are required for HNN prediction.")

        # Unpack params
        a = self.physics_params.get('radius', 50e-7)
        L = self.physics_params.get('length', 500e-7)
        g = self.physics_params.get('g_factor', 0.7 * np.exp(0.07 * 1j))
        n_harm = self.physics_params.get('harmonic', 2)
        
        # --- Simplified Vectorized FDM Logic ---
        # 1. Geometry Setup
        t = np.linspace(0, math.pi, 30)
        A_amp = 68e-7 
        z_t = A_amp * (1 - np.cos(t)) # Tip motion
        
        # 2. Beta Calculation
        beta = self._epsilon_to_beta(eps_complex) # Shape (N_freqs,)
        
        # Reshape for broadcasting: z (1, Time), beta (Freq, 1)
        z_grid = z_t[np.newaxis, :]
        beta_grid = beta[:, np.newaxis]
        
        # 3. Finite Dipole Model (Model 2 / Lightning Rod)
        W_0 = z_grid + 1.31 * a
        W_1 = z_grid + a / 2
        
        # Geometric factors (depend only on z)
        f_0 = (g - (a + z_grid + W_0) / (2 * L)) * np.log(4 * L / (a + 2 * z_grid + 2 * W_0)) / np.log(4 * L / a)
        f_1 = (g - (a + z_grid + W_1) / (2 * L)) * np.log(4 * L / (a + 2 * z_grid + 2 * W_1)) / np.log(4 * L / a)
        
        # Effective Polarizability
        alpha_eff = beta_grid * f_0 / (1 - beta_grid * f_1) # Shape (Freq, Time)
        
        # 4. Demodulation (Fourier Transform)
        fourier_kernel = np.cos(n_harm * t)[np.newaxis, :]
        Sn_sample = np.trapz(alpha_eff * fourier_kernel, axis=1) # Shape (Freq,)
        
        # 5. Reference Signal (Gold)
        eps_ref = -1e4 + 1e4j
        beta_ref = (eps_ref - 1)/(eps_ref + 1)
        alpha_ref = beta_ref * f_0 / (1 - beta_ref * f_1)
        Sn_ref = np.trapz(alpha_ref * fourier_kernel, axis=1) # Shape (1, Time) -> (1,)
        # Note: Sn_ref is technically constant per z-step, but here it matches dimensions
        
        # 6. Far Field Factor (FFF)
        # Simplified FFF for typical IR range (approximate if full formula is too heavy)
        # Or use the full logic if strictly needed. For now, we assume simple normalization.
        # If your data has FFF baked in, apply it here. 
        # Sn_final = (Sn_sample / Sn_ref) * 0.25 * FFF
        # For simplicity in this example, we assume FFF is roughly constant or handled externally
        # unless you want to paste the full FFF function here.
        
        return (Sn_sample / Sn_ref) # Return normalized signal

    def train(self, df_train, epochs=2000, lr=1e-3):
        """
        Train the model.
        Input df must have: 'eps1', 'eps2', 'Sn_exp_real', 'Sn_exp_imag'
        For HNN, it ALSO needs: 'Sn_fdm_real', 'Sn_fdm_imag'
        """
        # 1. Prepare Beta (Common to both)
        eps = df_train['eps1'].values + 1j * df_train['eps2'].values
        beta = self._epsilon_to_beta(eps)
        
        if self.model_type == 'NN':
            X = np.column_stack([beta.real, beta.imag])
        else: # HNN
            # We need the FDM columns from your CSV
            fdm_real = df_train['Sn_fdm_real'].values
            fdm_imag = df_train['Sn_fdm_imag'].values
            X = np.column_stack([beta.real, beta.imag, fdm_real, fdm_imag])

        y = df_train[['Sn_exp_real', 'Sn_exp_imag']].values

        # 2. Scale
        X_scaled = self.scaler_x.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        self.is_fitted = True

        # 3. Convert to Tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        # 4. Loop
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        print(f"Training {self.model_type} on {self.device}...")
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss {loss.item():.6f}")

    def predict(self, eps_complex, frequencies=None):
        """
        Universal Predictor.
        For NN: just uses eps_complex.
        For HNN: Calculates FDM baseline internally using eps_complex + frequencies.
        """
        if not self.is_fitted:
            raise ValueError("Model not trained!")
        
        self.model.eval()
        beta = self._epsilon_to_beta(eps_complex)

        if self.model_type == 'NN':
            X = np.column_stack([beta.real, beta.imag])
        else:
            # HNN REQUIREMENT: We must generate the physics baseline on the fly
            if frequencies is None:
                raise ValueError("HNN requires 'frequencies' array to calculate FDM baseline.")
            
            # Calculate FDM internally
            # Note: For highest accuracy, you might want to allow passing pre-computed FDM
            # But this automates it.
            fdm_signal = self._compute_fdm_baseline(eps_complex, frequencies)
            
            X = np.column_stack([beta.real, beta.imag, fdm_signal.real, fdm_signal.imag])

        # Scale -> Predict -> Inverse Scale
        X_scaled = self.scaler_x.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).cpu().numpy()
            
        pred_raw = self.scaler_y.inverse_transform(pred_scaled)
        return pred_raw[:, 0] + 1j * pred_raw[:, 1]

    def save(self, filepath):
        checkpoint = {
            'model_type': self.model_type,
            'physics_params': self.physics_params,
            'state_dict': self.model.state_dict(),
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'is_fitted': self.is_fitted
        }
        torch.save(checkpoint, filepath)
        print(f"Saved {self.model_type} engine to {filepath}")

    @classmethod
    def load(cls, filepath, device='cpu'):
        ckpt = torch.load(filepath, map_location=device)
        
        # Re-initialize engine with saved config
        engine = cls(
            model_type=ckpt['model_type'], 
            device=device,
            physics_params=ckpt.get('physics_params')
        )
        
        engine.model.load_state_dict(ckpt['state_dict'])
        engine.scaler_x = ckpt['scaler_x']
        engine.scaler_y = ckpt['scaler_y']
        engine.is_fitted = ckpt['is_fitted']
        return engine