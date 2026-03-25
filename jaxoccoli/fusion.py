import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
import optax
from functools import partial

def balloon_model(neural_state, tr, h_params):
    """
    Simplified Balloon Model to predict BOLD from Neural State.
    Uses a Hemodynamic Response Function (HRF) convolution kernel.
    """
    # Standard SPM HRF parameters (could be learnable)
    # For speed, we use a fixed canonical HRF basis.
    # h_params: [scale, delay_offset]
    
    # Generate kernel
    dt = 0.1 # Simulation step
    duration = 32.0 
    t = jnp.arange(0, duration, dt)
    
    # Gamma PDF components
    # p1=6, p2=16, ratio=1/6
    def gamma_pdf(t, alpha, beta):
        return (t ** (alpha - 1)) * jnp.exp(-beta * t)
        
    hrf = gamma_pdf(t, 6.0, 1.0) - 1/6 * gamma_pdf(t, 16.0, 1.0)
    hrf = hrf / jnp.sum(hrf) # Normalize sum
    
    # Scale and Shift (approximation)
    scale, _ = h_params
    hrf = hrf * scale
    
    # Upsample neural state to simulation grid?
    # Simple convolution for now:
    # Assume neural_state is already at TR or we upsample it.
    
    bold = convolve(neural_state, hrf, mode='full')[:len(neural_state)]
    return bold

class SymmetricalFusion:
    def __init__(self, eeg_data, fmri_data, tr):
        """
        Joint EEG-fMRI Fusion Model.
        eeg_data: (Time_E,) - Envelope of relevant band (e.g. Alpha/Beta)
        fmri_data: (Time_F,) - BOLD signal from relevant ROI
        tr: Repetition Time (s)
        """
        self.eeg = jnp.array(eeg_data)
        self.fmri = jnp.array(fmri_data)
        self.tr = tr
        
        # Resample EEG to fMRI TR for simple fusion
        # (Preprocessing step assumed done or simple aggregation)
        # Using simple binning/decimation for this demo class
        ratio = len(eeg_data) // len(fmri_data)
        if ratio > 0:
            # Downsample EEG envelope to Match fMRI TR
            # Reshape to (Time_F, Ratio) then mean
            trim_len = len(fmri_data) * ratio
            eeg_trimmed = self.eeg[:trim_len].reshape(len(fmri_data), ratio)
            self.eeg_bold_domain = jnp.mean(eeg_trimmed, axis=1)
        else:
            self.eeg_bold_domain = self.eeg
            
        # Optimizer
        self.optimizer = optax.adam(learning_rate=0.01)

    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, params):
        """
        Joint Loss:
        L = || BOLD_real - Balloon(Neural(EEG)) || + || EEG_real - Inverse(Neural) ||
        
        Simplified "Asymmetric" Fusion for Real-Time:
        Predict BOLD from EEG.
        Neural State = alpha * EEG_envelope + beta
        BOLD_pred = HRF * Neural State
        Loss = MSE(BOLD_pred, BOLD_real)
        """
        alpha, beta, hrf_scale = params
        
        # 1. Neural State Estimate from EEG
        neural_est = alpha * self.eeg_bold_domain + beta
        
        # 2. Predict BOLD
        # Simple HRF convolution
        # We assume a standard kernel shape, scaling intensity
        h_params = [hrf_scale, 0.0]
        bold_pred = balloon_model(neural_est, self.tr, h_params)
        
        # 3. Loss
        mse = jnp.mean((self.fmri - bold_pred) ** 2)
        return mse

    def fit(self, n_iter=100):
        # Params: [alpha, beta, hrf_scale]
        params = jnp.array([1.0, 0.0, 1.0])
        opt_state = self.optimizer.init(params)
        
        @jax.jit
        def step(p, state):
            grads = jax.grad(self.loss_fn)(p)
            updates, new_state = self.optimizer.update(grads, state)
            new_params = optax.apply_updates(p, updates)
            return new_params, new_state, grads
            
        for i in range(n_iter):
             params, opt_state, grads = step(params, opt_state)
             
        return params

    @partial(jax.jit, static_argnums=(0,))
    def predict(self, params, eeg_segment):
        """
        Forward prediction of BOLD from new EEG segment
        """
        alpha, beta, hrf_scale = params
        neural = alpha * eeg_segment + beta
        return balloon_model(neural, self.tr, [hrf_scale, 0])
