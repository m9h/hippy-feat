"""Joint EEG-fMRI fusion with a differentiable balloon/HRF model.

Implements a simplified asymmetric fusion approach: EEG band-power
envelopes are mapped to a latent neural state via a learnable affine
transform, convolved with a canonical hemodynamic response function
(double-gamma HRF from SPM), and compared to observed BOLD.  The model
parameters (gain, offset, HRF scale) are optimised end-to-end with
``optax.adam`` through JAX autodiff.

Key components:
    - ``balloon_model`` -- canonical double-gamma HRF convolution that
      converts a neural state time series to predicted BOLD signal.
    - ``SymmetricalFusion`` -- class that holds co-registered EEG
      envelope and fMRI data, fits the affine + HRF parameters, and
      provides a ``predict`` method for real-time BOLD forecasting.

This module is complementary to the rest of the hippy-feat pipeline:
the fMRI side can be preprocessed with :mod:`jaxoccoli.motion` and
:mod:`jaxoccoli.spatial`, and the resulting betas or connectivity
matrices from :mod:`jaxoccoli.glm` / :mod:`jaxoccoli.covariance` can
be cross-validated against EEG-derived predictions.

References:
    Friston et al. (2000) "Nonlinear responses in fMRI: the balloon
    model, Volterra kernels, and other hemodynamics."
    Huster et al. (2012) "Methods for simultaneous EEG-fMRI: an
    introductory review."
"""

import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
import optax
from functools import partial

def balloon_model(neural_state, tr, h_params):
    """Predict BOLD signal from neural activity via HRF convolution.

    Generates a canonical double-gamma HRF kernel (SPM parameterisation:
    peak at ~6 s, undershoot at ~16 s, ratio 1/6) at 0.1 s resolution,
    scales it by ``h_params[0]``, and convolves with *neural_state*.

    Args:
        neural_state: (T,) neural activity time series sampled at the
            fMRI repetition time.
        tr: Repetition time in seconds (currently used for reference;
            the kernel is generated at a fixed 0.1 s resolution).
        h_params: Sequence [scale, delay_offset].  *scale* multiplicatively
            adjusts the HRF amplitude; *delay_offset* is reserved for
            future learnable delay shifting.

    Returns:
        (T,) predicted BOLD signal (truncated to input length).
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
    """Asymmetric EEG-to-BOLD fusion model with a differentiable HRF.

    Maps an EEG band-power envelope to predicted BOLD via a learnable
    affine transform and canonical HRF convolution.  Parameters
    ``[alpha, beta, hrf_scale]`` are optimised with Adam to minimise
    MSE against observed fMRI.

    The EEG envelope is downsampled to the fMRI TR grid by simple
    bin-averaging during ``__init__``.

    Args:
        eeg_data: (T_eeg,) EEG envelope time series (e.g. alpha power).
        fmri_data: (T_fmri,) BOLD signal from the ROI of interest.
        tr: fMRI repetition time in seconds.
    """

    def __init__(self, eeg_data, fmri_data, tr):
        """Initialise the fusion model and downsample EEG to the fMRI grid.

        Args:
            eeg_data: (T_eeg,) EEG band-power envelope.
            fmri_data: (T_fmri,) BOLD signal from a single ROI.
            tr: Repetition time in seconds.
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
        """Mean squared error between observed and predicted BOLD.

        Simplified asymmetric forward model:
            neural_state = alpha * EEG_envelope + beta
            BOLD_pred    = HRF(hrf_scale) * neural_state
            loss         = MSE(BOLD_observed, BOLD_pred)

        Args:
            params: (3,) array [alpha, beta, hrf_scale].

        Returns:
            Scalar MSE loss.
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
        """Fit the fusion model to the stored EEG and fMRI data.

        Runs *n_iter* Adam gradient steps, optimising
        ``[alpha, beta, hrf_scale]`` starting from ``[1, 0, 1]``.

        Args:
            n_iter: Number of optimisation iterations (default 100).

        Returns:
            (3,) fitted parameters [alpha, beta, hrf_scale].
        """
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
        """Forward-predict BOLD from a new EEG segment.

        Args:
            params: (3,) fitted parameters [alpha, beta, hrf_scale].
            eeg_segment: (T,) new EEG envelope segment (at fMRI TR rate).

        Returns:
            (T,) predicted BOLD signal.
        """
        alpha, beta, hrf_scale = params
        neural = alpha * eeg_segment + beta
        return balloon_model(neural, self.tr, [hrf_scale, 0])
