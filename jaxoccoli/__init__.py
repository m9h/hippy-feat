"""GPU-accelerated fMRI preprocessing and differentiable connectivity in JAX.

jaxoccoli (JAX Operations for Connectivity, Covariance, and Linear Imaging)
is the core library of hippy-feat.  It provides a modular collection of
pure-JAX primitives for the full fMRI analysis pipeline:

    1. **I/O** -- NIfTI load/save with automatic JAX array conversion.
    2. **Spatial** -- Bilateral filtering, Gaussian/spherical convolution.
    3. **Motion** -- Rigid-body registration (Adam and Gauss-Newton solvers).
    4. **GLM** -- General Linear Model with Cholesky-accelerated OLS.
    5. **Bayesian beta** -- Conjugate and NUTS-based single-trial estimation
       with variance propagation (addresses Rissman/Mumford gap).
    6. **Statistics** -- Voxelwise t/F-statistics and p-values.
    7. **Permutation** -- Max-T permutation testing with batched vmap.
    8. **Covariance** -- Empirical, partial, and posterior-marginalised
       covariance/correlation with variance-aware extensions.
    9. **Matrix** -- SPD manifold operations, Cholesky inversion, Toeplitz.
   10. **Fourier** -- Hilbert transform, analytic signal, frequency filtering.
   11. **Graph** -- Laplacian, spectral embedding, Chebyshev filtering,
       modularity, sparse message passing.
   12. **Interpolation** -- Linear, spectral, and hybrid temporal interpolation.
   13. **Learnable** -- vbjax-style make_*() factories for learnable atlases,
       covariance, frequency filters, and manifold constraints.
   14. **Losses** -- Differentiable losses for connectivity, modularity,
       reliability-weighted objectives, and information-theoretic measures.
   15. **Connectivity** -- Sliding-window and dynamic functional connectivity.
   16. **Transport** -- Sinkhorn, Wasserstein, and Gromov-Wasserstein
       distances for parcellation-free FC comparison.
   17. **Signatures** -- Rough-path log-signatures for effective connectivity
       and lead-lag (Levy area) extraction via signax.
   18. **Fusion** -- Joint EEG-fMRI fusion with a differentiable balloon model.

Design philosophy:
    - Pure functions, JIT/vmap/grad compatible, no Equinox dependency.
    - vbjax-style factory pattern: ``make_*() -> (params, forward_fn)``.
    - Bayesian variance propagation throughout the pipeline so that
      downstream connectivity estimates account for single-trial uncertainty.
"""

from .glm import GeneralLinearModel
from .permutation import PermutationTest
from .stats import compute_t_stat, compute_f_stat
from .io import load_nifti, save_nifti
from .spatial import (
    bilateral_filter_3d,
    kernel_gaussian,
    spatial_conv,
    spherical_conv,
    spherical_geodesic,
)
from .motion import RigidBodyRegistration, GaussNewtonRegistration

# Phase 1: Core functional primitives
from .covariance import (
    attenuated_corr,
    conditional_cov,
    corr,
    cov,
    paired_cov,
    partial_corr,
    partial_cov,
    posterior_corr,
    precision,
    weighted_corr,
)
from .matrix import (
    cholesky_invert,
    cone_project_spd,
    ensure_spd,
    mean_geom_spd,
    mean_logeuc_spd,
    sym2vec,
    symmetric,
    tangent_project_spd,
    toeplitz,
    vec2sym,
)
from .fourier import (
    analytic_signal,
    envelope,
    hilbert_transform,
    instantaneous_frequency,
    instantaneous_phase,
    product_filter,
    product_filtfilt,
    unwrap,
)

# Phase 2: Graph, spectral embedding, Chebyshev filtering, sparse ops
from .graph import (
    adjacency_to_edge_index,
    chebyshev_filter,
    degree,
    diffusion_mapping,
    girvan_newman_null,
    graph_laplacian,
    laplacian_eigenmaps,
    make_chebyshev_filter,
    modularity_matrix,
    relaxed_modularity,
    sparse_aggregate,
    sparse_degree,
    sparse_graph_conv,
    spectral_features,
)
from .interpolate import (
    hybrid_interpolate,
    linear_interpolate,
    spectral_interpolate,
)

# Phase 3: Learnable components (vbjax factory pattern)
from .learnable import (
    fisher_rao_metric,
    make_atlas_linear,
    make_atlas_linear_uncertain,
    make_atlas_natural_grad,
    make_freq_filter,
    make_learnable_cov,
    make_orthogonal_constraint,
    make_simplex_constraint,
    make_spd_constraint,
    natural_gradient,
    natural_gradient_step,
    init_butterworth_spectrum,
    init_ideal_spectrum,
)
from .losses import (
    compactness_loss,
    connectopy_loss,
    dispersion_loss,
    eigenmaps_loss,
    entropy,
    equilibrium_loss,
    expected_modularity_loss,
    js_divergence,
    kl_divergence,
    modularity_loss,
    multivariate_kurtosis,
    qcfc_loss,
    reference_tether_loss,
    reliability_weighted_loss,
    smoothness_loss,
)

# Phase 4: Connectivity extensions
from .connectivity import (
    SlidingWindowConnectivity,
    dynamic_connectivity,
    sample_nonoverlapping_windows,
    sample_overlapping_windows,
    sliding_window_corr,
)

# Optimal transport for FC comparison
from .transport import (
    gromov_wasserstein,
    gromov_wasserstein_fc,
    sinkhorn,
    wasserstein_distance,
    wasserstein_fc_distance,
)

# TOF-MRA angiography preprocessing
from .angiography import (
    build_vessel_tree,
    estimate_radii,
    frangi_enhance,
    label_branches,
    skeletonize_vessels,
    threshold_vessels,
    tof_pipeline,
    vessel_density_map,
)

# HuggingFace foundation model adapters
from .hf_encoder import (
    HFModelAdapter,
    HFEncoderParams,
    TribeV2Adapter,
    make_cortical_projection,
    make_hf_encoder,
    register_adapter,
    get_adapter,
    torch_to_jax,
)

# NSD validation (RSA, noise ceiling, category selectivity)
from .nsd import (
    category_selectivity,
    compare_rdms,
    load_nsd_betas,
    noise_ceiling_r,
    rdm_from_betas,
    split_half_rdms,
    upper_triangle,
)

# DOT/fNIRS adapter (dot-jax FEM mesh → cortical surface)
from .dot_adapter import (
    DOTFrameProcessor,
    MeshToCortexParams,
    make_mesh_to_cortex,
    simulate_dot_mesh_nodes,
    simulate_hbo_frame,
)

# Real-time per-TR Bayesian decoding (Variant G AR(1) conjugate)
from .realtime import (
    RTPipeline,
    RTPipelineConfig,
    build_lss_design_matrix,
    confidence_mask,
    make_glover_hrf,
)
from .rtcloud import make_rtcloud_decoder

# Complex-valued fMRI: NORDIC denoising, phase regression, complex GLM
from .complex import (
    from_mag_phase,
    from_real_imag,
    to_mag_phase,
    unwrap_phase_temporal,
    detrend_phase_voxelwise,
    voxelwise_zscore_complex,
)
from .nordic import (
    estimate_noise_sigma_complex,
    marchenko_pastur_threshold,
    nordic_global,
    nordic_streaming_window,
)
from .phase_regression import (
    phase_as_design_column,
    phase_regress_residuals,
)
from .complex_glm import (
    complex_snr_map,
    complex_variant_g_forward,
)

# GLMsingle stages 2 + 3 — adaptive noise regressors and fractional ridge.
# Task 2.1 bake-off identified these as where GLMsingle's win lives.
from .glmdenoise import (
    GLMdenoiseResult,
    extract_noise_components,
    glmdenoise_fit,
    per_voxel_r2,
    select_noise_pool,
)
from .fracridge import (
    fracridge_cv,
    fracridge_solve,
)
from .compcor import (
    acompcor_components,
    append_compcor_to_design,
    tcompcor_components,
)
from .multiway_nordic import (
    hosvd_threshold_4d,
    hosvd_threshold_5d,
)
from .multiway_nordic_patch import patch_tucker_threshold_4d
from .motion_phase import (
    apply_translation,
    cross_power_spectrum,
    estimate_translation,
    hamming_window_3d,
    register_translation,
)
from .streaming_kalman import (
    StreamingKalmanState,
    StreamingKalmanAR1State,
    init_streaming_kalman,
    init_streaming_kalman_ar1,
    streaming_kalman_run,
    streaming_kalman_ar1_run,
    streaming_kalman_update,
    streaming_kalman_ar1_update,
)
