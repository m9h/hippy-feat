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
