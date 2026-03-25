import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from functools import partial
import optax


def _precompute_grid(vol_shape, center):
    """
    Build the centered homogeneous coordinate grid once.

    Args:
        vol_shape: tuple (X, Y, Z)
        center: jnp.array of shape (3,)

    Returns:
        coords_homo: jnp.array (4, N) where N = X*Y*Z
    """
    x = jnp.arange(vol_shape[0])
    y = jnp.arange(vol_shape[1])
    z = jnp.arange(vol_shape[2])
    grid = jnp.meshgrid(x, y, z, indexing='ij')
    coords = jnp.stack(grid)  # (3, X, Y, Z)
    coords_centered = coords - center[:, None, None, None]
    coords_flat = coords_centered.reshape(3, -1)
    coords_homo = jnp.vstack([coords_flat, jnp.ones((1, coords_flat.shape[1]))])
    return coords_homo


class RigidBodyRegistration:
    """
    JAX-based Rigid Body Motion Correction (6 DOF).
    Optimizes 3 rotations and 3 translations per volume.
    """
    def __init__(self, template, vol_shape, step_size=0.1, n_iter=50):
        self.template = template # (X, Y, Z)
        self.vol_shape = vol_shape # (X, Y, Z)
        self.center = jnp.array(vol_shape) / 2.0
        self.n_iter = n_iter
        
        # Optimizer
        self.optimizer = optax.adam(learning_rate=step_size)
    
    @staticmethod
    def make_affine_matrix(params):
        """
        Constructs a 4x4 affine matrix from 6 params: [tx, ty, tz, rx, ry, rz]
        rx, ry, rz are Euler angles in radians.
        """
        tx, ty, tz, rx, ry, rz = params
        
        # Rotation X
        Rx = jnp.array([
            [1, 0, 0, 0],
            [0, jnp.cos(rx), -jnp.sin(rx), 0],
            [0, jnp.sin(rx), jnp.cos(rx), 0],
            [0, 0, 0, 1]
        ])
        
        # Rotation Y
        Ry = jnp.array([
            [jnp.cos(ry), 0, jnp.sin(ry), 0],
            [0, 1, 0, 0],
            [-jnp.sin(ry), 0, jnp.cos(ry), 0],
            [0, 0, 0, 1]
        ])
        
        # Rotation Z
        Rz = jnp.array([
            [jnp.cos(rz), -jnp.sin(rz), 0, 0],
            [jnp.sin(rz), jnp.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Translation
        T = jnp.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
        
        # Combined: T @ Rz @ Ry @ Rx
        return T @ Rz @ Ry @ Rx

    @partial(jax.jit, static_argnums=(0,))
    def apply_transform(self, image, params):
        """
        Applies rigid transform to the image using interpolation.
        """
        matrix = self.make_affine_matrix(params)
        
        # Grid generation (implicit)
        # We need to map output coordinates (grid) back to input coordinates via Inverse Transform
        # Output grid coordinates
        x = jnp.arange(self.vol_shape[0])
        y = jnp.arange(self.vol_shape[1])
        z = jnp.arange(self.vol_shape[2])
        grid = jnp.meshgrid(x, y, z, indexing='ij')
        
        # Stack coords: (3, X, Y, Z)
        coords = jnp.stack(grid)
        
        # Center coordinates
        coords_centered = coords - self.center[:, None, None, None]
        
        # Reshape for matrix mul: (4, N) (homogenous)
        coords_flat = coords_centered.reshape(3, -1)
        coords_homo = jnp.vstack([coords_flat, jnp.ones((1, coords_flat.shape[1]))])
        
        # Apply Inverse transform (map target -> source)
        # matrix is source -> target? Wait.
        # Ideally we search for transform T that aligns Moving to Fixed.
        # Moving(T(x)) ~ Fixed(x)
        # So we want T^-1 to look up pixels in Moving image.
        
        inv_matrix = jnp.linalg.inv(matrix)
        
        transformed_coords_homo = inv_matrix @ coords_homo
        
        transformed_coords = transformed_coords_homo[:3, :]
        
        # Un-center
        transformed_coords = transformed_coords + self.center[:, None]
        
        # Reshape back to grid for map_coordinates
        # map_coordinates expects (Rank, OutputShape)
        # transformed_coords is (3, XYZ)
        sampled_coords = transformed_coords.reshape(3, *self.vol_shape)
        
        # Interpolate
        # order=1 (Linear) usually fast enough for motion correction loops
        resampled = map_coordinates(image, sampled_coords, order=1, mode='nearest')
        
        return resampled

    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, params, moving_image):
        """
        Mean Squared Error loss between Template and Transformed Moving Image.
        """
        transformed = self.apply_transform(moving_image, params)
        return jnp.mean((self.template - transformed) ** 2)

    @partial(jax.jit, static_argnums=(0,))
    def register_volume(self, moving_image, init_params=None):
        """
        Effectively optimizes the 6 parameters.
        Returns: (best_params, registered_image)
        """
        if init_params is None:
            init_params = jnp.zeros(6) # Identity
            
        params = init_params
        opt_state = self.optimizer.init(params)
        
        # Optimization Loop (unrolled with scan)
        def step(carrier, _):
            p, state = carrier
            grads = jax.grad(self.loss_fn)(p, moving_image)
            updates, new_state = self.optimizer.update(grads, state)
            new_params = optax.apply_updates(p, updates)
            return (new_params, new_state), None
            
        (best_params, _), _ = jax.lax.scan(step, (params, opt_state), None, length=self.n_iter)
        
        registered = self.apply_transform(moving_image, best_params)

        return best_params, registered


class GaussNewtonRegistration:
    """
    JAX-based Rigid Body Motion Correction (6 DOF) using Gauss-Newton
    optimization.

    Converges in 5-15 iterations instead of 50 Adam steps by computing
    the analytic Jacobian of the residual vector with respect to the
    6 rigid-body parameters and solving the normal equations:

        delta = (J'J)^{-1} J' r

    The output coordinate grid is precomputed once in __init__ rather
    than rebuilt every apply_transform call.

    Parameters match RigidBodyRegistration:
        template: (X, Y, Z) reference volume
        vol_shape: tuple (X, Y, Z)
        n_iter: max Gauss-Newton iterations (default 10)
        damping: Levenberg-Marquardt damping for (J'J + damping*I) (default 1e-4)
    """

    def __init__(self, template, vol_shape, n_iter=10, damping=1e-4):
        self.template = template
        self.vol_shape = vol_shape
        self.center = jnp.array(vol_shape, dtype=jnp.float32) / 2.0
        self.n_iter = n_iter
        self.damping = damping

        # Precompute the output coordinate grid (cached, not recomputed)
        self.coords_homo = _precompute_grid(vol_shape, self.center)

    @staticmethod
    def make_affine_matrix(params):
        """
        Constructs a 4x4 affine matrix from 6 params: [tx, ty, tz, rx, ry, rz].
        Identical to RigidBodyRegistration.make_affine_matrix.
        """
        tx, ty, tz, rx, ry, rz = params

        Rx = jnp.array([
            [1, 0, 0, 0],
            [0, jnp.cos(rx), -jnp.sin(rx), 0],
            [0, jnp.sin(rx), jnp.cos(rx), 0],
            [0, 0, 0, 1]
        ])
        Ry = jnp.array([
            [jnp.cos(ry), 0, jnp.sin(ry), 0],
            [0, 1, 0, 0],
            [-jnp.sin(ry), 0, jnp.cos(ry), 0],
            [0, 0, 0, 1]
        ])
        Rz = jnp.array([
            [jnp.cos(rz), -jnp.sin(rz), 0, 0],
            [jnp.sin(rz), jnp.cos(rz), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        T = jnp.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
        return T @ Rz @ Ry @ Rx

    @partial(jax.jit, static_argnums=(0,))
    def apply_transform(self, image, params):
        """
        Apply rigid transform using the precomputed coordinate grid.
        """
        matrix = self.make_affine_matrix(params)
        inv_matrix = jnp.linalg.inv(matrix)

        # Use precomputed grid instead of rebuilding
        transformed_coords_homo = inv_matrix @ self.coords_homo
        transformed_coords = transformed_coords_homo[:3, :]
        transformed_coords = transformed_coords + self.center[:, None]
        sampled_coords = transformed_coords.reshape(3, *self.vol_shape)

        resampled = map_coordinates(image, sampled_coords, order=1, mode='nearest')
        return resampled

    @partial(jax.jit, static_argnums=(0,))
    def residual_fn(self, params, moving_image):
        """
        Compute the residual vector: template - transformed(moving).
        Returns a flat (N,) vector.
        """
        transformed = self.apply_transform(moving_image, params)
        return (self.template - transformed).ravel()

    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, params, moving_image):
        """
        Mean Squared Error loss (for comparison / monitoring).
        """
        r = self.residual_fn(params, moving_image)
        return jnp.mean(r ** 2)

    @partial(jax.jit, static_argnums=(0,))
    def _gauss_newton_step(self, params, moving_image):
        """
        Single Gauss-Newton update step.

        Computes: delta = (J'J + damping*I)^{-1} J' r
                  params_new = params - delta

        Memory-efficient approach:
        - J'r via a single VJP (backward pass): grad(0.5*||r||^2) = J'r
        - J'J via 6 JVP (forward) passes, one per parameter, then dot products.
          This avoids materializing the full (N, 6) Jacobian.
        """
        # Residual at current params: (N,)
        r = self.residual_fn(params, moving_image)

        # --- J'r via VJP (single backward pass) ---
        # grad(0.5 * sum(r^2)) w.r.t. params = J' @ r
        def half_sos(p):
            res = self.residual_fn(p, moving_image)
            return 0.5 * jnp.sum(res ** 2)

        Jtr = jax.grad(half_sos)(params)  # (6,)

        # --- J'J via 6 JVP forward passes (unrolled for JIT) ---
        def res_fn(p):
            return self.residual_fn(p, moving_image)

        # Compute the 6 columns of J via JVP with basis tangent vectors
        e0 = jnp.array([1., 0., 0., 0., 0., 0.])
        e1 = jnp.array([0., 1., 0., 0., 0., 0.])
        e2 = jnp.array([0., 0., 1., 0., 0., 0.])
        e3 = jnp.array([0., 0., 0., 1., 0., 0.])
        e4 = jnp.array([0., 0., 0., 0., 1., 0.])
        e5 = jnp.array([0., 0., 0., 0., 0., 1.])

        _, j0 = jax.jvp(res_fn, (params,), (e0,))
        _, j1 = jax.jvp(res_fn, (params,), (e1,))
        _, j2 = jax.jvp(res_fn, (params,), (e2,))
        _, j3 = jax.jvp(res_fn, (params,), (e3,))
        _, j4 = jax.jvp(res_fn, (params,), (e4,))
        _, j5 = jax.jvp(res_fn, (params,), (e5,))

        # Build J'J directly from column dot products (avoids large matmul)
        cols = [j0, j1, j2, j3, j4, j5]
        JtJ = jnp.array([[jnp.dot(cols[i], cols[j]) for j in range(6)]
                          for i in range(6)])  # (6, 6)

        # Damped normal equations: (J'J + damping*I) delta = J'r
        A = JtJ + self.damping * jnp.eye(6)
        delta = jax.scipy.linalg.solve(A, Jtr, assume_a='pos')

        new_params = params - delta
        return new_params

    @partial(jax.jit, static_argnums=(0,))
    def register_volume(self, moving_image, init_params=None):
        """
        Register moving_image to the template using Gauss-Newton optimization.

        Returns: (best_params, registered_image)
        """
        if init_params is None:
            init_params = jnp.zeros(6)

        params = init_params

        def step(p, _):
            new_p = self._gauss_newton_step(p, moving_image)
            return new_p, None

        best_params, _ = jax.lax.scan(step, params, None, length=self.n_iter)

        registered = self.apply_transform(moving_image, best_params)
        return best_params, registered
