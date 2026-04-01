"""NIfTI I/O utilities bridging nibabel and JAX arrays.

Provides thin wrappers around ``nibabel`` for loading and saving NIfTI-1
files with automatic conversion between NumPy and JAX arrays.  These are
the entry and exit points of the hippy-feat preprocessing pipeline:
data loaded here flows through :mod:`jaxoccoli.spatial` (smoothing),
:mod:`jaxoccoli.motion` (registration), :mod:`jaxoccoli.glm` (model
fitting), and back out via ``save_nifti``.

Key functions:
    - ``load_nifti`` -- read a ``.nii`` / ``.nii.gz`` file, return a
      ``(jnp.ndarray, affine)`` tuple.
    - ``save_nifti`` -- write a JAX array and affine matrix to NIfTI.
"""

import jax.numpy as jnp
import nibabel as nib
import numpy as np

def load_nifti(path):
    """Load a NIfTI file and return a JAX array plus its affine matrix.

    Uses ``nibabel.load`` under the hood.  The image data is read via
    ``get_fdata()`` (float64 by default) and wrapped in a ``jnp.array``.

    Args:
        path: Filesystem path to a ``.nii`` or ``.nii.gz`` file.

    Returns:
        Tuple of (data, affine) where *data* is a ``jnp.ndarray``
        matching the image dimensions and *affine* is a (4, 4) NumPy
        array encoding the voxel-to-world transformation.
    """
    img = nib.load(path)
    data = img.get_fdata()
    # Convert to JAX array
    return jnp.array(data), img.affine

def save_nifti(data, affine, path):
    """Save a JAX array as a NIfTI-1 file.

    Converts the JAX array to NumPy, wraps it in a
    ``nibabel.Nifti1Image``, and writes to disk.

    Args:
        data: JAX or NumPy array of any dimensionality.
        affine: (4, 4) voxel-to-world affine matrix.
        path: Output filesystem path (``.nii`` or ``.nii.gz``).
    """
    # Convert to numpy
    data_np = np.array(data)
    img = nib.Nifti1Image(data_np, affine)
    nib.save(img, path)
