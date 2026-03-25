import jax.numpy as jnp
import nibabel as nib
import numpy as np

def load_nifti(path):
    """
    Load Nifti file and return JAX array and affine.
    """
    img = nib.load(path)
    data = img.get_fdata()
    # Convert to JAX array
    return jnp.array(data), img.affine

def save_nifti(data, affine, path):
    """
    Save JAX array to Nifti file.
    """
    # Convert to numpy
    data_np = np.array(data)
    img = nib.Nifti1Image(data_np, affine)
    nib.save(img, path)
