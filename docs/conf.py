# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "hippy-feat"
copyright = "2026, Morgan G. Hough"
author = "Morgan G. Hough"
release = "0.2.0"
version = "0.2"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "fieldlist",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# BibTeX
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

# -- Autodoc configuration ---------------------------------------------------
autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "equinox",
    "diffrax",
    "optax",
    "distrax",
    "lineax",
    "optimistix",
    "jaxtyping",
    "numpy",
    "scipy",
    "nibabel",
    "nilearn",
    "matplotlib",
    "h5py",
    "mne",
    "vbjax",
    "numpyro",
    "blackjax",
    "beartype",
    "tqdm",
    "signax",
    "neuromaps",
    "brainspace",
    "brainsmash",
    "templateflow",
    "hypothesis",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "hippy-feat"
html_theme_options = {
    "source_repository": "https://github.com/m9h/hippy-feat",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}
html_static_path = ["_static"]
