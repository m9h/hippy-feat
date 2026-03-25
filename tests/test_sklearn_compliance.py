"""
Tests for strict Scikit-Learn API compliance.
"""

import inspect
import pytest
import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.base import BaseEstimator

from neurojax.functional.alignment.srm import SharedResponseModel

def test_srm_signatures():
    """
    Verify strict API signatures for SharedResponseModel.
    """
    # 1. __init__ must not have *args, **kwargs
    init_sig = inspect.signature(SharedResponseModel.__init__)
    for param in init_sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            pytest.fail("Estimators must not have *args or **kwargs in __init__")

    # 2. fit must look like fit(self, X, y=None, ...)
    fit_sig = inspect.signature(SharedResponseModel.fit)
    fit_params = list(fit_sig.parameters.keys())
    assert fit_params[1] == "X", "fit() first arg must be X"
    assert "y" in fit_params, "fit() must have y argument"

def test_srm_initialization():
    """
    Verify parameter setting behavior.
    """
    model = SharedResponseModel(n_features=10, n_iter=5)
    
    # Init should just set attributes
    assert model.n_features == 10
    assert model.n_iter == 5
    
    # get_params should work out of the box (via BaseEstimator)
    params = model.get_params()
    assert "n_features" in params
    assert params["n_features"] == 10

def test_srm_return_types():
    """
    Verify fit/transform return types (must be CPU numpy).
    """
    n_subj, n_vox, n_time = 3, 100, 50
    X = [np.random.randn(n_vox, n_time) for _ in range(n_subj)]
    
    model = SharedResponseModel(n_features=5, n_iter=2)
    
    # Fit returns self
    assert model.fit(X) is model
    
    # Transform returns list of numpy arrays
    res = model.transform(X)
    assert isinstance(res, list)
    assert len(res) == n_subj
    assert isinstance(res[0], np.ndarray)
    
    # Inverse transform
    s_fake = np.random.randn(5, n_time)
    inv = model.inverse_transform(s_fake)
    assert isinstance(inv, list)
    assert isinstance(inv[0], np.ndarray)

if __name__ == "__main__":
    # Manually run tests if executed directly
    test_srm_signatures()
    test_srm_initialization()
    test_srm_return_types()
    print("All SRM compliance tests passed!")
