from . import rust_ext

__version__ = "0.1.0"

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class psd1d(BaseEstimator, RegressorMixin):
    def __init__(self, max_lag=100, delta=1.0):
        self.max_lag = max_lag
        self.delta = delta

    def fit(self, y, x=None):
        """
        y:
        x: Ingored
        """
        if y.ndim > 1:
            raise ValueError
        self.max_lag = min(self.max_lag, y.shape[0])

        if y.dtype in (np.float64, np.complex128):
            pass
        elif y.dtype in (np.complex64, np.complex256):
            y = y.astype(np.complex128)
        else:
            y = y.astype(np.float64)

        if y.dtype == np.float64:
            self.gamma_ = rust_ext.reflection_coeff_f64(self.max_lag, y)
            assert self.gamma_.shape[0] == self.max_lag + 1
            self.p_ = rust_ext.variance_f64(self.gamma_.diagonal(), y)
            self.lag_ = self.max_lag
        elif y.dtype == np.complex128:
            self.gamma_ = rust_ext.reflection_coeff_c64(self.max_lag, y)
            assert self.gamma_.shape[0] == self.max_lag + 1
            self.p_ = rust_ext.variance_c64(self.gamma_.diagonal(), y)
            self.lag_ = self.max_lag
        else:
            raise ValueError

        return self

    def predict(self, x):
        """
        x:
        """
        if x.ndim > 1:
            raise ValueError
        if self.lag_ > self.max_lag:
            raise ValueError

        if self.gamma_.dtype == np.float64:
            return rust_ext.power_spector_f64(
                self.delta, self.p_[self.lag_], self.gamma_[self.lag_, :], x
            )
        elif self.gamma_.dtype == np.complex128:
            return rust_ext.power_spector_c64(
                self.delta, self.p_[self.lag_], self.gamma_[self.lag_, :], x
            )
        else:
            raise ValueError
