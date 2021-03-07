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

        self.gamma_ = rust_ext.reflection_coeff(self.max_lag, y)
        assert self.gamma_.shape[0] == self.max_lag + 1

        self.p_ = rust_ext.variance(self.gamma_.diagonal(), y)
        self.lag_ = self.max_lag

        return self

    def predict(self, x):
        """
        x:
        """
        if x.ndim > 1:
            raise ValueError
        if self.lag_ > self.max_lag:
            raise ValueError

        return rust_ext.power_spector(
            self.delta, self.p_[self.lag_], self.gamma_[self.lag_, :], x
        )
