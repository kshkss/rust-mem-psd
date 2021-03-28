use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Ix2;
use ndarray::ViewRepr;
use num_complex::Complex64;
use smartcore::api::Predictor;
use smartcore::api::UnsupervisedEstimator;
use smartcore::error::Failed;

use crate::berg_c64;
use crate::berg_f64;

#[derive(Debug, Clone)]
pub struct PsdParameters {
    pub max_lag: usize,
}

impl PsdParameters {
    pub fn new(max_lag: usize) -> Self {
        PsdParameters { max_lag: max_lag }
    }

    pub fn with_max_lag(mut self, max_lag: usize) -> Self {
        self.max_lag = max_lag;
        self
    }
}

pub struct Psd1d<T> {
    lag: usize,
    delta: f64,
    gamma: Vec<Array2<T>>,
    p: Vec<Array1<f64>>,
}

impl<T> Psd1d<T> {
    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    pub fn with_lag(mut self, lag: usize) -> Result<Self, Failed> {
        if lag < self.p[0].shape()[0] {
            self.lag = lag;
            Ok(self)
        } else {
            Err(Failed::predict("Lag should be set less than max_lag"))
        }
    }
}

impl Psd1d<f64> {
    fn _fit(max_lag: usize, x: &ArrayView2<f64>) -> Self {
        let max_lag = usize::min(max_lag, x.shape()[0]);
        let mut gammas = Vec::new();
        let mut ps = Vec::new();

        for x in x.outer_iter() {
            let gamma = berg_f64::reflection_coeff(max_lag, &x);
            let p = berg_f64::variance(&gamma.diag(), &x);
            gammas.push(gamma);
            ps.push(p);
        }

        let lag = max_lag;

        Psd1d {
            delta: 1.,
            lag: lag,
            gamma: gammas,
            p: ps,
        }
    }

    fn _predict(&self, x: &ArrayView1<f64>) -> Array2<f64> {
        let mut y = Array2::<f64>::zeros((self.p.len(), x.shape()[0]));

        for (k, (p, gamma)) in self.p.iter().zip(self.gamma.iter()).enumerate() {
            y.slice_mut(s![k, ..]).assign(&berg_f64::power_spector(
                self.delta,
                p[self.lag],
                &gamma.slice(s![self.lag, ..]),
                x,
            ));
        }
        y
    }
}

impl UnsupervisedEstimator<ArrayView2<'_, f64>, PsdParameters> for Psd1d<f64> {
    fn fit(x: &ArrayView2<f64>, params: PsdParameters) -> Result<Self, Failed> {
        Ok(Psd1d::<f64>::_fit(params.max_lag, x))
    }
}

impl Predictor<ArrayView1<'_, f64>, Array2<f64>> for Psd1d<f64> {
    fn predict(&self, x: &ArrayView1<f64>) -> Result<Array2<f64>, Failed> {
        Ok(self._predict(x))
    }
}

impl Psd1d<Complex64> {
    fn _fit(max_lag: usize, x: &ArrayView2<Complex64>) -> Self {
        let max_lag = usize::min(max_lag, x.shape()[0]);
        let mut gammas = Vec::new();
        let mut ps = Vec::new();

        for x in x.outer_iter() {
            let gamma = berg_c64::reflection_coeff(max_lag, &x);
            let p = berg_c64::variance(&gamma.diag(), &x);
            gammas.push(gamma);
            ps.push(p);
        }

        let lag = max_lag;

        Psd1d {
            delta: 1.,
            lag: lag,
            gamma: gammas,
            p: ps,
        }
    }

    fn _predict(&self, x: &ArrayView1<f64>) -> Array2<f64> {
        let mut y = Array2::<f64>::zeros((self.p.len(), x.shape()[0]));

        for (k, (p, gamma)) in self.p.iter().zip(self.gamma.iter()).enumerate() {
            y.slice_mut(s![k, ..]).assign(&berg_c64::power_spector(
                self.delta,
                p[self.lag],
                &gamma.slice(s![self.lag, ..]),
                x,
            ));
        }
        y
    }
}

impl UnsupervisedEstimator<ArrayView2<'_, Complex64>, PsdParameters> for Psd1d<Complex64> {
    fn fit(x: &ArrayView2<Complex64>, params: PsdParameters) -> Result<Self, Failed> {
        Ok(Psd1d::<Complex64>::_fit(params.max_lag, x))
    }
}

impl Predictor<ArrayView1<'_, f64>, Array2<f64>> for Psd1d<Complex64> {
    fn predict(&self, x: &ArrayView1<f64>) -> Result<Array2<f64>, Failed> {
        Ok(self._predict(x))
    }
}
