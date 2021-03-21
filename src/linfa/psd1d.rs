
use num_complex::Complex64;
use ndarray::s;
//use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray::{ArrayBase, RawData, Ix1, Ix2};
use ndarray::prelude::*;
//use linfa::traits::PredictRef;
//use linfa::traits::Fit;
//use linfa::dataset::Records;
//use linfa::dataset::DatasetBase;
//use linfa::error::Error;
use linfa::prelude::*;

use crate::berg_f64;
use crate::berg_c64;

#[derive(Debug, Clone)]
pub struct Psd1d {
    max_lag: Option<usize>,
    delta: f64,
    noise: f64,
}

impl Psd1d {
    pub fn max_lag(mut self, lag: usize) -> Self {
        self.max_lag = Some(lag);
        self
    }

    pub fn delta(mut self, delta: f64) -> Self {
        assert!(delta > 0.);
        self.delta = delta;
        self
    }
}

impl Default for Psd1d {
    fn default() -> Self {
        Self {
            max_lag: None,
            delta: 1.0,
            noise: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct FittedPsd1d<T> {
    lag: usize,
    delta: f64,
    gamma: Vec<Array2<T>>,
    p: Vec<Array1<f64>>,
}

impl<T> FittedPsd1d<T> {
    pub fn lag(mut self, lag: usize) -> Result<Self> {
        let max_lag = self.p[0].shape()[0];
        if lag > max_lag {
            Err(Error::Parameters(format!("Lag is expected under {}, but found {}.", max_lag, lag)))
        } else {
            self.lag = lag;
            Ok(self)
        }
    }

    pub fn delta(mut self, delta: f64) -> Self {
        assert!(delta > 0.);
        self.delta = delta;
        self
    }
}

impl<'a, D: RawData<Elem = f64>, T> Fit<'a, ArrayBase<D, Ix2>, T> for Psd1d {
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<FittedPsd1d<f64>> {
        let x = &dataset.records;
        let samples = x.shape()[0];
        let max_lag = usize::min(self.max_lag.unwrap_or(samples), samples);
        let mut gammas = Vec::new();
        let mut ps = Vec::new();

        for x in x.outer_iter() {
            let gamma = berg_f64::reflection_coeff(max_lag, &x);
            let p = berg_f64::variance(&gamma.diag(), &x);
            gammas.push(gamma);
            ps.push(p);
        }

        let lag = max_lag;
        let result = FittedPsd1d::<f64> {
            lag: lag,
            delta: self.delta,
            gamma: gammas,
            p: ps,
        };

        Ok(result)
    }
}

impl PredictRef<ArrayView1<'a, f64>, Array2::<f64>> for FittedPsd1d::<f64> {
    fn predict_ref(&self, x: &ArrayView1::<f64>) -> Array2::<f64> {
        let mut y = Array2::<f64>::zeros((self.p.len(), x.shape()[0]));
        for (k, (p, gamma)) in self.p.iter().zip(self.gamma.iter()).enumerate() {
            y.slice_mut(s![k, ..]).assign(&berg_f64::power_spector(self.delta, p[self.lag], &gamma.slice(s![self.lag, ..]), x));
        }
        y
    }
}

