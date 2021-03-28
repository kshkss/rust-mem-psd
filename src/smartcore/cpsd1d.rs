use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::s;
use ndarray::stack;
use ndarray::stack_new_axis;
use ndarray::Zip;
use num_complex::Complex64;
use smartcore::api::Predictor;
use smartcore::api::UnsupervisedEstimator;
use smartcore::error::Failed;

use super::psd1d::PSDParameters;
use super::psd1d::PSD1D;

pub struct Cpsd1d<T> {
    auto: PSD1D<T>,
    co: PSD1D<T>,
    quad: PSD1D<Complex64>,
}

impl<T> Cpsd1d<T> {
    pub fn with_delta(mut self, delta: f64) -> Self {
        self.auto = self.auto.with_delta(delta);
        self.co = self.co.with_delta(delta);
        self.quad = self.quad.with_delta(delta);
        self
    }

    pub fn with_lag(mut self, lag: usize) -> Result<Self, Failed> {
        self.auto = self.auto.with_lag(lag)?;
        self.co = self.co.with_lag(lag)?;
        self.quad = self.quad.with_lag(lag)?;
        Ok(self)
    }
}

impl UnsupervisedEstimator<ArrayView2<'_, f64>, PSDParameters> for Cpsd1d<f64> {
    fn fit(x: &ArrayView2<f64>, params: PSDParameters) -> Result<Self, Failed> {
        let samples = x.shape()[0];
        //let features = x.shape()[1];
        let auto = PSD1D::<f64>::fit(x, params.clone())?;

        let combs = x.axis_iter(Axis(1)).combinations(2).count();
        let mut x_co = Array2::<f64>::zeros((samples, combs));
        for (xs, mut z) in x
            .axis_iter(Axis(1))
            .combinations(2)
            .zip(x_co.axis_iter_mut(Axis(1)))
        {
            //z.assign(&(&xs[0] + &xs[1]));
            Zip::from(&mut z)
                .and(&xs[0])
                .and(&xs[1])
                .apply(|c, &a, &b| *c = a + b);
        }
        let co = PSD1D::<f64>::fit(&x_co.view(), params.clone())?;

        let mut x_quad = Array2::<Complex64>::zeros((samples, combs));
        for (xs, mut z) in x
            .axis_iter(Axis(1))
            .combinations(2)
            .zip(x_quad.axis_iter_mut(Axis(1)))
        {
            Zip::from(&mut z)
                .and(&xs[0])
                .and(&xs[1])
                .apply(|c, &a, &b| *c = Complex64::new(a, b));
        }
        let quad = PSD1D::<Complex64>::fit(&x_quad.view(), params)?;

        Ok(Self {
            auto: auto,
            co: co,
            quad: quad,
        })
    }
}

impl Predictor<ArrayView1<'_, f64>, Array3<Complex64>> for Cpsd1d<f64> {
    fn predict(&self, x: &ArrayView1<f64>) -> Result<Array3<Complex64>, Failed> {
        let y_auto = self.auto.predict(x)?;
        let y_co = self.co.predict(x)?;
        let y_quad = self.quad.predict(x)?;

        let samples = x.shape()[0];
        let features = y_auto.shape()[1];
        let comb_index: Vec<_> = (0..features).combinations(2).collect();
        let mut y = Array3::<Complex64>::zeros((samples, features, features));
        for (j, mut y) in y.axis_iter_mut(Axis(1)).enumerate() {
            for (i, mut y) in y.axis_iter_mut(Axis(1)).enumerate() {
                if i == j {
                    Zip::from(&mut y)
                        .and(&y_auto.column(i))
                        .apply(|c, &xx| *c = Complex64::new(xx, 0.));
                } else if i < j {
                    let k = comb_index.iter().position(|v| *v == vec![i, j]).unwrap();
                    Zip::from(&mut y)
                        .and(&y_auto.column(i))
                        .and(&y_auto.column(j))
                        .and(&y_co.column(k))
                        .and(&y_quad.column(k))
                        .apply(|c, &xx, &yy, &xyx, &xyy| {
                            *c = Complex64::new(0.5 * (xyx - xx - yy), -0.5 * (xyy - xx - yy))
                        });
                } else {
                    let k = comb_index.iter().position(|v| *v == vec![j, i]).unwrap();
                    Zip::from(&mut y)
                        .and(&y_auto.column(i))
                        .and(&y_auto.column(j))
                        .and(&y_co.column(k))
                        .and(&y_quad.column(k))
                        .apply(|c, &xx, &yy, &xyx, &xyy| {
                            *c = Complex64::new(0.5 * (xyx - xx - yy), 0.5 * (xyy - xx - yy))
                        });
                }
            }
        }

        Ok(y)
    }
}
