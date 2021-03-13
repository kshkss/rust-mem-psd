use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::Zip;

use num_complex::Complex64;
use num_traits::identities::One;

pub fn reflection_coeff(max_lag: usize, y: &ArrayView1<f64>) -> Array2<f64> {
    let mut gamma = Array2::<f64>::zeros((max_lag + 1, max_lag + 1));
    gamma[[0, 0]] = 1.;

    for lag in 1..max_lag + 1 {
        let mut a = 0.;
        let mut b = 0.;

        let ac = gamma.slice(s![lag - 1, 0..lag]);
        let bc = gamma.slice(s![lag - 1, 0..lag;-1]);
        for n in 0..y.shape()[0] - lag {
            let forward = (&ac * &y.slice(s![n..n + lag])).sum();
            let backward = (&bc * &y.slice(s![n + 1..n + 1 + lag])).sum();
            a += forward * backward;
            b += forward.powf(2.) + backward.powf(2.);
        }
        gamma[[lag, lag]] = -2. * a / b;
        gamma[[lag, 0]] = 1.;

        for k in 1..lag {
            gamma[[lag, k]] = gamma[[lag - 1, k]] + gamma[[lag, lag]] * gamma[[lag - 1, lag - k]]
        }
    }
    gamma
}

pub fn variance(gamma: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
    let mut p = Array1::<f64>::zeros(gamma.shape()[0]);
    p[0] = y.map(|x| x.powf(2.)).mean().unwrap();

    for i in 1..p.shape()[0] {
        p[i] = p[i - 1] * (1. - gamma[i].powf(2.));
    }

    p
}

pub fn power_spector(
    dt: f64,
    p: f64,
    gamma: &ArrayView1<f64>,
    qs: &ArrayView1<f64>,
) -> Array1<f64> {
    let ks = Array1::<f64>::linspace(1., gamma.shape()[0] as f64 - 1., gamma.shape()[0] - 1);
    let psd = qs.map(|qk| {
        let a = Zip::from(&ks)
            .and(gamma.slice(s![1..]))
            .fold(Complex64::one(), |acc, &k, &gk| {
                acc + Complex64::from_polar(gk, -2. * std::f64::consts::PI * qk * dt * k)
            });
        dt * p / (a.re.powf(2.) + a.im.powf(2.))
    });

    psd
}

pub fn berg(lag: usize, delta: f64, y: &ArrayView1<f64>, q: &ArrayView1<f64>) -> Array1<f64> {
    let gamma = reflection_coeff(lag, &y);
    let p = variance(&gamma.diag(), &y);
    let psd = power_spector(delta, p[lag], &gamma.slice(s![lag, ..]), &q);

    psd
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use ndarray::s;
    use ndarray::stack;
    use ndarray::Array1;
    use ndarray::Axis;
    use ndarray_npy::write_npy;

    use num_complex::Complex64;

    #[test]
    fn complex_array_test() {
        let f = Array1::<Complex64>::ones(1);
        assert_eq!(f[0], Complex64::new(1., 0.));
    }

    #[test]
    fn slice_reverse_test() {
        let a = Array1::<f64>::linspace(0., 9., 10);
        let b = arr1(&[9., 8., 7.]);
        let c = a.slice(s![7..10;-1]);
        assert_eq!(c, b);
    }

    #[test]
    fn sgn() {
        // Prepare dataset
        let xs = Array1::<f64>::linspace(-5., 4.99, 1000);
        let mut ys = Array1::<f64>::zeros(1000);
        for (x, y) in xs.iter().zip(ys.iter_mut()) {
            if *x > 0. {
                *y = 1.;
            } else {
                *y = -1.;
            }
        }

        // Compute PSD
        let q = Array1::<f64>::logspace(10., -5., 2., 100);
        let f = berg(100, xs[1] - xs[0], &ys.view(), &q.view());

        // Save result
        assert!(write_npy("sgn.npy", &stack![Axis(1), xs, ys]).is_ok());
        assert!(write_npy("sgn_result.npy", &stack![Axis(1), q, f]).is_ok());
    }

    #[test]
    fn gauss() {
        // Prepare dataset
        let xs = Array1::<f64>::linspace(-10., 10., 1000);
        let ys = xs.map(|x| (-1.0 * x.powf(2.)).exp());

        // Compute PSD
        let q = Array1::<f64>::linspace(0., 3., 100);
        let f = berg(361, xs[1] - xs[0], &ys.view(), &q.view());

        // Save result
        assert!(write_npy("gauss.npy", &stack![Axis(1), xs, ys]).is_ok());
        assert!(write_npy("gauss_result.npy", &stack![Axis(1), q, f]).is_ok());
    }

    #[test]
    fn exp() {
        // Prepare dataset
        let xs = Array1::<f64>::linspace(-20., 20., 2000);
        let ys = xs.map(|x| (-1.0 * x.abs()).exp());

        // Compute PSD
        let q = Array1::<f64>::linspace(-1., 1.5, 100);
        let f = berg(1000, xs[1] - xs[0], &ys.view(), &q.view());

        // Save result
        assert!(write_npy("exp.npy", &stack![Axis(1), xs, ys]).is_ok());
        assert!(write_npy("exp_result.npy", &stack![Axis(1), q, f]).is_ok());
    }
}
