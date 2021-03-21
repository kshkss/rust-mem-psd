use ndarray::s;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayView1;
use ndarray::Zip;

use num_complex::Complex64;
use num_traits::identities::One;
use num_traits::identities::Zero;

pub fn reflection_coeff(max_lag: usize, y: &ArrayView1<Complex64>) -> Array2<Complex64> {
    let mut gamma = Array2::<Complex64>::zeros((max_lag + 1, max_lag + 1));
    gamma[[0, 0]] = Complex64::one();

    for lag in 1..max_lag + 1 {
        let mut a = Complex64::zero();
        let mut b = Complex64::zero();

        let ac = gamma.slice(s![lag - 1, 0..lag]).map(|x| x.conj());
        let bc = gamma.slice(s![lag - 1, 0..lag;-1]);
        for n in 0..y.shape()[0] - lag {
            let forward = (&ac * &y.slice(s![n..n + lag])).sum();
            let backward = (&bc * &y.slice(s![n + 1..n + 1 + lag])).sum();
            a += forward.conj() * backward;
            b += forward.re.powf(2.)
                + forward.im.powf(2.)
                + backward.re.powf(2.)
                + backward.im.powf(2.);
        }
        gamma[[lag, lag]] = -2. * a / b;
        gamma[[lag, 0]] = Complex64::one();

        for k in 1..lag {
            gamma[[lag, k]] =
                gamma[[lag - 1, k]] + gamma[[lag, lag]] * gamma[[lag - 1, lag - k]].conj()
        }
    }
    gamma
}

pub fn variance(gamma: &ArrayView1<Complex64>, y: &ArrayView1<Complex64>) -> Array1<f64> {
    let mut p = Array1::<f64>::zeros(gamma.shape()[0]);
    p[0] = y.map(|x| x.re.powf(2.) + x.im.powf(2.)).mean().unwrap();

    for i in 1..p.shape()[0] {
        p[i] = p[i - 1] * (1. - (gamma[i].re.powf(2.) + gamma[i].im.powf(2.)));
    }

    p
}

pub fn power_spector(
    dt: f64,
    p: f64,
    gamma: &ArrayView1<Complex64>,
    qs: &ArrayView1<f64>,
) -> Array1<f64> {
    let ks = Array1::<f64>::linspace(1., gamma.shape()[0] as f64 - 1., gamma.shape()[0] - 1);
    let psd = qs.map(|qk| {
        let a = Zip::from(&ks)
            .and(gamma.slice(s![1..]))
            .fold(Complex64::one(), |acc, &k, &gk| {
                acc + gk * Complex64::from_polar(1., -2. * std::f64::consts::PI * qk * dt * k)
            });
        dt * p / (a.re.powf(2.) + a.im.powf(2.))
    });

    psd
}

pub fn berg(lag: usize, delta: f64, y: &ArrayView1<Complex64>, q: &ArrayView1<f64>) -> Array1<f64> {
    let gamma = reflection_coeff(lag, &y);
    let p = variance(&gamma.diag(), &y);
    let psd = power_spector(delta, p[lag], &gamma.slice(s![lag, ..]), &q);

    psd
}
