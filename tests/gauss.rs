use mem_psd::smartcore::*;
use ndarray::s;
use ndarray::stack;
use ndarray::stack_new_axis;
use ndarray::Array1;
use ndarray::Axis;
use ndarray_npy::write_npy;
use num_complex::Complex64;
use smartcore::api::Predictor;
use smartcore::api::UnsupervisedEstimator;

#[test]
fn exp_f64() {
    let xs = Array1::<f64>::linspace(-10., 10., 1000);
    let ys = xs.map(|x| (-1.0 * x.powf(2.)).exp());

    let model =
        Psd1d::<f64>::fit(&ys.view().insert_axis(Axis(0)), PsdParameters::new(361)).unwrap();
    let model = model.with_delta(xs[1] - xs[0]);
    // Compute PSD
    let q = Array1::<f64>::linspace(0., 3., 100);
    let f = model.predict(&q.view()).unwrap();

    // Save result
    assert!(write_npy("gauss.npy", &stack![Axis(1), xs, ys]).is_ok());
    assert!(write_npy(
        "gauss_result.npy",
        &stack_new_axis![Axis(1), q, f.slice(s![0, ..])]
    )
    .is_ok());
}

#[test]
fn gauss_c64() {
    // Prepare dataset
    let xs = Array1::<f64>::linspace(-10., 10., 1000);
    let ys = xs.map(|x| Complex64::new((-1.0 * x.powf(2.)).exp(), 0.));
    let model =
        Psd1d::<Complex64>::fit(&ys.view().insert_axis(Axis(0)), PsdParameters::new(361)).unwrap();
    let model = model.with_delta(xs[1] - xs[0]);

    // Compute PSD
    let q = Array1::<f64>::linspace(0., 3., 100);
    let f = model.predict(&q.view()).unwrap();

    // Save result
    assert!(write_npy("gauss_c.npy", &stack![Axis(1), xs, ys.map(|x| x.re)]).is_ok());
    assert!(write_npy(
        "gauss_result_c.npy",
        &stack_new_axis![Axis(1), q, f.slice(s![0, ..])]
    )
    .is_ok());
}
