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
fn sgn_f64() {
    let xs = Array1::<f64>::linspace(-5., 4.99, 1000);
    let mut ys = Array1::<f64>::zeros(1000);
    for (x, y) in xs.iter().zip(ys.iter_mut()) {
        if *x > 0. {
            *y = 1.;
        } else {
            *y = -1.;
        }
    }
    let model =
        PSD1D::<f64>::fit(&ys.view().insert_axis(Axis(0)), PSDParameters::new(100)).unwrap();
    let model = model.with_delta(xs[1] - xs[0]);
    // Compute PSD
    let q = Array1::<f64>::logspace(10., -5., 2., 100);
    let f = model.predict(&q.view()).unwrap();

    // Save result
    assert!(write_npy("sgn.npy", &stack![Axis(1), xs, ys]).is_ok());
    assert!(write_npy(
        "sgn_result.npy",
        &stack_new_axis![Axis(1), q, f.slice(s![0, ..])]
    )
    .is_ok());
}

#[test]
fn sgn_c64() {
    // Prepare dataset
    let xs = Array1::<f64>::linspace(-5., 4.99, 1000);
    let mut ys = Array1::<Complex64>::zeros(1000);
    for (x, y) in xs.iter().zip(ys.iter_mut()) {
        if *x > 0. {
            *y = Complex64::new(1., 0.);
        } else {
            *y = Complex64::new(-1., 0.);
        }
    }
    let model =
        PSD1D::<Complex64>::fit(&ys.view().insert_axis(Axis(0)), PSDParameters::new(100)).unwrap();
    let model = model.with_delta(xs[1] - xs[0]);

    // Compute PSD
    let q = Array1::<f64>::logspace(10., -5., 2., 100);
    let f = model.predict(&q.view()).unwrap();

    // Save result
    assert!(write_npy("sgn_c.npy", &stack![Axis(1), xs, ys.map(|x| x.re)]).is_ok());
    assert!(write_npy(
        "sgn_result_c.npy",
        &stack_new_axis![Axis(1), q, f.slice(s![0, ..])]
    )
    .is_ok());
}
