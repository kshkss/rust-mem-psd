use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

mod berg_c64;
mod berg_f64;

pub mod smartcore;

#[pymodule]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "reflection_coeff_f64")]
    fn reflection_coeff_f64<'py>(
        py: Python<'py>,
        max_lag: usize,
        y: PyReadonlyArray1<f64>,
    ) -> &'py PyArray2<f64> {
        let y = y.as_array();
        berg_f64::reflection_coeff(max_lag, &y).into_pyarray(py)
    }

    #[pyfn(m, "variance_f64")]
    fn variance_f64<'py>(
        py: Python<'py>,
        gamma: PyReadonlyArray1<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<f64> {
        let gamma = gamma.as_array();
        let y = y.as_array();
        berg_f64::variance(&gamma, &y).into_pyarray(py)
    }

    #[pyfn(m, "power_spector_f64")]
    fn power_spector_f64<'py>(
        py: Python<'py>,
        dt: f64,
        p: f64,
        gamma: PyReadonlyArray1<f64>,
        q: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<f64> {
        let gamma = gamma.as_array();
        let q = q.as_array();
        berg_f64::power_spector(dt, p, &gamma, &q).into_pyarray(py)
    }

    #[pyfn(m, "reflection_coeff_c64")]
    fn reflection_coeff_c64<'py>(
        py: Python<'py>,
        max_lag: usize,
        y: PyReadonlyArray1<Complex64>,
    ) -> &'py PyArray2<Complex64> {
        let y = y.as_array();
        berg_c64::reflection_coeff(max_lag, &y).into_pyarray(py)
    }

    #[pyfn(m, "variance_c64")]
    fn variance_c64<'py>(
        py: Python<'py>,
        gamma: PyReadonlyArray1<Complex64>,
        y: PyReadonlyArray1<Complex64>,
    ) -> &'py PyArray1<f64> {
        let gamma = gamma.as_array();
        let y = y.as_array();
        berg_c64::variance(&gamma, &y).into_pyarray(py)
    }

    #[pyfn(m, "power_spector_c64")]
    fn power_spector_c64<'py>(
        py: Python<'py>,
        dt: f64,
        p: f64,
        gamma: PyReadonlyArray1<Complex64>,
        q: PyReadonlyArray1<f64>,
    ) -> &'py PyArray1<f64> {
        let gamma = gamma.as_array();
        let q = q.as_array();
        berg_c64::power_spector(dt, p, &gamma, &q).into_pyarray(py)
    }

    Ok(())
}
