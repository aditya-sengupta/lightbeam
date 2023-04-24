use pyo3::prelude::*;
extern crate ndarray;
extern crate numpy;
extern crate ndarray_linalg;
use crate::numpy::IntoPyArray;
use ndarray_linalg::solve::Inverse;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyRuntimeError, pymodule, types::PyModule, PyResult, Python};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn inv<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, xi: PyArray2<f64>) -> PyResult<&'py PyArray2<f64>> {
    let x = x.as_array();
    let y = x
        .inv()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(y.into_pyarray(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn lightbeam(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(inv, m)?)?;
    Ok(())
}