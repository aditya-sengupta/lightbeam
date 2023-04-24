use numpy::{PyArray, IntoPyArray};
use pyo3::Python;

Python::with_gil(|py| {
    let py_array = vec![1, 2, 3].into_pyarray(py);

    assert_eq!(py_array.readonly().as_slice().unwrap(), &[1, 2, 3]);

    // Array cannot be resized when its data is owned by Rust.
    unsafe {
        assert!(py_array.resize(100).is_err());
    }
});