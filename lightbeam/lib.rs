use pyo3::{
    pymodule, types::{PyModule},
    PyResult, Python,
};
use numpy::{PyReadonlyArrayDyn,PyReadwriteArrayDyn,Complex64};

/// A Python module implemented in Rust.
#[pymodule]
fn lightbeamrs(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn tri_solve_vec(
        n: i32, a: PyReadonlyArrayDyn<'_, Complex64>, b: PyReadonlyArrayDyn<'_, Complex64>, c: PyReadonlyArrayDyn<'_, Complex64>, r: PyReadonlyArrayDyn<'_, Complex64>,
        mut g: PyReadwriteArrayDyn<'_, Complex64>, mut u: PyReadwriteArrayDyn<'_, Complex64>
    ) {
        let a = a.as_array();
        let b = b.as_array();
        let c = c.as_array();
        let r = r.as_array();
        let mut g = g.as_array_mut();
        let mut u = u.as_array_mut();
        let mut beta = b[0];
        u[0] = r[0] / beta;
        
        for j in 1..n as usize {
            g[j] = c[j-1]/beta;
            beta = b[j] - a[j]*g[j];
            u[j] = (r[j] - a[j]*u[j-1])/beta;
        }

        for j in 0..(n-1) {
            let k = (n - 2 - j) as usize;
            u[k] = u[k] - g[k+1]*u[k+1];
        }
    }
    
    // m.add_function(wrap_pyfunction!(_chord, m)?)?;
    Ok(())
}