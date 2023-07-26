use pyo3::{
    pymodule, types::PyModule,
    PyResult, Python
};
use numpy::{PyReadonlyArray2,PyReadwriteArray2,Complex64,Ix2};
use ndarray::Zip;

/// A Python module implemented in Rust.
#[pymodule]
fn lightbeamrs(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn zip_testing(
        a: PyReadonlyArray2<'_, Complex64>,
        mut b: PyReadwriteArray2<'_, Complex64>
    ) {
        let mut b_arr = b.as_array_mut();
        Zip::from(b_arr.rows_mut())
            .and(a.as_array().rows())
            .for_each(|mut b_row, a_row| { 
            b_row.assign(&a_row);
        });
    }

    #[pyfn(m)]
    fn tri_solve_vec(
        n: i32, a: PyReadonlyArray2<'_, Complex64>, b: PyReadonlyArray2<'_, Complex64>, c: PyReadonlyArray2<'_, Complex64>, r: PyReadonlyArray2<'_, Complex64>,
        mut g: PyReadwriteArray2<'_, Complex64>, mut u: PyReadwriteArray2<'_, Complex64>
    ) {    
        let a = a.as_array();
        let b = b.as_array();
        let c = c.as_array();
        let r = r.as_array();
        let mut g = g.as_array_mut();
        let mut u = u.as_array_mut();
        (0..n as usize).for_each(|i|  {
            let j0 = Ix2(i,0);
            let mut beta = *b.get(j0).unwrap();
            u[j0] = r.get(j0).unwrap() / beta;
            
            (1..n as usize).for_each(|j|  {
                let ij = Ix2(i,j);
                let ijm1 = Ix2(i,j-1);
                g[Ix2(i,j)] = c.get(ijm1).unwrap() / beta;
                beta = b.get(ij).unwrap() - (a.get(ij).unwrap() * g.get(ij).unwrap());
                u[Ix2(i,j)] = (r.get(ij).unwrap() - a.get(ij).unwrap()*u[ijm1])/beta;
            });

            (0..(n - 1) as i32).for_each(|j| {
                let k = (n - 2 - j) as usize;
                let ik = Ix2(i,k);
                let ikp1 = Ix2(i,k+1);
                u[Ix2(i,k)] = u.get(ik).unwrap() - g.get(ikp1).unwrap() * u.get(ikp1).unwrap() ;
            });
        }); 
    }
    
    Ok(())
}