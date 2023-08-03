use pyo3::Py;
use pyo3::{
    pymodule, types::PyModule,
    PyResult, Python
};
use pyo3::ffi::PyObject;
use numpy::{PyReadonlyArray1,PyReadonlyArray2,PyReadwriteArray2,Complex64,ToPyArray,PyArray};
use ndarray::{Array,Zip,Axis,Dim,s};
use ndarray::parallel::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn lightbeamrs(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn tri_solve_vec(
        a: PyReadonlyArray2<'_, Complex64>, b: PyReadonlyArray2<'_, Complex64>, c: PyReadonlyArray2<'_, Complex64>, r: PyReadonlyArray2<'_, Complex64>,
        mut g: PyReadwriteArray2<'_, Complex64>, mut u: PyReadwriteArray2<'_, Complex64>
    ) {    
        let a = a.as_array();
        let b = b.as_array();
        let c = c.as_array();
        let r = r.as_array();
        let mut g = g.as_array_mut();
        let mut u = u.as_array_mut();

        let n = a.shape()[0] as i32;

        Zip::from(a.axis_iter(Axis(1))).and(b.axis_iter(Axis(1))).and(c.axis_iter(Axis(1))).and(r.axis_iter(Axis(1))).and(g.axis_iter_mut(Axis(1))).and(u.axis_iter_mut(Axis(1))).into_par_iter()
            .for_each(|(ar, br, cr, rr, mut gr, mut ur)| {
                let mut beta = br[0];
                ur[0] = rr[0] / beta;
                
                (1..n as usize).for_each(|j|  {
                    gr[j] = cr[j-1] / beta;
                    beta = br[j] - (ar[j] * gr[j]);
                    ur[j] = (rr[j] - ar[j]*ur[j-1])/beta;
                });
    
                (0..(n - 1) as i32).for_each(|j| {
                    let k = (n - 2 - j) as usize;
                    ur[k] = ur[k] - gr[k+1] * ur[k+1] ;
                });
            }); 
    }

    #[pyfn(m)]
    fn _arc(py: Python, x: PyReadonlyArray1<'_, f64>, y0: PyReadonlyArray1<'_, f64>, y1: PyReadonlyArray1<'_, f64>, r: f64) -> PyResult<Py<PyArray<f64, Dim<[usize; 1]>>>> {
        let x = x.as_array();
        let y0 = y0.as_array();
        let y1 = y1.as_array();
        let res = Zip::from(x).and(y0).and(y1)
            .par_map_collect(|xe, y0e, y1e| 0.5 * r * r * (y1e.atan2(*xe) - y0e.atan2(*xe)));

        Ok(res.to_pyarray(py).to_owned())
    }

    #[pyfn(m)]
    fn _chord(py: Python, x: PyReadonlyArray1<'_, f64>, y0: PyReadonlyArray1<'_, f64>, y1: PyReadonlyArray1<'_, f64>) -> PyResult<Py<PyArray<f64, Dim<[usize; 1]>>>> {
        let x = x.as_array().to_owned();
        let y0 = y0.as_array().to_owned();
        let y1 = y1.as_array().to_owned();
        let res = 0.5 * x * (y1 - y0);

        Ok(res.to_pyarray(py).to_owned())
    }

    /*#[pyfn(m)]
    fn where_zero(py: Python, x: PyReadonlyArray1<'_, f64>) -> PyResult<Py<PyArray<bool, Dim<[usize; 1]>>>> {
        let x = x.as_array().to_owned();
        let res = x.mapv(|x| x == 0.0);
        Ok(res.to_pyarray(py).to_owned())
    }*/

    /*#[pyfn(m)]
    /* 
    Compute the area of intersection between a triangle and a circle.
    The circle is centered at the origin and has a radius of r.  The
    triangle has verticies at the origin and at (x,y0) and (x,y1).
    This is a signed area.  The path is traversed from y0 to y1.  If
    this path takes you clockwise the area will be negative.
     */
    fn _oneside(py: Python, x: PyReadonlyArray1<'_, f64>, y0: PyReadonlyArray1<'_, f64>, y1: PyReadonlyArray1<'_, f64>, r: f64) -> PyResult<Py<PyArray<f64, Dim<[usize; 1]>>>> {
        let x = x.as_array();
        let y0 = y0.as_array();
        let y1 = y1.as_array();

        if x.mapv(|x| x == 0.0).all() {
            return Ok(x.to_pyarray(py).to_owned());
        }

        let sx = x.raw_dim();
        let mut ans = Array::zeros(sx);
        let yh = Array::zeros(sx);
        let to = s!(x.mapv(|x| x.abs() >= r));
        let ti = s!(x.mapv(|x| x.abs() < r));
        if to.len() > 0 {
            ans[to] = _arc(x[to], y0[to], y1[to], r);
        }
        if to.len() == 0 {
            return ans;
        }

        yh[ti] = sqrt(r**2 - x[ti]**2);

        let i = (slice!(y0 <= -yh) & ti);
        if i.len() > 0 {
            let j = (slice!(y1 <= -yh) & i);
            if j.len() > 0 {
                ans[j] = _arc(x[j], y0[j], y1[j], r);
            }
            j = (slice!(y1 > -yh) & slice!(y1 <= yh) & i);
            if j.len() > 0 {
                ans[j] = _arc(x[j], y0[j], -yh[j], r) + _chord(x[j], -yh[j], y1[j])
            }
            j = (slice!(y1 > yh) & i);
            if j.len() > 0 {
                ans[j] = _arc(x[j], y0[j], -yh[j], r) + _chord(x[j], -yh[j], yh[j]) + _arc(x[j], yh[j], y1[j], r)
            }
        }
        i = (slice!(y0 > -yh) & slice!(y0 < yh) & ti);
        if i.len() > 0 {
            j = (slice!(y1 <= -yh) & i);
            if j.len() > 0 {
                ans[j] = _chord(x[j], y0[j], -yh[j]) + _arc(x[j], -yh[j], y1[j], r)
            }

            j = (s!(y1 > -yh) & s!(y1 <= yh) & i);
            if j.len() > 0 {
                ans[j] = _chord(x[j], y0[j], y1[j])
            }

            j = (s!(y1 > yh) & i);
            if j.len() > 0 {
                ans[j] = _chord(x[j], y0[j], yh[j]) + _arc(x[j], yh[j], y1[j], r)
            }
        }
        i = ((y0 >= yh) & ti);
        if i.len() > 0 {
            j = ((y1 <= -yh) & i);
            if j.len() > 0 {
                ans[j] = _arc(x[j], y0[j], yh[j], r) + _chord(x[j], yh[j], -yh[j]) + _arc(x[j], -yh[j], y1[j], r)
            }

            j = ((y1 > -yh) & (y1 <= yh) & i);
            if j.len() > 0 {
                ans[j] = _arc(x[j], y0[j], yh[j], r) + _chord(x[j], yh[j], y1[j])
            }

            j = ((y1 > yh) & i);
            if j.len() > 0 {
                ans[j] = _arc(x[j], y0[j], y1[j], r)
            }
        }
        return ans
    }*/
    
    Ok(())
}