use pyo3::{
    Py,
    pymodule, types::PyModule,
    PyResult, Python, pyclass, pymethods
};
use numpy::{PyReadonlyArray1,PyReadonlyArray2,PyArray,PyReadwriteArray2,Complex64, ToPyArray};
use ndarray::{Zip,Axis,Array1,Dim};
use ndarray::parallel::prelude::*;


/* a collection of functions for antialiasing circles'''

## calculate circle-square overlap.

# original code in pixwt.c by Marc Buie
# ported to pixwt.pro (IDL) by Doug Loucks, Lowell Observatory, 1992 Sep
# subsequently ported to python by Michael Fitzgerald,
# LLNL, fitzgerald15@llnl.gov, 2007-10-16
# (hopefully finally) ported to python-callable Rust by Aditya Sengupta,
# UCSC, adityars@ucsself.edu, 2023 Aug

### Marc Buie, you are my hero
*/

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

    fn arc(x: f64, y0: f64, y1: f64, r: f64) -> f64 {
        0.5 * r * r * (y1.atan2(x) - y0.atan2(x))
    }

    fn chord(x: f64, y0: f64, y1: f64) -> f64 {
        0.5 * x * (y1 - y0)
    }

    fn oneside_one(x: f64, y0: f64, y1: f64, r: f64) -> f64 {
        if x == 0.0 { return 0.0 }
        if x.abs() >= r { return arc(x, y0, y1, r) }
        let yh = (r * r - x * x).sqrt();
        if y0 <= -yh {
            if y1 <= -yh { return arc(x, y0, y1, r) }
            else if (y1 > -yh) & (y1 <= yh) { return arc(x, y0, -yh, r) + chord(x, -yh, y1) }
            else { return arc(x, y0, -yh, r) + chord(x, -yh, yh) + arc(x, yh, y1, r) }
        } else if (y0 > -yh) & (y0 < yh) {
            if y1 <= -yh { return chord(x, y0, -yh) + arc(x, -yh, y1, r) }
            else if (y1 > -yh) & (y1 <= yh) { return chord(x, y0, y1) }
            else { return chord(x, y0, yh) + arc(x, yh, y1, r) }
        } else {
            if y1 <= -yh { return arc(x, y0, yh, r) + chord(x, yh, -yh) + arc(x, -yh, y1, r) }
            else if (y1 > -yh) & (y1 <= yh) { return arc(x, y0, yh, r) + chord(x, yh, y1) }
            else { return arc(x, y0, y1, r) }
        }
    }

    fn oneside(x: Array1<f64>, y0: Array1<f64>, y1: Array1<f64>, r: f64) -> Array1<f64> {
        let mut onesides = Array1::zeros(x.len());
        Zip::from(&mut onesides).and(&x).and(&y0).and(&y1).into_par_iter().for_each(|(o, xi, y0i, y1i)| *o = oneside_one(*xi, *y0i, *y1i, r));
        return onesides;
    }

    /*struct CircleRectangle {
        r: f64,
        x0: Array1<f64>,
        x1: Array1<f64>,
        y0: Array1<f64>,
        y1: Array1<f64>
    }

    impl CircleRectangle {
        fn intarea(&self) -> Array1<f64> {
            return oneside(&self.x1, &self.y0, &self.y1, self.r) + 
                   oneside(&self.y1, &(-self.x1.clone()), &(-self.x0.clone()), self.r) + 
                   oneside(&(-self.x0.clone()), &(-self.y1.clone()), &(-self.y0.clone()), self.r) + 
                   oneside(&(-self.y0.clone()), &self.x0, &self.x1, self.r);
        }
    }*/

    // pixwt is this but with x0 = x - 0.5, x1 = x + 0.5 and the same for y
    #[pyfn(m)]
    fn intarea(py: Python, xc: f64, yc: f64, r: f64, x0: PyReadonlyArray1<'_, f64>, x1: PyReadonlyArray1<'_, f64>, y0: PyReadonlyArray1<'_, f64>, y1: PyReadonlyArray1<'_, f64>) -> Py<PyArray<f64, Dim<[usize; 1]>>> {
        let x0 = x0.as_array().to_owned() - xc;
        let y0 = y0.as_array().to_owned() - yc;
        let x1 = x1.as_array().to_owned() - xc;
        let y1 = y1.as_array().to_owned() - yc;
        let res = oneside(x1.clone(), y0.clone(), y1.clone(), r) + 
        oneside(y1.clone(), -x1.clone(), -x0.clone(), r) + 
        oneside(-x0.clone(), -y1, -y0.clone(), r) + 
        oneside(-y0, x0, x1, r);
        return res.to_pyarray(py).to_owned();
    }
    
    Ok(())
}