use pyo3::prelude::*;
use ndarray::{array,Array1,ArrayViewD,ArrayViewMutD};
use numpy::{PyArray1,IntoPyArray,ToPyArray};

// original code in pixwt.c by Marc Buie
// ported to pixwt.pro (IDL) by Doug Loucks, Lowell Observatory, 1992 Sep
// subsequently ported to python by Michael Fitzgerald,
// LLNL, fitzgerald15@llnl.gov, 2007-10-16
// and ported to Rust for Python usage by Aditya Sengupta, UC Santa Cruz, (date)

// Marc Buie, you are my hero

#[pyfunction]
fn _arc_ind(x: f64, y0: f64, y1: f64, r: f64) -> f64 {
    /*
    Compute the area within an arc of a circle.  The arc is defined by
    the two points (x,y0) and (x,y1) in the following manner: The
    circle is of radius r and is positioned at the originp.  The origin
    and each individual point define a line which intersects the
    circle at some point.  The angle between these two points on the
    circle measured from y0 to y1 defines the sides of a wedge of the
    circle.  The area returned is the area of this wedge.  If the area
    is traversed clockwise then the area is negative, otherwise it is
    positive.
     */
    Some(0.5 * r * r * (y1.atan2(x) - y0.atan2(x))).unwrap()
}

#[pyfunction]
fn _arc(
    py: Python<'_>,
    x: Vec<f64>, y0: Vec<f64>, y1: Vec<f64>, r: f64) 
    -> PyResult<Py<PyArray1<f64>>> {

    let mut ans = x.clone();
    for i in 0..x.len() {
        ans[i] = _arc_ind(x[i], y0[i], y1[i], r);
    }
    
    Ok(Array1::from_vec(ans).to_pyarray(py).to_owned())
}

/*fn _chord_ind(x: f64, y0: f64, y1: f64) -> f64 {
    /*
    Compute the area of a triangle defined by the origin and two
    points, (x,y0) and (x,y1).  This is a signed area.  If y1 > y0
    then the area will be positive, otherwise it will be negative. 
    */
    Some(0.5 * x * (y1 - y0)).unwrap()
}

#[pyfunction]
fn _chord(
    py: Python<'_>, 
    x: Vec<f64>, y0: Vec<f64>, y1: Vec<f64>,) -> PyResult<Py<PyArray1<f64>>> {
    // let mut res = x.clone();
    Ok((0.5 * x * (y1 - y0)).to_pyarray(py).to_owned())
    //Ok(Array1::from_vec(res).to_pyarray(py).to_owned())
} */

/*#[pyfunction]
fn _oneside(x: f64, y0: f64, y1: f64, r: f64) -> PyReadonlyArray1<'static, f64> {
    Ok(())
} */

/// A Python module implemented in Rust.
#[pymodule]
fn lightbeamrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_arc, m)?)?;
    // m.add_function(wrap_pyfunction!(_chord, m)?)?;
    Ok(())
}