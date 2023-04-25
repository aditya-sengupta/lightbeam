# lightbeam
Simulate light through weakly guiding waveguides using the finite-differences beam propagation method on an adaptive mesh.

## installation
The master branch includes the raw files of the beam propagation code. For a packaged version of the code, you can use the "package" branch. Just run <pip install git+https://github.com/jw-lin/lightbeam.git@package>. You will also need the following packages: NumPy, SciPy, Matplotlib, Numba, and Numexpr.

## getting started
A jupyter notebook tutorial is provided in the <tutorial> folder. For Python script examples, check out run_bpm_example.py and the associated config file config_example.py.

## acknowledgements
This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant DGE-203483, as well as by the National Science Foundation under Grant 2109232.

## Aditya's changelog (to be removed before merge)
- refactor into lightbeam (src) and scripts
- removal or splitting off into a new script of any `if __name__ == "__main__"` sections of files in lightbeam
- deletion of screens.py, Zernike index utilities, the function `timeit`, custom progress bars in favor of existing functionality in hcipy, tqdm, and ipython
- Julia version of `tri_solve_vec`