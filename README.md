# lightbeam
Simulate light through weakly guiding waveguides using the finite-differences beam propagation method on an adaptive mesh.

## installation
The master branch includes the raw files of the beam propagation code. For a packaged version of the code, you can use the "package" branch. Just run <pip install git+https://github.com/jw-lin/lightbeam.git@package>. You will also need the following packages: NumPy, SciPy, Matplotlib, Numba, and Numexpr.

## getting started
A jupyter notebook tutorial is provided in the <tutorial> folder. For Python script examples, check out run_bpm_example.py and the associated config file config_example.py.

## acknowledgements
This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant DGE-203483, as well as by the National Science Foundation under Grant 2109232.

## Aditya's changelog (to be removed before merge, if that happens)
- refactor into lightbeam (src) and scripts
- removal or splitting off into a new script of any `if __name__ == "__main__"` sections of files in lightbeam
- deletion of screens.py, Zernike index utilities, custom progress bars, other misc things in favor of existing functionality in hcipy, tqdm, and ipython
- new `Lantern` class with checks for if the SMFs only support one mode and if the MMF supports as many modes as there are SMFs
- refactor of `Prop3D.prop2end`: combine `prop2end` and `prop2end_uniform`, turn internal variables into class variables
- move meshing logic into `Prop3D._remesh` and propagation setup logic into `Prop3D._prop_setup`
- experimental lib.rs stuff to replace `tri_solve_vec` and eventually some of the matrix allocations and remeshing