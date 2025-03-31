# pysimfcs
Python simulation and analysis tools for fluorescence correlation spectroscopy and flavors thereof.

This tool was inspired by SimFCS--the long standing software package developed by Enrico Gratton at the LFD (https://www.lfd.uci.edu).  The codebase attempts to reproduce the ImageJ/Fiji plugins in my Jay_Plugins package but with python code and where possible utilizing numpy and scipy libraries.

This is a work in progress so expect bugs and changes for a while.

### Organization
Various jupyter notebooks are named according to their application.  The pysimfcs_utils.py file has the core functions for simulations.  The analysis_utils.py file has the core functions for fitting and calculating correlation functions.  Please read the [fcs_tutorial.ipynb](tutorials/fcs_tutorial.ipynb) for background information and references.  Then good place to start is with [sim_fcs.ipynb](sim_fcs.ipynb) and then [fit_correlations.ipynb](fit_correlations.ipynb) to fit that data.  Both of those notebooks should run easily on the free instances in google colab.  Note that you need to save the desired files from each google colab run from the files tab on the left and then upload them into the next run if they are needed for things like analysis.  Here is a list of simulation and analysis tools:

* [sim_fcs.ipynb](sim_fcs.ipynb): Simulates single point multichannel FCS data.
* [fit_correlations.ipynb](fit_correlations.ipynb): Fits correlation functions with non-linear least squares.
* [fit_pch.ipynb](fit_pch.ipynb): Calculates and fits photon counting histograms with non-linear least squares.
* [sim_rics.ipynb](sim_rics.ipynb): Simulates multichannel raster scanning image correlation (RICS) data.
* [fit_rics.ipynb](fit_rics.ipynb): Fits horizontal and vertical RICS lines.
* [sim_movie.ipynb](sim_movie.ipynb): Simulates camera movie data (in photon counting mode).
* [n_and_b.ipynb](n_and_b.ipynb): Illustrates how to calculate N and B histograms.
* [gate_n_and_b.ipynb](gate_n_and_b.ipynb): Illustrates how to gate and highlight N and B histograms.
* [sim_movie_analog.ipynb](sim_movie_analog.ipynb): Simulates camera movie data with an analog detector.
* [n_and_b_analog.ipynb](n_and_b_analog.ipynb): Illustrates how to calculate N and B histograms and estimate the analog S parameter.
* [carpet_stics_pcf.ipynb](carpet_stics_pcf.ipynb): Illustrates how to calculate spatiotemporal image correlation (STICS) and pair correlation functions (pCFs) from simulated camera data.
* [fit_pch2d.ipynb](fit_pch2d.ipynb): Fit 2D PCH multicolor photon counting histograms.

### Setup
I have made an effort to rely only on a few libraries for these codes.  Those include:
* numpy: For general array and matrix manipulation and FFTs
* pandas: For table manipulation and io
* scipy: For fitting and smoothing images and special math functions
* tifffile: For reading and writing multidimensional tif images
* tqdm: To display progress bars
* matplotlib: For plotting and image display

Reasonably up to date versions of any of those libraries should work.  If you are new to python, I would recommend starting with Jupyter Desktop: https://github.com/jupyterlab/jupyterlab-desktop.
