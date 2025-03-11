# pysimfcs
Python simulation and analysis tools for fluorescence correlation spectroscopy and flavors thereof.

This tool was inspired by SimFCS--the long standing software package developed by Enrico Gratton at the LFD (https://www.lfd.uci.edu).  The codebase attempts to reproduce the ImageJ/Fiji plugins in my Jay_Plugins package but with python code and where possible utilizing numpy and scipy libraries.

This is a work in progress so expect bugs and changes for a while.

### Organization
Various jupyter notebooks are named according to their application.  The pysimfcs_utils.py file has the core functions for simulations.  The analysis_utils.py filie has the core functions for fitting and calculating correlation functions.  Please read the [fcs_tutorial.ipynb](tutorials/fcs_tutorial.ipynb) for background information.  Then good place to start is with [sim_fcs.ipynb](sim_fcs.ipynb) and then [fit_correlations.ipynb](fit_correlations.ipynb) to fit that data.  Both of those notebooks should run easily on the free instances in google colab.  Note that you need to save the desired files from each google colab run from the files tab on the left and then upload them into the next run if they are needed for things like analysis.

### Setup
I have made an effort to rely only on a few libraries for these codes.  Those include numpy, pandas, scipy, tqdm, and matplotlib.  Reasonably up to date versions of any of those libraries should work.  If you are new to python, I would recommend starting with Jupyter Desktop: https://github.com/jupyterlab/jupyterlab-desktop.
