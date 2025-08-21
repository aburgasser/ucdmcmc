# UCDMCMC
 Markov Chain Monte Carlo (MCMC) fitting code for low-temperature stars, brown dwarfs ande extrasolar planet spectra, tuned particularly to the near-infrared.

## INSTALLATION NOTES

`ucdmcmc` can be installed from pip:

	pip install ucdmcmc

or from git:

	git clone
	cd ucdmcmc
	python -m setup.py install


It is recommended that you install in a conda environment to ensure the dependencies do not conflict with your own installation

	conda create -n ucdmcmc python=3.13
	conda activate ucdmcmc
	pip install ucdmcmc

A check that this worked is that you can import `ucdmcmc` into python/jupyter noteobook, and that the `ucdmcmc.MODEL_FOLDER` points to the models folder that was downloaded

`ucdmcmc` uses the following extenal packages:
* `astropy`
* `astroquery`
* `corner`
* `emcee`
* `matplotlib`
* `numpy<2.0`
* `pandas`
* `tables`
* `requests`
* `scipy`
* `statsmodels`
* `tqdm`

### Optionally install SPLAT

To generate new model sets using the built-in `generateModels()` function, you will need to install `SPLAT` (note: this is not necessary for the other functionality in this code). `SPLAT` is not automatically installed on setup. The instructions are essentially the same:

	git clone https://github.com/aburgasser/splat.git
	cd splat
	python -m pip install .

See https://github.com/aburgasser/splat for additional instructions

## Models

`ucdmcmc` comes with a starter set of models that play nicely with the code. An extended set can be downloaded from https://spexarchive.coolstarlab.ucsd.edu/ucdmcmc/

## Spectra

`ucdmcmc` comes with a starter set of spectra for the following instruments:
* EUCLID: TBD
* NIR: TRAPPIST1 spectrum from Davoudi et al. (2024) https://ui.adsabs.harvard.edu/abs/2024ApJ...970L...4D/abstract
* SPEX-PRISM: 2MASS J0559-1404 from Burgasser et al. (2006) https://ui.adsabs.harvard.edu/abs/2006ApJ...637.1067B/abstract
* JWST-NIRSPEC-PRISM: UNCOVER 33436 from Burgasser et al. (2024) https://ui.adsabs.harvard.edu/abs/2024ApJ...962..177B/abstract
* JWST-NIRSPEC-G395H: TBD
* JWST-MIRI-LRS: TBD 
* JWST-NIRSPEC-MIRI: Combined NIRSpec/PRISM and MIRI/LRS of SDSS J1624+0029 from Beiler et al. (2024) https://ui.adsabs.harvard.edu/abs/2024arXiv240708518B/abstract

## Usage

[TBD examples]

## Citing the code

[TBD]

`ucdmcmc` and its antecedents has been used in the following publications:

* Burgasser et al. (2024): https://ui.adsabs.harvard.edu/abs/2024ApJ...962..177B/abstract
* Lueber & Burgasser (2025): https://ui.adsabs.harvard.edu/abs/2025ApJ...988...31L/abstract

