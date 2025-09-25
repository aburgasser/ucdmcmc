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
* `astropy`: https://www.astropy.org/
* `astroquery`: https://astroquery.readthedocs.io/en/latest/
* `corner`: https://corner.readthedocs.io/en/latest/
* `emcee`: https://emcee.readthedocs.io/en/stable/
* `matplotlib`: https://matplotlib.org/
* `numpy<2.0`: https://numpy.org/
* `pandas`: https://pandas.pydata.org/
* `(py)tables`: https://www.pytables.org/
* `requests`: https://requests.readthedocs.io/en/latest/
* `scipy`: https://scipy.org/
* `spectres`: https://spectres.readthedocs.io/en/latest/
* `statsmodels`: https://www.statsmodels.org/stable/index.html
* `tqdm`: https://tqdm.github.io/

### Optionally install SPLAT

To generate new model sets using the built-in `generateModels()` function, you will need to install `SPLAT` (note: this is not necessary for the other functionality in this code). `SPLAT` is not automatically installed on setup. The instructions are essentially the same:

	git clone https://github.com/aburgasser/splat.git
	cd splat
	python -m pip install .

See https://github.com/aburgasser/splat for additional instructions

## Models

`ucdmcmc` comes with a starter set of models that play nicely with the code. An extended set can be downloaded from https://spexarchive.coolstarlab.ucsd.edu/ucdmcmc/. These should be placed in the folder `.ucdmcmc_models` in your home directory (i.e., `/home/adam/.ucdmcmc.models`). If it doesn't already exist, this directory will be created on the first call to `ucdmcmcm`. In addition, models that exist on this website and not present in this folder will be downloaded directly when `getModelSet()`` is called. You can also generate your own set of models using the `generateModels()` function (see note above).

## Spectra

`ucdmcmc` comes with a starter set of spectra for the following instruments:
* EUCLID: TBD
* NIR: TRAPPIST1 spectrum from Davoudi et al. (2024) https://ui.adsabs.harvard.edu/abs/2024ApJ...970L...4D/abstract
* SPEX-PRISM: 2MASS J0559-1404 from Burgasser et al. (2006) https://ui.adsabs.harvard.edu/abs/2006ApJ...637.1067B/abstract
* JWST-NIRSPEC-PRISM: UNCOVER 33436 from Burgasser et al. (2024) https://ui.adsabs.harvard.edu/abs/2024ApJ...962..177B/abstract
* JWST-NIRSPEC-G395H: TBD
* JWST-MIRI-LRS: TBD 
* JWST-NIRSPEC-MIRI: Combined NIRSpec/PRISM and MIRI/LRS of SDSS J1624+0029 from Beiler et al. (2024) https://ui.adsabs.harvard.edu/abs/2024arXiv240708518B/abstract

User spectra can be read in using `ucdmcmc.Spectrum("filename")`. Files can be `.fits`, `.csv`, `.txt` (space-delimited), or `.tsv` (tab-delimited), and should have wavelength, flux, and uncertainty arrays. You can also read in these files separately and create a Spectrum object using the call `ucdmcmc.Spectrum(wave=[wave array,flux=[flux array],noise=[uncertainty array])`. See the docstring for `ucdmcmc.Spectrum` for further details.

## Usage

[TBD examples]

## Opacities

[TBD]

## Citing the code

If you use this code in your research, publications, or presentatinos, please include the following citation:

	Adam Burgasser. (2025). aburgasser/ucdmcmc (vXXX). Zenodo. https://doi.org/10.5281/zenodo.16923762

or in bibtex:

	@software{adam_burgasser_2025_16921711,
		author = {Adam Burgasser},
		doi = {10.5281/zenodo.16921711},
		month = aug,
		publisher = {Zenodo},
		title = {aburgasser/ucdmcmc},
		url = {https://doi.org/10.5281/zenodo.16921711},
		version = {vXXX},
		year = 2025,
		bdsk-url-1 = {https://doi.org/10.5281/zenodo.16921711}}

 where (vXXX) corresponds to the version used.  

`ucdmcmc` and its antecedents has been used in the following publications:

* Burgasser et al. (2024, ApJ 962, 177): https://ui.adsabs.harvard.edu/abs/2024ApJ...962..177B/abstract
* Burgasser et al. (2025, ApJ 982, 79): https://ui.adsabs.harvard.edu/abs/2025ApJ...982...79B/abstract
* Lueber & Burgasser (2025, ApJ 988, 31): https://ui.adsabs.harvard.edu/abs/2025ApJ...988...31L/abstract
* Morrissey et al. (2025, AJ, submitted)
* Burgasser et al. (2025, Science, submitted)

Please let me know if you make use of the code so we can include your publication in the list above!



