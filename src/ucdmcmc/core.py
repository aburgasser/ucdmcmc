"""

	ucdmcmc

	Package Description
	-------------------

	UCDMCMC performs spectral model fitting of cool stars, brown dwarfs, and exoplanets using predefined published grids.
	Options are available to conduct straight grid fits (best fit among individual grids) and MCMC (interpolation between grids).
	UCDMCMC makes heavy use of the SPLAT package, which must be installed separately from https://github.com/aburgasser/splat.
	Please try the provided tutorial for examples of how to use UCDMCMC routines.

	Pre-set models
	--------------
	UCDMCMC comes with the following models pre-loaded in the models/ folder:

	* atmo20 - ATMO2020 model set from Phillips et al. (2020) bibcode: 2020A%26A...637A..38P
	* atmo20pp - ATMO2020++ model set from Meisner et al. (2023) bibcode: 2023AJ....166...57M
	* btdusty16 - BT-Dusty model set from TBD bibcode: TBD
	* btsettl08 - BT-Settled model set from Allard et al. (2012) bibcode: 2012RSPTA.370.2765A
	* dback24 - Sonora Diamondback model set from Morley et al. (2024) bibcode: 2024arXiv240200758M
	* drift - Drift model set from Witte et al. (2011) bibcode: 2011A&A...529A..44W
	* elfowl24 - Sonora Elfowl model set from Mukherjee et al. (2024) bibcode: 2024arXiv240200756M
	* karalidi21 - Sonora Cholla model set from Karalidi et al. (2021) bibcode: 2021ApJ...923..269K
	* lowz - LOWZ model set from Meisner et al. (2021) bibcode: 2021ApJ...915..120M
	* sand24 - SAND model set from Alvardo et al. (2024) bibcode: 2024RNAAS...8..134A
	* sonora21 - Sonora Bobcat model set from Marley et al. (2021) bibcode: 2021ApJ...920...85M
	* tremblin15 - Model set from Tremblin et al. (2015) bibcode: 2015ApJ...804L..17T

	These are calculated for a subset of the following instruments:

	* NIR: generic low resolution NIR range, covering 0.85-2.4 micron at ma edian resolution of 442
	* SPEX-PRISM: IRTF SpeX PRISM mode, covering 0.65-2.56 micron at a median resolution of 423, using data from 
		Burgasser et al. (2006) as a template
	* JWST-NIRSPEC-PRISM: JWST NIRSPEC PRISM, covering 0.5--6 micron at a median resolution of 590, using data from 
		Burgasser et al. (2024) as a template
	* JWST-NIRSPEC-G395H: JWST NIRSPEC G395H, covering 0.5--6 micron at a median resolution of 590, using data from 
		Burgasser et al. (2024) as a template
	* JWST-MIRI-LRS: JWST MIRI LRS, covering 4.55--13.5 micron at a median resolution of XXX,
		using data from TBD as a template
	* JWST-NIRSPEC-MIRI: combination of NIRSPEC PRISM and MIRI LRS, covering 2.8-5.2 micron at a median resolution of XXX,
		using data from Beiler et al. (2024) as a template


"""

# WHAT NEEDS TO BE DONE
# - add in examples for JWST MIRI LRS and NIRSPEC G395H
# - comparison plot with x and y log scales

# MCMC model fitting code
import copy
import corner
import glob
#import h5py
import matplotlib.pyplot as plt
import numpy
import os
import pandas
from scipy.interpolate import griddata
#from scipy.optimize import minimize,curve_fit
import scipy.stats as stats
from tqdm import tqdm
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, CartesianRepresentation, CartesianDifferential, Galactic, Galactocentric
import splat
import splat.model as spmdl
import astropy.units as u
from statsmodels.stats.weightstats import DescrStatsW


#######################################################
###############   INSTALLATION NOTES  #################
#######################################################

# git clone
# cd ucdmcmc
# python -m setup.py install

# a check is that ucdmcmc.MODEL_FOLDER points to the models folder that was downloaded
# altnerately ucdmcmc.modelInfo() returns models

#######################################################
#######################################################
#################   INITIALIZATION  ###################
#######################################################
#######################################################


# code parameters
VERSION = '19 August 2025'
GITHUB_URL = 'http://www.github.com/aburgasser/ucdmcmc/'
ERROR_CHECKING = True
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(CODE_PATH,'models/')
SPECTRA_FOLDER = os.path.join(CODE_PATH,'spectra/')
MODEL_FILE_PREFIX = 'models_'
WAVE_FILE_PREFIX = 'wave_'

# defaults
DEFAULT_FLUX_UNIT = u.erg/u.s/u.cm/u.cm/u.micron
DEFAULT_FLUX_NAME = 'flux'
DEFAULT_WAVE_UNIT = u.micron
DEFAULT_WAVE_NAME = 'wave'

# baseline wavelength grid
DEFAULT_WAVE_RANGE = [0.9,2.45]
DEFAULT_RESOULTION = 300

# parameters
PARAMETER_PLOT_LABELS = {
	'teff':r'T$_{eff}$ (K)',
	'logg':r'$\log{g}$ (cm/s$^2$)',
	'z':'[M/H]',
	'enrich':r'[$\alpha$/Fe]',
	'co':'C/O',
	'kzz':r'$\log\kappa_{zz}$ (cm$^2$/s)',
	'fsed':r'$f_{sed}$',
	'cld':'Cloud Model',
	'ad':r'$\gamma$',
	'radius':r'R (R$_\odot$)',
	'chis':r'$\chi^2$',
}

# parameters
PARAMETERS = {
	'teff': {'type': float,'label': r'T$_{eff}$ (K)','fmt': '{:.0f}','step':25,},
	'logg': {'type': float,'label': r'$\log{g}$ (cm/s$^2$)','fmt': '{:.2f}','step':0.1,},
	'z': {'type': float,'label': '[M/H]','fmt': '{:.2f}','step':0.1,},
	'enrich': {'type': float,'label': r'[$\alpha$/Fe]','fmt': '{:.2f}','step':0.05,},
	'co': {'type': float,'label': 'C/O','fmt': '{:.2f}','step':0.05,},
	'kzz': {'type': float,'label': r'$\log\kappa_{zz}$ (cm$^2$/s)','fmt': '{:.2f}','step':0.25,},
	'fsed': {'type': float,'label': r'$f_{sed}$','fmt': '{:.2f}','step':0.25,},
	'cld': {'type': str,'label': 'Cloud Model','fmt': '{}','step':-99,},
	'ad': {'type': float,'label': r'$\gamma$','fmt': '{:.3f}','step':0.01,},
	'radius': {'type': float,'label': r'R (R$_\odot$)','fmt': '{:.3f}','step':0.001,},
	'chis': {'type': float,'label': r'$\chi^2$','fmt': '{:.0f}','step':-99,}
}

DEFAULT_MCMC_STEPS = {'teff': 25, 'logg': 0.1, 'z': 0.1, 'enrich': 0.05, 'co': 0.05, 'kzz': 0.25, 'fsed': 0.25, 'ad': 0.01}


DEFINED_INSTRUMENTS = {
#	'EUCLID': {'instrument_name': 'EUCLID NISP', 'altname': [''], 'wave_range': [0.9,1.9]*u.micron, 'resolution': 350, 'bibcode': '', 'sample': '','sample_name': '', 'sample_bibcode': ''},
	'NIR': {'instrument_name': 'Generic near-infrared', 'altname': [''], 'wave_range': [0.9,2.45]*u.micron, 'resolution': 300, 'bibcode': '', 'sample': 'NIR_TRAPPIST1_Davoudi2024.csv','sample_name': 'TRAPPIST-1', 'sample_bibcode': '2024ApJ...970L...4D'},
	'SPEX-PRISM': {'instrument_name': 'IRTF SpeX prism', 'altname': ['SPEX'], 'wave_range': [0.7,2.5]*u.micron, 'resolution': 150, 'bibcode': '2003PASP..115..362R', 'sample': 'SPEX-PRISM_J0559-1404_Burgasser2006.csv','sample_name': '2MASS J0559-1404', 'sample_bibcode': '2006ApJ...637.1067B'},
	'JWST-NIRSPEC-PRISM': {'instrument_name': 'JWST NIRSpec (prism mode)', 'altname': ['JWST-NIRSPEC','NIRSPEC'], 'wave_range': [0.6,5.3]*u.micron, 'resolution': 150, 'bibcode': '', 'sample': 'JWST-NIRSPEC-PRISM_UNCOVER33436_Burgasser2024.csv','sample_name': 'UNCOVER 33336', 'sample_bibcode': '2024ApJ...962..177B'},
	'JWST-NIRSPEC-G395H': {'instrument_name': 'JWST NIRSpec (G395H mode)', 'altname': ['G395H','NIRSPEC-G395H'], 'wave_range': [2.8,5.2]*u.micron, 'resolution': 2000, 'bibcode': '', 'sample': '','sample_name': '', 'sample_bibcode': ''},
	'JWST-MIRI-LRS': {'instrument_name': 'JWST MIRI (LRS mode)', 'altname': ['MIRI','JWST-MIRI'], 'wave_range': [4.6,13.5]*u.micron, 'resolution': 150, 'bibcode': '', 'sample': '','sample_name': '', 'sample_bibcode': ''},
	'JWST-NIRSPEC-MIRI': {'instrument_name': 'JWST NIRSpec (prism mode) + MIRI (LRS mode)', 'altname': ['NIRSPEC-MIRI','JWST-LOWRES'], 'wave_range': [0.8,12.2]*u.micron, 'resolution': 150, 'bibcode': '', 'sample': 'JWST-NIRSPEC-MIRI_J1624+0029_Beiler2024.csv','sample_name': 'SDSS J1624+0029', 'sample_bibcode': '2024arXiv240708518B'},
#	'KECK-NIRES': {'instrument_name': 'Keck NIRES', 'altname': ['NIRES'], 'wave_range': [0.94,2.45]*u.micron, 'resolution': 2700, 'bibcode': '2000SPIE.4008.1048M', 'sample': '','sample_bibcode': ''},
}

DEFINED_SPECTRAL_MODELS = {\
	'atmo20': {'instruments': {}, 'name': 'ATMO2020', 'citation': 'Phillips et al. (2020)', 'bibcode': '2020A%26A...637A..38P', 'altname': ['atmos','phillips','phi20','atmos2020','atmos20','atmo2020','atmo20'], 'default': {'teff': 1500., 'logg': 5.0, 'z': 0.0, 'kzz': 0.,'cld': 'LC','broad': 'A','ad': 1.0,'logpmin': -8, 'logpmax': 4}}, \
	'atmo20pp': {'instruments': {}, 'name': 'ATMO2020++', 'citation': 'Meisner et al. (2023)', 'bibcode': '2023AJ....166...57M', 'altname': ['atmo','atmo++','meisner23','mei23','atmo2020++','atmo20++','atmos2020++','atmos20++'], 'default': {'teff': 1200., 'logg': 5.0, 'z': 0.0,'kzz': 5.0}}, \
	'btdusty16': {'instruments': {}, 'name': 'BT Dusty 2016', 'citation': 'Allard et al. (2012)', 'bibcode': '2012RSPTA.370.2765A', 'altname': ['btdusty2016','dusty16','dusty2016','dusty-bt','bt-dusty','bt-dusty2016','btdusty','bt-dusty16','btd'], 'default': {'teff': 2000., 'logg': 5.0, 'z': 0.0, 'enrich': 0.0}}, \
	'btsettl08': {'instruments': {}, 'name': 'BT Settl 2008', 'citation': 'Allard et al. (2012)', 'bibcode': '2012RSPTA.370.2765A', 'altname': ['allard','allard12','allard2012','btsettl','btsettled','btsettl08','btsettl2008','BTSettl2008','bts','bts08'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'enrich': 0.}}, \
	'burrows06': {'instruments': {}, 'name': 'Burrows et al. (2006)', 'citation': 'Burrows et al. (2006)', 'bibcode': '2006ApJ...640.1063B', 'altname': ['burrows','burrows2006'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'cld': 'nc'}}, \
	'dback24': {'instruments': {}, 'name': 'Sonora Diamondback', 'citation': 'Morley et al. (2024)', 'bibcode': '2024arXiv240200758M', 'altname': ['diamondback','sonora-diamondback','sonora-dback','dback24','diamondback24','morley24','mor24'], 'default': {'teff': 1200., 'logg': 5.0, 'z': 0., 'fsed': 2.}}, \
	'elfowl24': {'instruments': {}, 'name': 'Sonora Elfowl', 'citation': 'Mukherjee et al. (2024)', 'bibcode': '2024ApJ...963...73M', 'altname': ['elfowl','sonora-elfowl','elfowl24','mukherjee','mukherjee24','muk24'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'co': 1, 'kzz': 2.}}, \
	'elfowl24-ph3': {'instruments': {}, 'name': 'Sonora Elfowl + PH3', 'citation': 'Beiler et al. (2024)', 'bibcode': '2024ApJ...973...60B', 'altname': ['elfowl-ph3','sonora-elfowl-ph3','beiler24','beiler','bei24'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'co': 1, 'kzz': 2.}}, \
	'karalidi21': {'instruments': {}, 'name': 'Sonora Cholla', 'citation': 'Karalidi et al. (2021)', 'bibcode': '2021ApJ...923..269K', 'altname': ['karalidi2021','karalidi','sonora-cholla','cholla'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'kzz': 4.}}, \
	'lacy23': {'instruments': {}, 'name': 'Lacy & Burrows (2023)', 'citation': 'Lacy & Burrows (2023)', 'bibcode': '2023ApJ...950....8L', 'altname': ['lacy2023','lac23','lacy'], 'default': {'teff': 400., 'logg': 4.0, 'z': 0., 'cld': 'nc', 'kzz': 0.}}, \
	'lowz': {'instruments': {}, 'name': 'LowZ models', 'citation': 'Meisner et al. (2021)', 'bibcode': '2021ApJ...915..120M', 'altname': ['meisner','meisner2021','mei21','line21','line2021'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'kzz': 2., 'co': 0.85}}, \
	'sand24': {'instruments': {}, 'name': 'SAND', 'citation': 'Alvarado et al. (2024)', 'bibcode': '2024RNAAS...8..134A', 'altname': ['sand','san24','sand2024'], 'default': {'teff': 1500., 'logg': 5.0, 'z': 0.1, 'enrich': 0.0}}, \
	'sonora21': {'instruments': {}, 'name': 'Sonora Bobcat', 'citation': 'Marley et al. (2021)', 'bibcode': '2021ApJ...920...85M', 'altname': ['marley2021','sonora','sonora2021','bobcat','sonora-bobcat'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'co': 1}}, \
	'tremblin15': {'instruments': {}, 'name': 'Tremblin et al. 2015', 'citation': 'Tremblin et al. 2015', 'bibcode': '2015ApJ...804L..17T', 'altname': ['tremblin','tre15','tremblin2015'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.0, 'kzz': 8.0, 'ad': 1.20}}, \
	# 'saumon12': {'instruments': {}, 'name': 'Saumon et al. 2012', 'citation': 'Saumon et al. (2012)', 'bibcode': '2012ApJ...750...74S', 'altname': ['saumon','sau12','saumon2012'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.}}, \
	# 'btcond': {'instruments': {}, 'name': 'BT Cond', 'citation': 'Allard et al. (2012)', 'bibcode': '2012RSPTA.370.2765A', 'altname': ['dusty-cond','bt-cond','btc'], 'default': {'teff': 1500., 'logg': 5.0, 'z': 0.0, 'enrich': 0.0}}, \
	# 'btnextgen': {'instruments': {}, 'name': 'BT NextGen', 'citation': 'Allard et al. (2012)', 'bibcode': '2012RSPTA.370.2765A', 'altname': ['nextgen-bt','btnextgen','btn'], 'default': {'teff': 3000., 'logg': 5.0, 'z': 0.0, 'enrich': 0.}}, \
	# 'btsettl15': {'instruments': {}, 'name': 'BT Settl 2015', 'citation': 'Allard et al. (2015)', 'bibcode': '2015A&A...577A..42B', 'altname': ['allard15','allard2015','btsettl015','btsettl2015','BTSettl2015','bts15'],  'default': {'teff': 1500., 'logg': 5.0, 'z': 0.}}, \
	# 'cond01': {'instruments': {}, 'name': 'AMES Cond', 'citation': 'Allard et al. (2001)', 'bibcode': '2001ApJ...556..357A', 'altname': ['cond','cond-ames','amescond'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.0}}, \
	# 'drift': {'instruments': {}, 'name': 'Drift', 'citation': 'Witte et al. (2011)', 'bibcode': '2011A&A...529A..44W', 'altname': ['witte','witte11','witte2011','helling'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.}}, \
	# 'dusty01': {'instruments': {}, 'name': 'AMES Dusty', 'citation': 'Allard et al. (2001)', 'bibcode': '2001ApJ...556..357A', 'altname': ['dusty','dusty-ames','amesdusty'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.0}}, \
	# 'madhusudhan11': {'instruments': {}, 'name': 'Madhusudhan et al. (2011)', 'citation': 'Madhusudhan et al. (2011)', 'bibcode': '2011ApJ...737...34M', 'altname': ['madhu','madhusudhan','madhu11','madhu2011','madhusudhan2011'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.,'cld': 'ae60', 'kzz': 'eq','fsed': 'eq'}}, \
	# 'morley12': {'instruments': {}, 'name': 'Morley et al. (2012)', 'citation': 'Morley et al. (2012)', 'bibcode': '2012ApJ...756..172M', 'altname': ['morley','morley2012'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'fsed': 'f5'}}, \
	# 'morley14': {'instruments': {}, 'name': 'Morley et al. (2014)', 'citation': 'Morley et al. (2014)', 'bibcode': '2014ApJ...787...78M', 'altname': ['morley2014'], 'default': {'teff': 300., 'logg': 5.0, 'z': 0., 'fsed': 'f5', 'cld': 'h50'}}, \
	# 'saumon08': {'instruments': {}, 'name': 'Saumon & Marley 2008', 'citation': 'Saumon & Marley 2008', 'bibcode': '2008ApJ...689.1327S', 'altname': ['sau08','saumon2008'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.}}, \
	# 'sonora18': {'instruments': {}, 'name': 'Sonora Alpha', 'citation': 'Marley et al. (2018)', 'bibcode': 'marley_mark_2018_1309035', 'altname': ['marley','marley18','marley2018','sonora2018'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'cld': 'nc'}}, \
	# 'gerasimov20': {'instruments': {}, 'name': 'Gerasimov et al. 2020', 'citation': 'Gerasimov et al. (2020)', 'bibcode': '2020RNAAS...4..214G', 'altname': ['phxlowz','ger20'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.}}, \
	# 'veyette': {'instruments': {}, 'name': 'Veyette et al. 2017', 'citation': 'Veyette et al. 2017', 'bibcode': '2017ApJ...851...26V', 'altname': ['veyette17','veyette2017'], 'default': {'teff': 3000., 'logg': 5.0, 'z': 0.0, 'enrich': 0.0, 'carbon': 0.0, 'oxygen': 0.0}}, \
}

# welcome message on load in
print('\n\nWelcome to the UCDMCMC spectral fitting code!')
print('This code is designed to conduct both grid and MCMC fitting of spectral data of ultracool dwarfs')
print('You are currently using version {}\n'.format(VERSION))
# print('If you make use of any features of this toolkit for your research, please remember to cite the SPLAT paper:')
# print('\n{}; Bibcode: {}\n'.format(CITATION,BIBCODE))
# print('If you make use of any spectra or models in this toolkit, please remember to cite the original source.')
print('Please report any errors are feature requests to our github page, {}\n\n'.format(GITHUB_URL))
if ERROR_CHECKING==True: print('Currently running in error checking mode')


#######################################################
#######################################################
################  VARIOUS UTILITIES  ##################
#######################################################
#######################################################

def checkName(ref,refdict,altref='altname',output=False,verbose=ERROR_CHECKING):
	'''

	Purpose
	-------

	General usage program to check if a key is present in a dictionary, with the option to look through alternate names

	Parameters
	----------

	ref : str
		A string that corresponds to the relevant key

	refdict: dict
		Dictionary for which to search for a key

	altref = 'altname' : str
		If present, and refdict is a dictionary of dictionaries, will check the altname keys of the embedded dictionaries
		to identify alternate names

	output = False : bool
		Default returned value if key is missing

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	
	Returns the correct key from the dictionary, or if missing the value specified by output

	Example
	-------

	>>> import ucdmcmc
	>>> ucdmcmc.checkName('lowz',ucdmcmc.DEFINED_SPECTRAL_MODELS)

	'lowz'

	>>> ucdmcmc.checkName('meisner2021',ucdmcmc.DEFINED_SPECTRAL_MODELS)

	'lowz'

	>>> ucdmcmc.checkName('me',ucdmcmc.DEFINED_SPECTRAL_MODELS)

	Could not find item me in input dictionary; try: ['atmo20', 'btdusty16', 'btsettl08', 'burrows06', 
	'dback24', 'elfowl24', 'lowz', 'saumon12', 'sonora21', 'sand24']
	False

	Dependencies
	------------
		
	copy

	'''

# check reference	
	refc = copy.deepcopy(ref)
	if not isinstance(refc,str): return output
	for k in list(refdict.keys()):
		if refc==k: output = k
		if altref in list(refdict[k].keys()):
			if refc in [x for x in list(refdict[k][altref])]: output = k

# return correct key or indicate an error
	if output == False:
		if verbose==True: print('\nCould not find item {} in input dictionary; try: {}'.format(ref,list(refdict.keys())))
	return output

def isUnit(s):
	'''

	Purpose
	-------

	Checks if something is an astropy unit quantity; written in response to the many ways that astropy codes unit quantities

	Parameters
	----------

	s : various
		Quantity to check if unitted

	Outputs
	-------
	
	Returns True if unitted, False if not

	Example
	-------

	>>> import ucdmcmc
	>>> import astropy.units as u
	>>> ucdmcmc.isUnit(5)

	False
	
	>>> ucdmcmc.isUnit(5*u.m)

	True

	>>> ucdmcmc.isUnit((5*u.m).value)

	False

	Dependencies
	------------
		
	astropy.unit

	'''

	return isinstance(s,u.quantity.Quantity) or \
		isinstance(s,u.core.Unit) or \
		isinstance(s,u.core.CompositeUnit) or \
		isinstance(s,u.core.IrreducibleUnit) or \
		isinstance(s,u.core.NamedUnit) or \
		isinstance(s,u.core.PrefixUnit)


#######################################################
#######################################################
#####  BASIC SPECTRAL MANIPULATION AND ANALYSIS  ######
#######################################################
#######################################################

def compareSpec(f1,f2,unc,weights=[],stat='chi-square',verbose=ERROR_CHECKING):
	'''
	
	Purpose
	-------

	Compares two flux vectors and corresponding uncertainty vector and returns a qualitative measure of agreement.
	Note that is assumed the  function, computes chi square with optimal scale factor

	Parameters
	----------

	f1 : numpy.array
		An array of floats corresponding to the first spectrum; this quantity should not have units

	f2 : numpy.array
		An array of floats corresponding to the second spectrum; this quantity should not have units

	unc : numpy.array
		An array of floats corresponding to the joint uncertainty; this quantity should not have units

	weights = [] : numpy.array
		An optional array of floats corresponding to the weighting of the flux values, with large values corresponding
		to higher weights. Weights of zero do not contribute to the quality of fit. By default all weights are 1

	stat = 'chi-square' : str
		Statistic to quantify agreement. NOTE: CURRENTLY THIS IS ONLY CHI-SQUARE

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	
	Returns three (3) floats: the statistic, the optimal relative scaling factor, and the degrees of freedom.
	The scaling factor is defined such that f2 is multiplied to bring it to optimal agreement with f1
	The degrees of freedom takes into account nan values in the fluxes and uncertainty, and weights set to zero

	Example
	-------

	>>> import splat
	>>> import ucdmcmc
	>>> sp1,sp2 = splat.getSpectrum(spt='T5')[:2] # grabs 2 T5 spectra from SPLAT library
	>>> sp2.toWavelengths(sp1.wave)
	>>> ucdmcmc.compareSpec(sp1.flux.value,sp2.flux.value,sp1.noise.value)
	(16279.746979311662, 0.9281232247150684, 561)

	Dependencies
	------------
		
	numpy

	'''
# weighting - can be used to mask bad pixels or weight specific regions
	if len(weights)!=len(f1): wt = numpy.ones(len(f1))
	else: wt=numpy.array(weights)
# mask out bad pixels in either spectrum or uncertainty
	w = numpy.where(numpy.logical_and(numpy.isnan(f1+f2+unc)==False,wt*unc!=0))
	dof = len(f1[w])
	if dof<=1: raise ValueError('Not enough flux or noise values are non-nan')
# compute chi-square - CURRENTLY ONLY OPTION
	scl = numpy.nansum(wt[w]*f1[w]*f2[w]/(unc[w]**2))/numpy.nansum(wt[w]*(f2[w]**2)/(unc[w]**2))
	chi = numpy.nansum(wt[w]*((f1[w]-scl*f2[w])**2)/(unc[w]**2))
	return chi, scl, dof-1

# NOTE: need to rework this using Johnson method
def resample(sp,wave,method='weighted integrate',wave_unit=DEFAULT_WAVE_UNIT,flux_unit=DEFAULT_FLUX_UNIT,default_noise=numpy.nan,smooth=1,verbose=ERROR_CHECKING):
	'''
	
	Purpose
	-------

	Resamples a spectrum onto a wavelength grid with optional smoothing

	Parameters
	----------

	sp : splat.Spectrum class
		splat Spectrum object to resample onto wave grid

	wave : numpy.ndarray or list
		wave grid to resample spectrum onto; if unitted, this is converted to units specified in `wave_unit`, 
		otherwise assumed to be in the units of `wave_unit`

	method = 'integrate' : str
		Method by which spectrum is resampled onto new wavelength grid; options are:
		* 'integrate': flux in integrated across wach wavelength grid point (also 'int')
		* 'weighted integrate' (default): weighted integration, where weights are equal to 1/uncertainty**2 (also 'wint')
		* 'mean': mean value in each wavelength grid point is used (also 'average', 'mn', 'ave')
		* 'weighted mean': weighted mean value with weights are equal to 1/uncertainty**2 (also 'wmn', 'weighted')
		* 'median': median value in each wavelength grid point is used (also 'wmn', 'weighted')

	default_noise = numpy.nan : int or float
		default noise value if not provided in noise array

	smooth = 1 : int
		pixel scale over which to do additional (boxcar) smoothing

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	
	Returns a spectrum object in which the orginal spectrum has been resampled onto the given wave grid, 
	with additional smoothing if noted. 

	Example
	-------

STOPPED HERE

	>>> import splat
	>>> import ucdmcmc
	>>> sp1,sp2 = splat.getSpectrum(spt='T5')[:2] # grabs 2 T5 spectra from SPLAT library
	>>> sp2.toWavelengths(sp1.wave)
	>>> ucdmcmc.compareSpec(sp1.flux.value,sp2.flux.value,sp1.noise.value)
	(16279.746979311662, 0.9281232247150684, 561)

	Dependencies
	------------
		
	`isUnit()`
	splat
	

	'''
# prepare input flux
# 	if isUnit(flux): flx0=flux.to(flux_unit).value
# 	else: flx0 = numpy.array(copy.deepcopy(flux))

# # prepare input uncertainty
# 	if isUnit(noise): unc0=noise.to(flux_unit).value
# 	else: unc0 = numpy.array(copy.deepcopy(noise))
# 	if len(noise)==0: 
# 		if isUnit(default_noise): dns=default_noise.to(flux_unit).value
# 		else: dns = numpy.array(copy.deepcopy(default_noise))
# 		unc0 = numpy.array([dns]*len(flx0))

# # prepare input wavelength grid
# 	if isUnit(wave0): wv0=wave0.to(wave_unit).value
# 	else: wv0 = numpy.array(copy.deepcopy(wave0))

# prepare output wavelength grid
	if isUnit(wave): wv=wave.to(sp.wave.unit).value
	else: wv = numpy.array(copy.deepcopy(wave))
	wshift = 2.*numpy.absolute(numpy.nanmedian(numpy.roll(wv,-1)-wv))


# trim if necessary
#	print(numpy.nanmin(sp.wave.value),numpy.nanmax(sp.wave.value),numpy.nanmin(wv),numpy.nanmax(wv))
	sp.trim([wv[0]-3.*wshift,wv[-1]+3.*wshift])
#	print(numpy.nanmin(sp.wave.value),numpy.nanmax(sp.wave.value),len(sp.wave))
	# wv0 = wv0[wtr]
	# flx0 = flx0[wtr]
	# unc0 = unc0[wtr]
	# wtr = numpy.where(numpy.logical_and(wv0>=wv[0]-3.*wshift,wv0<=wv[0]+3.*wshift))
	# if len(wv0[wtr])==0:
	# 	raise ValueError('Input wavelength grid {:.2f}-{:2f} does not overlap with new input wavelength grid {:.2f}-{:.2f}'.format(numpy.nanmin(wv0),numpy.nanmax(wv0),numpy.nanmin(wv),numpy.nanmax(wv)))

# prepare spectrum object
	# spc = copy.deepcopy(sp)
	# spc.trim([wv[0]-3.*wshift,wv[-1]+3.*wshift])

# run interpolation
	flx = [numpy.nan]*len(wv)
	unc = [numpy.nan]*len(wv)
	smind = int(smooth)
	for i,w in enumerate(wv):
		if i<smind: wrng = [w-(wv[smind]-w),wv[smind]]
		elif i>=len(wave)-smind: wrng = [wv[i-smind],w+(w-wv[i-smind])]
		else: wrng = [wv[i-smind],wv[i+smind]]
		wsel = numpy.where(numpy.logical_and(sp.wave.value>=wrng[0],sp.wave.value<=wrng[1]))
		cnt = len(sp.wave.value[wsel])
# expand range
		if cnt <= 1:
			wsel = numpy.where(numpy.logical_and(sp.wave.value>=wrng[0]-wshift,sp.wave.value<=wrng[1]+wshift))
			cnt = len(sp.wave.value[wsel])
		if cnt >= 1:
			flx0s = sp.flux.value[wsel]
			unc0s = sp.noise.value[wsel]
			wv0s = sp.wave.value[wsel]
			wn = numpy.where(~numpy.isnan(flx0s))
			if len(flx0s[wn])>0:
				if method.lower() in ['mean','mn','average','ave']:
					flx[i] = numpy.nanmean(flx0s[wn])
					if numpy.isfinite(numpy.nanmax(unc0s))==True: unc[i] = numpy.nanmean(unc0s[wn])/((len(unc0s[wn])-1)**0.5)
				elif method.lower() in ['weighted mean','wmn','weighted']:
					wts = 1./unc0s[wn]**2
					if numpy.isnan(numpy.nanmin(wts))==True: wts = numpy.ones(len(wv0s[wn]))
					flx[i] = numpy.nansum(wts*flx0s[wn])/numpy.nansum(wts)
					if numpy.isfinite(numpy.nanmax(unc0s))==True: unc[i] = (numpy.nansum(wts*unc0s[wn]**2)/numpy.nansum(wts))**0.5
				elif method.lower() in ['integrate','int']:
					wts = numpy.ones(len(wv0s[wn]))
					if cnt > 1: 
						flx[i] = numpy.trapz(wts*flx0s[wn],wv0s[wn])/numpy.trapz(wts,wv0s[wn])
						if numpy.isfinite(numpy.nanmax(unc0s))==True: unc[i] = (numpy.trapz(wts*unc0s[wn]**2,wv0s[wn])/numpy.trapz(wts,wv0s[wn]))**0.5
					else:
						flx[i] = numpy.nansum(wts*flx0s[wn])/numpy.nansum(wts)
						if numpy.isfinite(numpy.nanmax(unc0s))==True: unc[i] = (numpy.nansum(wts*unc0s[wn]**2)/numpy.nansum(wts))**0.5
				elif method.lower() in ['weighted integrate','wint']:
					wts = 1./unc0s[wn]**2
					if numpy.isnan(numpy.nanmin(wts))==True: wts = numpy.ones(len(wv0s[wn]))
					if cnt > 1: 
						flx[i] = numpy.trapz(wts*flx0s[wn],wv0s[wn])/numpy.trapz(wts,wv0s[wn])
						if numpy.isfinite(numpy.nanmax(unc0s))==True: unc[i] = (numpy.trapz(wts*unc0s[wn]**2,wv0s[wn])/numpy.trapz(wts,wv0s[wn]))**0.5
					else:
						flx[i] = numpy.nansum(wts*flx0s[wn])/numpy.nansum(wts)
						if numpy.isfinite(numpy.nanmax(unc0s))==True: unc[i] = (numpy.nansum(wts*unc0s[wn]**2)/numpy.nansum(wts))**0.5
					# unc[i] = (numpy.trapz(numpy.ones(len(wv0[wn])),wv0[wn])/numpy.trapz(1/unc0[wn]**2,wv0[wn]))**0.5
					# flx[i] = numpy.trapz(flx0[wn],wv0[wn])/numpy.trapz(numpy.ones(len(wv0[wn])),wv0[wn])
# median by default
				else:
					flx[i] = numpy.nanmedian(flx0s[wn])
					if numpy.isfinite(numpy.nanmax(unc0s))==True: unc[i] = flx[i]/numpy.nanmedian(flx0s[wn]/unc0s[wn])
		# else:
		# 	print('no wavepoints in {:.2f}-{:.2f}'.format(wrng[0],wrng[1]))
#					unc[i] = numpy.nanmedian(unc0[wn])/((len(unc0[wn])-1)**0.5)

# return flux
	# return flx*flux_unit

# return Spectrum object
	return splat.Spectrum(wave=numpy.array(wv)*sp.wave.unit,flux=flx*sp.flux.unit,noise=unc*sp.flux.unit,name=sp.name)


# getSample
def getSample(instrument='NIR',verbose=ERROR_CHECKING):
	'''
	Purpose
	-------

	Reads in one of the pre-set sample spectra

	Parameters
	----------

	instrument = 'NIR' : string
		Instrument sample to upload, must equal one of the keys or alternates in the DEFINED_INSTRUMENTS dictionary

	verbose = False : bool [optional]
		set to True to return verbose output, including listing all models 

	Outputs
	-------

	Spectrum object of the stored sample spectrum

	Example
	-------

	>>> import ucdmcmc
	>>> sp = ucdmcmc.getSample('JWST-NIRSPEC-PRISM')
	>>> sp.info()

	JWST-NIRSPEC-PRISM spectrum of UNCOVER 333436

	If you use these data, please cite:
	
	bibcode: 2024ApJ...962..177B

	History:
		JWST-NIRSPEC-PRISM spectrum successfully loaded

	Dependencies
	------------
		`checkName()`_
		os
		splat
	'''	
# check instrument name
	inst = checkName(instrument,DEFINED_INSTRUMENTS,output='')	
	if inst=='': 
		raise ValueError('Instrument {} is not one of the defined instruments; try {}'.format(instrument,list(DEFINED_INSTRUMENTS.keys())))
# does it have a sample?
	if DEFINED_INSTRUMENTS[inst]['sample']=='':
		raise ValueError('No sample spectrum defined yet for instrument {}'.format(instrument))
# check for file
	sfile = os.path.join(SPECTRA_FOLDER,DEFINED_INSTRUMENTS[inst]['sample'])
	if os.path.exists(sfile)==False:
		raise ValueError('Cannot find sample file {} for instrument {}; check the path and file name'.format(sfile,instrument))
# read in a return
	sp = splat.Spectrum(file=sfile,name=DEFINED_INSTRUMENTS[inst]['sample_name'],instrument=inst)
	sp.published = 'Y'
	sp.data_reference = DEFINED_INSTRUMENTS[inst]['sample_bibcode']
	if verbose==True: print('Reading in sample spectrum for instrument {} of source {}'.format(inst,sp.name))
	return sp



#######################################################
#######################################################
################   MODEL FUNCTIONS  ###################
#######################################################
#######################################################

def modelInfo(model=None,instrument=None,verbose=ERROR_CHECKING):
	'''
	Purpose
	-------

	Provides an overview of the spectral models available with UCDMCCM

	Parameters
	----------

	model = None : string
		name of the model to summarize; set to None to list all models

	verbose = False : bool [optional]
		set to True to return verbose output, including listing all models 

	Outputs
	-------

	Prints a summary of the models available and their parameter ranges

	Example
	-------

	>>> import ucdmcmc
	>>> ucdmcmc.checkName('lowz',ucdmcmc.DEFINED_SPECTRAL_MODELS)
	ucdmcmc.checkName('meisner2021',ucdmcmc.DEFINED_SPECTRAL_MODELS)
	'lowz'

	>>> ucdmcmc.checkName('meisner',ucdmcmc.DEFINED_SPECTRAL_MODELS)

	Model btsettl08:
		Reference: Allard, F. et al. (2012, Philosophical Transactions of the Royal Society A, 370, 2765-2777)
		Bibcode: 2012RSPTA.370.2765A
		Computed for instruments RAW, SPEX-PRISM
		Parameters:
			teff: 500.0 K to 3500.0 K
			logg: 3.0 dex to 5.5 dex
			z: -2.5 dex to 0.5 dex
			enrich: 0.0 dex to 0.4 dex

	Dependencies
	------------
		`splat.citations.shortRef()`_
		`splat.model.loadModelParameters()`_
		`splat.utilities.checkSpectralModelName()`_
		copy
	'''	

# check to see if there are files available
	allfiles = glob.glob(os.path.join(MODEL_FOLDER,'{}*.h5'.format(MODEL_FILE_PREFIX)))
	if len(allfiles)==0:
		print('No pre-calculated models currently available in installation')
		return False

# populate all possible models
	availmodels = {}
	for a in allfiles:
		var = os.path.basename(a).split('_')
		mname = checkName(var[1],DEFINED_SPECTRAL_MODELS,output=var[1])
		if mname not in list(availmodels.keys()): availmodels[mname] = {'instruments': [], 'files': []}
		inst = checkName(var[2].replace('.h5',''),DEFINED_INSTRUMENTS,output=var[2].replace('.h5',''))
		availmodels[mname]['instruments'].append(inst)
		availmodels[mname]['files'].append(os.path.basename(a))
	models = list(availmodels.keys())
	models.sort()

# downselect preferred model
	if model != None:
		mname = checkName(model,DEFINED_SPECTRAL_MODELS,output=model)
		if mname in models: models = [mname]
		else:
			print('Model set {} is not currently available in installation'.format(model))
			return False

# print information about models
	for mdl in models:
		print('\nModel set {}:'.format(mdl))
		f = availmodels[mdl]['instruments'][0]
		if len(availmodels[mdl]['instruments']) > 0:
			for i in availmodels[mdl]['instruments'][1:]: f=f+', {}'.format(i)
		print('\tComputed for instruments {}'.format(f))
		print('\tParameters:')
		mpars,wave = getModelSet(availmodels[mdl]['files'][0])
		kys = list(mpars.columns)
		for x in ['model','file',DEFAULT_FLUX_NAME]:
			if x in kys: kys.remove(x)
		for k in kys:
			vals = list(set(list(mpars[k])))
			vals.sort()
			if isinstance(mpars.loc[0,k],float)==True:
				if len(vals)==1: print('\t\t{}: {}'.format(k,vals[0]))
				else: print('\t\t{}: {} to {}'.format(k,numpy.nanmin(vals),numpy.nanmax(vals)))
			else:
				f = vals[0]
				if len(vals) > 0:
					for i in vals[1:]: f=f+', {}'.format(i)
				print('\t\t{}: {}'.format(k,f))

# information from DEFINED_SPECTRAL_MODELS
		if mdl in list(DEFINED_SPECTRAL_MODELS.keys()):
			print('\taka {} models from {} (bibcode = {})'.format(DEFINED_SPECTRAL_MODELS[mdl]['name'],DEFINED_SPECTRAL_MODELS[mdl]['citation'],DEFINED_SPECTRAL_MODELS[mdl]['bibcode']))

# success
	return True

def generateWave(wave_range,wstep,method='resolution',verbose=ERROR_CHECKING):
	'''
	Purpose
	-------

	Generates a wavelength array by specifying range and either a resolution or constant step size

	Parameters
	----------

	wave_range : list or numpy array
		limits of wavelength range to be modeled; if not unitted, assumed to be microns

	wstep : float
		the wavelength spacing whose value is interpreted based on the method parameter

	method = 'resolution' : str
		the method to use to generate the wave length array; options are:
		* 'resolution' (default): wstep is a constant spectral resolution (also 'res', 'ldl')
		* 'wavelength': wstep is a constant step in wavelength space (also 'lam','lambda','step','linear','wave')
		* 'frequency': wstep is a constant step in frequency space (also 'freq','f','nu')

	verbose = False : bool [optional]
		set to True to return verbose output, including listing all models 

	Outputs
	-------

	Prints a summary of the models available and their parameter ranges

	Example
	-------

	>>> import ucdmcmc
	>>> ucdmcmc.checkName('lowz',ucdmcmc.DEFINED_SPECTRAL_MODELS)
	ucdmcmc.checkName('meisner2021',ucdmcmc.DEFINED_SPECTRAL_MODELS)
	'lowz'

	>>> ucdmcmc.checkName('meisner',ucdmcmc.DEFINED_SPECTRAL_MODELS)

	Model btsettl08:
		Reference: Allard, F. et al. (2012, Philosophical Transactions of the Royal Society A, 370, 2765-2777)
		Bibcode: 2012RSPTA.370.2765A
		Computed for instruments RAW, SPEX-PRISM
		Parameters:
			teff: 500.0 K to 3500.0 K
			logg: 3.0 dex to 5.5 dex
			z: -2.5 dex to 0.5 dex
			enrich: 0.0 dex to 0.4 dex

	Dependencies
	------------
		`splat.citations.shortRef()`_
		`splat.model.loadModelParameters()`_
		`splat.utilities.checkSpectralModelName()`_
		copy
	'''	
# prepare wavelength range
	wunit = DEFAULT_WAVE_UNIT
	if len(wave_range) != 2: raise ValueError('input wave length range must be a 2-element list or numpy array, you passed {}'.format(wave_range))
	if isUnit(wave_range): 
		wunit = wave_range.unit
		wv=wave_range.value
	if isUnit(wave_range[0]):
		wunit = wave_range[0].unit
		wv=[x.value for x in wave_range]
	else: wv = copy.deepcopy(wave_range)

# generate wavelength grid based on different methods
	if method in ['resolution','res','ldl']:
		if verbose==True: print('Generate wavelength grid from {} to {} at constant resolution {}'.format(wv[0]*wunit,wv[1]*wunit,wstep))
		wave = [wv[0]]
		while wave[-1] <= wv[1]: wave.append(wave[-1]*(1+1/wstep))
	elif method.lower() in ['wave','wavelength','lambda','lam','step','linear']:
		if verbose==True: print('Generate wavelength grid from {} to {} at constant lambda step size {}'.format(wv[0]*wunit,wv[1]*wunit,wstep*wunit))
		wave = [wv[0]]
		while wave[-1] <= wv[1]: wave.append(wave[-1]+wstep)
	elif method.lower() in ['frequency','freq','f','nu']:
		if isUnit(wstep): ws.to(u.Hz,equivalencies=u.spectral())
		else: ws=wstep*u.Hz
		if verbose==True: print('Generate wavelength grid from {} to {} at constant frequency step size {}'.format(wv[0]*wunit,wv[0]*wunit,ws))
		wave = [wv[0]]
		while wave[-1] <= wv[1]: wave.append(((wave[-1]*wunit).to(u.Hz,equivalencies=u.spectral())-wstep).to(wunit,equivalencies=u.spectral()).value)

# return
	return numpy.array(wave)*wunit
# default wavelength grid	
DEFAULT_WAVE = generateWave(DEFAULT_WAVE_RANGE,DEFAULT_RESOULTION,method='resolution',verbose=ERROR_CHECKING)

def readWave(inp='SPEX-PRISM',prefix=WAVE_FILE_PREFIX,cname='wave',verbose=ERROR_CHECKING):
	'''
	Reads in an csv file for wave
	'''
# check if the file already exists in sample	
	files = glob.glob(os.path.join(MODEL_FOLDER,'{}{}.csv'.format(prefix,inp)))
	if len(files)>0:
		if verbose==True: print('Reading in wavelength array for {} instrument'.format(inp))
		file = files[0]
	elif os.path.exists(inp)==True:
		file = copy.deepcopy(inp)
	else:
		raise ValueError('WARNING: wave file {} cannot be found, check your file name'.format(inp))
	dp = pandas.read_csv(file)
	if cname not in list(dp.columns): cname = list(dp.columns)[0]
	return numpy.array(dp[cname])*DEFAULT_WAVE_UNIT

def getWave(**kwargs):
	return readWave(**kwargs)

def writeWave(wave,file='wave.csv',overwrite=True,verbose=ERROR_CHECKING):
	'''
	Writes wavelength array to file
	'''	
	if os.path.exists(file)==True:
		if overwrite==False: raise ValueError('WARNING: wave file {} is already in place; set overwrite=True to overwrite'.format(file))
		else:
			if verbose==True: print('WARNING: overwriting wave file {}'.format(file))
	dp = pandas.DataFrame()
	if isUnit(wave): dp['wave'] = wave.value
	else: dp['wave'] = wave
	dp.to_csv(file,index=False)
	if verbose==True: print('Saved wavelength array to {}'.format(file))
	return True


def readModelSet(file,verbose=ERROR_CHECKING):
	'''
	Reads in an h5 model set
	'''	
	if os.path.exists(file)==False:
		raise ValueError('WARNING: model set file {} cannot be found, check your file name'.format(file))
	return pandas.read_hdf(file)


def getModelSet(modelset='',instrument='SPEX-PRISM',wavefile='',file_prefix=MODEL_FILE_PREFIX,wave_prefix=WAVE_FILE_PREFIX,info=False,verbose=ERROR_CHECKING):
	'''
	Purpose
	-------

	Loads in a saved model structure and corresponding wavelength grid. 

	Parameters
	----------

	modelset = '' : str
		The name of the model set, or the full path to the .h5 file containing the model data

	instrument = 'SPEX-PRISM': str
		Name of the instrument for which the models and wavelength grid have been computed

	wavefile = '': str
		Name of the full file path for the wavelength grid

	file_prefix = MODEL_FILE_PREFIX : str
		Optional parameter providing the default file name prefix for .h5 model files

	wave_prefix = WAVE_FILE_PREFIX : str
		Optional parameter providing the default file name prefix for .csv wavelength files

	info = False : bool
		Set to True to report a summary of model parameters

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	
	TBD

	Example
	-------

	>>> import ucdmcmc
	>>> models,wave = ucdmcmc.getModelSet('elfowl24','JWST-NIRSPEC-PRISM')
	>>> models


	Dependencies
	------------
		
	`compareSpec()`
	`getGridModel()`
	`plotCopmare()`
	astropy.unit
	copy
	numpy
	pandas

	'''

# list the stored models that are available	
	if info==True or modelset=='':
		modelInfo(model=modelset)
		return

# construct expected file name and check it's there
	if os.path.exists(modelset)==True: 
		file=copy.deepcopy(modelset)
	elif os.path.exists(os.path.join(MODEL_FOLDER,modelset))==True: 
		file=os.path.join(MODEL_FOLDER,modelset)
	elif os.path.exists('{}{}_{}.h5'.format(file_prefix,modelset,instrument))==True: 
		file = '{}{}_{}.h5'.format(file_prefix,modelset,instrument)
	else: 
		file = os.path.join(MODEL_FOLDER,'{}{}_{}.h5'.format(file_prefix,modelset,instrument))
	if verbose==True: print('Using model data file {}'.format(file))
	if os.path.exists(file)==False:
		print('WARNING: model set file for {} cannot be found, check your file name'.format(modelset))
		modelInfo()
		raise ValueError
	models = readModelSet(file,verbose=verbose)

# read in appropriate wave file
# NOTE: currently raises error of no wave file - could make this optional and not return a wave file
	wfile = ''
	if wavefile!='':
		if os.path.exists(wavefile)==True: wfile=copy.deepcopy(wavefile)
		elif os.path.exists(os.path.join(MODEL_FOLDER,wavefile))==True: 
			wfile=os.path.join(MODEL_FOLDER,wavefile)
		else: 
			if verbose==True: print('WARNING: Could not locate wavelength file {}; going with instrument {}'.format(wavefile,instrument))
	if wfile=='': 
		wfile = os.path.join('{}{}.csv'.format(wave_prefix,instrument))
		if os.path.exists(wfile)==False: 
			wfile = os.path.join(MODEL_FOLDER,'{}{}.csv'.format(wave_prefix,instrument))
	if os.path.exists(wfile)==False:
		print('Could not locate wavelength file for {}; please pass a direct filename'.format(instrument))
		files = glob.glob(os.path.join(MODEL_FOLDER,'{}.csv'.format(wave_prefix)))
		if len(files) > 0: 
			print('Available wavelength grids:')
			for f in files: print('\t{}'.format(os.path.basename(f)))
		raise ValueError
	wave = readWave(wfile,verbose=verbose)

	return models, wave


# generateModelSet
def generateModelSet(modelset,wave=DEFAULT_WAVE,modelpars={},constraints={},initial_instrument='RAW',
	method='integrate',doresample=True,smooth=2,flux_name=DEFAULT_FLUX_NAME,file_prefix=MODEL_FILE_PREFIX,
	save_wave=False,wave_prefix=WAVE_FILE_PREFIX,verbose=ERROR_CHECKING):
	'''
	Purpose
	-------

	Generates a model set interpolated onto the provided wavelength range, with optional constraints.
	This function requires access to a 'RAW' model grid either integrated into SPLAT or provided as an
	input parameter.

	Parameters
	----------

	modelset = '' : str
		The name of the model set to generate, or optionally the full path to the folder containing the RAW model files

	wave = DEFAULT_WAVE : str or list or numpy.ndarray
		Either the name of the instrument that will serve as the baseline wavelength array, or the array 
		of wavelengths to sample the spectra to, which can be of type list or numpy.ndarray and can be 
		unitted or assumed to be in microns

	contraints = {} : dict
		Optional dictionary providing the parameters constraints to apply to the input models, the format of which
		should be {'`key`': [`rng1`,`rng2`]} where `key` is the name of the parameter and `rng1` and `rng2` are
		the lower and upper limits of the parameter range if quantitative. If the parameter is a fixed set of options
		(e.g., cloud parameters), the list should contain all parameter values that you want included.

	modelpars = {} : pandas.DataFrame or dict or str
		Optional input providing the parameters corresponding to the input models. Format should be a 
		pandas Dataframe with a "file" column listing the model filenames, then columns for each of the 
		model parameters; or equivalent dict structure; or a .xslx or .csv file containing these parameters. 
		If not provided, code will attempt to reconstruct model parameters from filename, but note this 
		only works with SPLAT model file name conventions.

	initial_instrument = 'RAW': str
		Name of the instrument for which the models and wavelength grid should be computed; by default this
		is 'RAW' 

	method = 'integrate': str
		The method by which to interpolate the origial model set onto the wavelength grid  (used by `resample`); options are:
		* 'integrate': flux in integrated across wach wavelength grid point (also 'int'; DEFAULT)
		* 'weighted integrate' (default): weighted integration, where weights are equal to 1/uncertainty**2 (also 'wint')
		* 'mean': mean value in each wavelength grid point is used (also 'average', 'mn', 'ave')
		* 'weighted mean': weighted mean value with weights are equal to 1/uncertainty**2 (also 'wmn', 'weighted')
		* 'median': median value in each wavelength grid point is used (also 'wmn', 'weighted')

	smooth = 1 : int
		pixel scale over which to do additional (boxcar) smoothing (used by `resample`)

	file_prefix = MODEL_FILE_PREFIX : str
		Optional parameter providing the default output file name prefix for the resulting .h5 model files

	save_wave = False : bool
		Set to True to save the wavelength grid to a separate file (wavelength is not stored in the h5 file)

	wave_prefix = WAVE_FILE_PREFIX : str
		If saving the wavelegnth grid, sets the file name prefix for .csv wavelength files

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	
	Returns True of model generation was successful, otherwise returns an error message
	Saves the model parameters and interpolated surface fluxes to the .h5 file corresponding to the provided file_prefix
	Optionally saves the wavelength array (in micron) to the .csv file corresponding to the provided wave_prefix

	Example
	-------

	>>> import ucdmcmc
	>>> modelset = 'morley12'
	>>> instrument = 'JWST-NIRSPEC-PRISM'
	>>> constraints = {'teff':[200,500],'logg':[4.5,5.5]}
	>>> ucdmcmc.generateModelSet(modelset,instrument,constraints=constraints,file_prefix='testmodels_{}_{}'.format(modelset,instrument))

	Dependencies
	------------
		
	`readWave()`

	`getGridModel()`
	`plotCopmare()`
	astropy.unit
	copy
	numpy
	pandas

	'''

# load up models and parameters

# first check if this is a folder containing models
	if os.path.isdir(modelset):

# first check if we are reading in folder with instrument name
		files = glob.glob(os.path.join(modelset,'*'))
		if len(files)==0: 
			if os.path.isdir(os.path.join(modelset,initial_instrument))==True:
				modelset = os.path.join(modelset,initial_instrument)
		files = glob.glob(os.path.join(modelset,'*'))
		if len(files)==0: 
			raise ValueError('Unable to find any files in {}'.format(modelset))
		if verbose==True: print('Reading in files from {}'.format(modelset))

# check for modelpars
		if len(modelpars)==0: 
			modelpars = pandas.DataFrame()
			modelpars['file'] = files
			mpar = {}
			for i in range(len(files)):
				par = spmdl.ModelNameToParameters(files[i])
#				print(par)
				for x in list(par.keys()): 
					if i==0: mpar[x] = [par[x]]
					else: mpar[x].append(par[x])
			for x in list(mpar.keys()): modelpars[x] = mpar[x]
			mset = modelpars.loc[i,'model']
#			print(modelpars)
	#		raise ValueError('Must provide modelpars parameter with file names and parameters to use this read in option')
		if isinstance(modelpars,dict)==True: modelpars = pandas.DataFrame(mpars)
		if 'file' not in list(modelpars.columns): 
			raise ValueError('modelpars parameters must have a file column specifying file name')

	# check that first file	is present
		if os.path.exists(modelpars.loc[0,'file'])==False:
			if os.path.exists(os.path.join(modelset,modelpars.loc[0,'file']))==False:
				raise ValueError('Unable to locate the file {} in the modelpars input variable'.format(modelpars.loc[0,'file']))
			else: modelpars['file'] = [os.path.join(modelset,x) for x in modelpars['file']]


# or go through SPLAT path
	else:
# check modelset name
		mset = spmdl.checkSpectralModelName(modelset)
		if isinstance(mset,bool):
			print('WARNING: Model set {} is not contained in SPLAT, cannot run this'.format(modelset))
			return False
		if verbose==True: print('Processing {} models'.format(mset))

# read in model parameters
		mpars = spmdl.loadModelParameters(mset,instrument=initial_instrument)['parameter_sets']
		modelpars = pandas.DataFrame(mpars)
# add in file name
		modelpars['file'] = [os.path.join(splat.SPECTRAL_MODELS[mset]['instruments'][initial_instrument],spmdl.generateModelName(p)) for p in mpars]
		if os.path.exists(modelpars.loc[0,'file'])==False:
			modelpars['file'] = [os.path.join(splat.SPECTRAL_MODELS[mset]['instruments'][initial_instrument],spmdl.generateModelName(p)+'.gz') for p in mpars]
			if os.path.exists(modelpars.loc[0,'file'])==False:
				raise ValueError('Unable to find first model file name {}; check path',format(modelpars.loc[0,'file']))
		# if modelpars.loc[0,'instrument']!=initial_instrument:
		# 	print('WARNING: No {} models for set {} are available in SPLAT, cannot run this'.format(initial_instrument,modelset))
		# 	return

# make constraints if needed
	for k in list(constraints.keys()):
#		if k in list(spmdl.SPECTRAL_MODEL_PARAMETERS.keys()) and k in list(modelpars.columns):
		if k in list(modelpars.columns):
#			if spmdl.SPECTRAL_MODEL_PARAMETERS[k]['type'] == 'continuous':
# discrete (string) variables
			if isinstance(modelpars.loc[0,k],str)==True:
				par = list(set(list(modelpars[k])))
				if verbose==True: print('Constaining {} to within {}'.format(k,constraints[k]))
				for p in par:
					if p not in constraints[k]: 
						modelpars = modelpars[modelpars[k]!=p]
						modelpars.reset_index(inplace=True,drop=True)
# continuous variables
			else:
				if verbose==True: print('Constaining {} to {} to {}'.format(k,constraints[k][0],constraints[k][1]))
				modelpars = modelpars[modelpars[k]>=constraints[k][0]]
				modelpars = modelpars[modelpars[k]<=constraints[k][1]]
				modelpars.reset_index(inplace=True,drop=True)
	if verbose==True: print('Processing {:.0f} {} models'.format(len(modelpars),mset))


# wavelength grid if resampling
# if a string, try reading in
	wave0 = []
	if doresample==True:
		if isinstance(wave,str):
			wave0 = readWave(wave,verbose=verbose)
# check if unitted and convert if so
		elif isinstance(wave,list) or isinstance(wave,numpy.ndarray):
			if isUnit(wave): wave0 = wave.to(DEFAULT_WAVE_UNIT).value
			elif isUnit(wave[0]): wave0 = [w.to(DEFAULT_WAVE_UNIT).value for w in wave]
			else: wave0 = copy.deepcopy(wave)
			if len(wave0) < 2:
				print('Input wavelength array has only {:.0f} elements, skipping resample'.format(len(wave0)))
				doresample = False
# return if cannot process wave grid
		else:
			print('Unable to read wave input of type {}: skipping resample'.format(type(wave)))
			doresample = False

# read in the models trying a few different methods
	pars = []
	step = numpy.ceil(len(modelpars)/10.)
#	for i in tqdm(range(len(dp))):
	for i in range(len(modelpars)):
		if i!=0 and numpy.mod(i,step)==0 and verbose==True: print('\t{:.0f}% complete'.format(i/step*10),end='\r')
		par = dict(modelpars.loc[i,:])

# read in with splat.Spectrum
		mdl = splat.Spectrum(modelpars.loc[i,'file'])
		wv,flx = mdl.wave.value,mdl.flux.value
# read in with spmdl.loadModel
		if numpy.isfinite(numpy.nanmedian(flx))==False:
			par = dict(modelpars.loc[i,:])
			mdl = spmdl.loadModel(**par,force=True)
			wv,flx = mdl.wave.value,mdl.flux.value
# read in with splat.readSpectrum
		if numpy.isfinite(numpy.nanmedian(flx))==False:
			mdl = splat.readSpectrum(modelpars.loc[i,'file'])
			wv,flx = mdl['wave'].value,mdl['flux'].value
# read in with pandas
		if numpy.isfinite(numpy.nanmedian(flx))==False:
			if '.txt' in modelpars.loc[i,'file']: delim='\t'
			elif '.csv' in modelpars.loc[i,'file']: delim=','
			else: delim='\s+'
			dp = pandas.read_csv(modelpars.loc[i,'file'],delimiter=delim,names=['wave','flux'],comment='#')
			wv,flx = dp['wave'],dp['flux']
# don't know what to do
#		print(modelpars.loc[i,'file'],len(flx))
		try: md = numpy.isfinite(numpy.nanmedian(flx))
		except: raise ValueError('Could not read in file {}'.format(modelpars.loc[i,'file']))

# resample if desired
		if doresample==True:
			mdl = splat.Spectrum(wave=wv*DEFAULT_WAVE_UNIT,flux=flx*DEFAULT_FLUX_UNIT)
			mdlsm = resample(mdl,wave0,smooth=smooth,method=method)
			wv,flx = mdlsm.wave.value,mdlsm.flux.value
		else:
			if i==0: wave0 = copy.deepcopy(wv)

		par[flux_name] = flx
		pars.append(par)



# # read in and (optionally) resample the models
# 	if verbose==True: print('Processing {:.0f} {} models'.format(len(modelpars),mset))
# 	pars = []
# # using a very dumbed down tqdm to save on memory issues	
# 	step = numpy.ceil(len(modelpars)/10.)
# #	for i in tqdm(range(len(dp))):
# 	for i in range(len(modelpars)):
# 		if i!=0 and numpy.mod(i,step)==0 and verbose==True: print('\t{:.0f}% complete'.format(i/step*10),end='\r')
# 		par = dict(modelpars.loc[i,:])
# # ***** REPLACE THIS WITH REGULAR FILE READ
# 		# mdl = spmdl.loadModel(**par,force=True)
# 		mdl = splat.Spectrum(modelpars.loc[i,'file'])
# #		par[flux_name] = resample(mdl.flux,mdl.wave,wv,smooth=smooth,method=method)
# 		mdlsm = resample(mdl,wv,smooth=smooth,method=method)
# 		par[flux_name] = mdlsm.flux.value
# 		pars.append(par)

# save the models
	outfile = file_prefix+'.h5'
	dpo = pandas.DataFrame(pars)
	for x in ['instrument','file']:
		if x in list(dpo.columns): del dpo[x]
	if verbose==True: print('Saving {} models to {}'.format(mset,outfile))
	try: dpo.to_hdf(outfile,'models','w',complevel=4,index=False)
	except: 
		print('WARNING: unable to write hdf file {}; returning wave grid and models as outputs'.format(outfile))
	if save_wave==True: 
		dpw = pandas.DataFrame()
		dpw['wave'] = wv
		outfile = wave_prefix+'.csv'
		dpw.to_csv(outfile,index=False)
		if verbose==True: print('Saving wavelength array to {}'.format(outfile))
	return wv,dpo


# getGridModel()
def getGridModel(models,par,wave=[],flux_name=DEFAULT_FLUX_NAME,scale=True,verbose=ERROR_CHECKING):
	'''
	Purpose
	-------

	Gets a specific model from a model set.

	Parameters
	----------

	models : pandas.DataFrame
		Dataframe containing the model parameters and fluxes

	par : dict
		Dictionary specifying the parameters of the grid model desired. The format is 
		{'`key`': `value`}, where `key` is the name of the parameter and `value` its value, which should
		have the same type as the parameter values in the models dataframe.

	wave = [] : list or numpy.ndarray
		Array of wavelengths that corresponds to the flux values, and must have the same length as the
		flux values. Can be unitted or is assumed to be in microns

	flux_name = DEFAULT_FLUX_NAME : str
		Column name in which the flux values are specified in the models dataframe

	scale = True : bool
		Set to True if a `scale` parameter is included in par and should be used to scale the fluxes

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	
	Returns a Spectrum object containing the wavelength and model flux values, optionally scaled.
	If the wave parameter is not included or of a different length than the fluxe array, returns 
	just the flux array.

	Example
	-------

	>>> import ucdmcmc
	>>> models,wave = ucdmcmc.getModels('dback24','SPEX-PRISM')
	>>> par = {'fsed': 4.0, 'logg': 5.00, 'teff': 1000.0, 'z': 0.5}
	>>> mdl = ucdmcmc.getGridModel(models,par,wave=wave)
	>>> mdl.name

	dback24 model: fsed=4.0 logg=5.0 teff=1000.0 z=0.5
	
	>>> par = {'logg': 5.00, 'teff': 1000.0, 'z': 0.5}
	>>> mdl = ucdmcmc.getGridModel(models,par,wave=wave)
	>>> mdl.name

	dback24 model
	
	6 models statisfy criteria, returning the first one
	dback24 model: fsed=2.0 logg=5.0 teff=1000.0 z=0.5


	Dependencies
	------------
		
	`isUnit()`
	`plotCopmare()`
	copy
	pandas
	splat.Spectrum

	'''
# prep wavelegngth array
	if isUnit(wave)==False: wv = wave*DEFAULT_WAVE_UNIT
	else: wv = wave.to(DEFAULT_WAVE_UNIT)

# prep downselect
	kys = list(models.columns)
	for x in ['model',flux_name,'file']:
		if x in kys: kys.remove(x)
	smdls = copy.deepcopy(models)

# do downselect
	for k in kys:
		if k in list(par.keys()): 
			# if k=='kzz': smdls = smdls[smdls[k]==str(par[k])]
			# else: 
			smdls = smdls[smdls[k]==par[k]]
			smdls.reset_index(inplace=True,drop=True)
	if len(smdls)==0: raise ValueError('No models match parameters {}'.format(par))
	elif len(smdls)>1: 
		if verbose==True: print('{:.0f} models statisfy criteria, returning the first one'.format(len(smdls)))
	flx = smdls.loc[0,flux_name]
	name = '{} model: '.format(models.loc[0,'model'])
# NEED TO ADD - FORMAT STRING FROM PARAMETERS
	for x in kys: 
		name=name+'{}={} '.format(x,smdls.loc[0,x])
	mdl = splat.Spectrum(wave=wave,flux=flx*DEFAULT_FLUX_UNIT,name=name)
	if 'scale' in list(par.keys()) and scale==True: mdl.scale(par['scale'])
	mdl.parameters = dict(smdls.loc[0,:])
	for x in [flux_name,'file']:
		if x in list(mdl.parameters.keys()): del mdl.parameters[x]
	return mdl


def getInterpModel(models,par,wave=[],flux_name=DEFAULT_FLUX_NAME,scale=True,defaults={},verbose=ERROR_CHECKING):
	'''
	Purpose
	-------

	Takes a set of models and generates and interpolated model over provided parameters, using 
	log flux interpolation

	Parameters
	----------

	models : pandas.DataFrame
		Dataframe containing the model parameters and fluxes

	par : dict
		Dictionary specifying the parameters of the interpolated model desired. The format is 
		{'`key`': `value`}, where `key` is the name of the parameter and `value` its value, which should
		have the same type as the parameter values in the model's dataframe. Any parameters not
		provided will be assumed to have the default values from DEFINED_SPECTRAL_MODELS

	wave = [] : list or numpy.ndarray
		Array of wavelengths that corresponds to the flux values, and must have the same length as the
		flux values. Can be unitted or is assumed to be in microns

	flux_name = DEFAULT_FLUX_NAME : str
		Column name in which the flux values are specified in the models dataframe

	scale = True : bool
		Set to True if a `scale` parameter is included in par and should be used to scale the fluxes

	defaults = {} : dict
		Dictionary specifying the default parameters to assume if not constrained by par. This keyword
		is really a catch if making us of models that are not part of the defined ucdmcmc package,
		and the default behavior is to use the `default` dictionary provided for each model in
		DEFINED_SPECTRAL_MODELS

		Set to True if a `scale` parameter is included in par and should be used to scale the fluxes

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	
	Returns a Spectrum object containing the wavelength and model flux values, optionally scaled.
	If model cannot be successfully interpolated (out of range parameters), a ValueError is raised. 
	If the wave parameter is not included or of a different length than the fluxe array, returns 
	just the flux array.

	Example
	-------

	>>> import ucdmcmc
	>>> models,wave = ucdmcmc.getModels('dback24','SPEX-PRISM')
	>>> par = {'fsed': 4.0, 'logg': 5.20, 'teff': 1150.0, 'z': 0.3}
	>>> mdl = ucdmcmc.getInterpModel(models,par,wave=wave)
	>>> mdl.name

	dback24 model: fsed=4.0 logg=5.2 teff=1150.0 z=0.3

	>>> par = {'logg': 5.20, 'teff': 1150.0}
	>>> mdl = ucdmcmc.getInterpModel(models,par,wave=wave)

	dback24 model: logg=5.2 teff=1150.0 fsed=2.0 z=-0.0

	>>> par = {'fsed': 4.0, 'logg': 5.20, 'teff': 350.0, 'z': 0.3}
	>>> mdl = ucdmcmc.getInterpModel(models,par,wave=wave)

	ValueError: No model satisfies parameter selection (failed at teff = 350.0)

	Dependencies
	------------
		
	`isUnit()`
	`plotCopmare()`
	copy
	pandas
	splat.Spectrum

	'''
# prep wavelength array
	if isUnit(wave)==False: wv = wave*DEFAULT_WAVE_UNIT
	else: wv = wave.to(DEFAULT_WAVE_UNIT)


# get model defaults
	if len(defaults) == 0:
		mset = checkName(models.loc[0,'model'],DEFINED_SPECTRAL_MODELS,output=False)
		if not isinstance(mset,bool): defaults = DEFINED_SPECTRAL_MODELS[mset]['default']		
#	

# check all model parameters are provided in set parameters or defaults
	par0 = copy.deepcopy(par)
	kys = list(models.columns)
	for x in ['model',flux_name,'file']:
		if x in kys: kys.remove(x)
	for k in kys:
# if parameter not provided, use defaults or bail out
		if k not in list(par0.keys()):
			if k not in list(defaults.keys()):
				raise ValueError('Model parameter {} is not defined for input parameters or defaults; must specify all parameters'.format(k))
			par0[k] = defaults[k]

# downselect models or bail out if we're outside parameter range
	smdls = copy.deepcopy(models)
	limits,steps = {},{}

	for k in kys:
		vals = list(set(list(smdls[k])))
		vals.sort()
# discrete parameters - match exactly		
		if isinstance(smdls.loc[0,k],str)==True: 
			if par0[k] not in vals:
				raise ValueError('Parameter {} = {} is not among values in models: {}'.format(k,par0[k],vals))
			smdls = smdls[smdls[k]==par0[k]]
			smdls.reset_index(inplace=True,drop=True)
		else:
# continuous parameters			
#				print(par[k],numpy.nanmin(vals),numpy.nanmax(vals)) 
			if par0[k] in vals:
				smdls = smdls[smdls[k]==par0[k]]
				smdls.reset_index(inplace=True,drop=True)
			else:
				valstep = numpy.absolute(numpy.array(vals)-numpy.roll(vals,1))
				step = numpy.nanmedian(valstep[1:])				
				limits[k] = [numpy.nanmax([numpy.nanmin(smdls[k]),par0[k]-step]),numpy.nanmin([numpy.nanmax(smdls[k]),par0[k]+step])]
				if step>0:
					smdls = smdls[smdls[k]>=limits[k][0]]
					smdls = smdls[smdls[k]<=limits[k][1]]													 
					smdls.reset_index(inplace=True,drop=True)
# overselected - no models to interpolate
		if len(smdls)==0: 
			raise ValueError('No model satisfies parameter selection (failed at {} = {})'.format(k,par0[k]))
	
# eliminate degenerate parameters
	kys0 = copy.deepcopy(kys)
	for k in kys:
		if len(set(list(smdls[k])))<2: 
			kys0.remove(k)
			par0[k] = smdls.loc[0,k]
	kys = copy.deepcopy(kys0)
	
# prep models for griddata interpolation
# note that we are taking the log of teff and co
	fitvals,parvals = (),[]
	for k in kys:
		if k=='teff' or k=='co': 
			fitvals+=tuple([[numpy.log10(x) for x in smdls[k]]])
			parvals.append(numpy.log10(par0[k]))
		else:
			fitvals+=tuple([list(smdls[k])])
			parvals.append(par0[k])
	parvals = numpy.array([parvals])
	fitvals = numpy.transpose(numpy.vstack(fitvals))

# run interpolation
	flx = []
	for i in range(len(smdls.loc[0,flux_name])):
		fs = [numpy.log10(x[i]) for x in smdls[flux_name]]
		try: flx.append(griddata(fitvals,tuple(fs),parvals,method='linear',rescale=True)[0])
		except: 
			if verbose==True: print('getInterpModel failed for values '.format)
			raise ValueError('Insufficient model coverage; try reducing parameter constraints')
	flx = numpy.array(flx)
	flx = 10.**flx
	if numpy.isnan(numpy.nanmedian(flx))==True: raise ValueError('Could not interpolate {} over grid, possibly due to grid gaps'.format(par0))
#	print(truepar)

# turn into Spectrum and scale if desired
	name = '{} model: '.format(models.loc[0,'model'])
# NEED TO ADD - FORMAT STRING FROM PARAMETERS
	for x in list(par0.keys()): 
		name=name+'{}={} '.format(x,par0[x])
	mdl = splat.Spectrum(wave=wave,flux=flx*DEFAULT_FLUX_UNIT,name=name)
	if 'scale' in list(par.keys()) and scale==True: mdl.scale(par['scale'])
	mdl.parameters = par0
	return mdl



def getModel(mdls,par,wave,scale=True,verbose=ERROR_CHECKING):
	try: sp = getGridModel(mdls,par,wave,scale=scale,verbose=verbose)
	except: sp = getInterpModel(mdls,par,wave,scale=scale,verbose=verbose)
#	if 'scale' in par and rescale==True: sp.scale(par['scale'])
	return sp


########################################################################
# FITTING METHODS
########################################################################


def fitGrid(spc,models,constraints={},flux_name=DEFAULT_FLUX_NAME,output='parameters',absolute=False,
	report=True,xscale='linear',yscale='linear',file_prefix='gridfit_',verbose=ERROR_CHECKING):
	'''
	Purpose
	-------

	Fit a spectrum to model grid, with optional constraints. This function uses a chi-square statistic and determines the
	overall best fit model from a set of model fluxes. 

	Parameters
	----------

	ref : spc
		A Spectrum class object that contains the spectrum to be fit, assumed to be onthe same wavelength scale as the models

	models: pandas DataFrame
		A pandas DataFrame that is read in through `readModels()` or `getModelSet()` that contains the model parameters 
		and associated flux arrays

	contraints = {} : dict
		Optional parameter constraints, with the keys corresponding to the model parametes associated with 2-element
		arrays that specify the lower and upper limits. NOTE: currently only works with quantitative variables

	output = 'parameters' : str
		Specify what the program should return; options are:
			* 'parameters' (DEFAULT): return a dict of the best-fit model parameters
			* 'spectrum': return the spectrum of the best-fit model as a Spectrum class object
			* 'allvalues': return the input models pandas DataFrame with the optimal scale factor, chi-square, 
				and degrees of freedom added

	absolute = True : bool
		Set to True if spectrum fluxes are in absolute flux units (flux at 10 pc), such that the optimal scaling factor 
		provides a realistic estimate of the radius

	xscale = 'linear' : str
		Scaling of the x-axis, based on the options in matplotlib.set_xscale()

	yscale = 'linear' : str
		Scaling of the y-axis, based on the options in matplotlib.set_xscale()

	report = True : bool
		Set to True to save an output file showing the best fit model

	file_prefix = 'gridfit_' : str
		Prefix to append to output file if report = True

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	
	Determined by the `output` keyword; options are:
		* 'parameters' (DEFAULT): return a dict of the best-fit model parameters
		* 'spectrum': return the spectrum of the best-fit model as a Spectrum class object
		* 'allvalues': return the input models pandas DataFrame with the optimal scale factor, chi-square, 
			and degrees of freedom added

	Example
	-------

	>>> import ucdmcmc
	>>> sp = ucdmcmc.getSample('JWST-NIRSPEC-PRISM')
	>>> models,wave = ucdmcmc.getModelSet('elfowl24','JWST-NIRSPEC-PRISM')
	>>> spsm = ucdmcmc.resample(sp,wave)
	>>> par0 = ucdmcmc.fitGrid(spsm,models,verbose=True)

	Best fit model:
		model = elfowl24
		co = 0.5
		kzz = 2.0
		logg = 5.0
		teff = 1000.0
		z = -0.0
		scale = 1.387711853929069e-24
		chi = 956.6257633740316
		radius = 0.0005224902503080241
		dof = 850.0
		rchi = 1.1254420745576843
		reduced chi2 = 1.1254420745576843

	Dependencies
	------------
		
	`compareSpec()`
	`getGridModel()`
	`plotCopmare()`
	astropy.unit
	copy
	numpy
	pandas

	'''
# make sure object spectrum is sampled to same wavelength scale as models
	if len(spc.flux)!=len(models.loc[0,flux_name]):
		raise ValueError('Spectrum and models are not on same wavelength scale; be sure to resample observed spectrum onto model scale')
	spscl = copy.deepcopy(spc)

# constrain models if needed
	mdls = copy.deepcopy(models)
	for k in list(constraints.keys()):
		if k in list(mdls.columns):
			if isinstance(mdls.loc[0,k],str):
				par = list(set(list(dp[k])))
				if verbose==True: print('Constaining {} to within {}'.format(k,constraints[k]))
				for p in par:
					if p not in constraints[k]: mdls = mdls[mdls[k]!=p]
			else:
				if verbose==True: print('Constaining {} to {}-{}'.format(k,constraints[k][0],constraints[k][1]))
				mdls = mdls[mdls[k]>=constraints[k][0]]
				mdls = mdls[mdls[k]<=constraints[k][1]]
				mdls.reset_index(inplace=True,drop=True)
	
# run through each grid point
	for x in ['scale','chis','radius','dof']: mdls[x] = [numpy.nan]*len(mdls)
	for jjj in range(len(mdls)):
		chi,scl,dof = compareSpec(spscl.flux.value,numpy.array(mdls.loc[jjj,flux_name]),spscl.noise.value,verbose=verbose)
		mdls.loc[jjj,'chis'] = chi
		mdls.loc[jjj,'scale'] = scl
		mdls.loc[jjj,'dof'] = dof
# radius scaling assuming spectrum is in absolute flux units
		mdls.loc[jjj,'radius'] = (10.*u.pc*(scl**0.5)).to(u.Rsun).value
#	mdls['model'] = [mset]*len(mdls)

# best fit
	mpar = dict(mdls.loc[numpy.argmin(mdls['chis']),:])
	mpar['rchi'] = mpar['chis']/mpar['dof']
	dpars = list(mdls.keys())
	for x in [flux_name]:
		if x in list(mpar.keys()): del mpar[x]
	if verbose==True: 
		print('Best fit model:')
		for k in mpar:
#			mpar[k] = mdls.loc[ibest,k]
			print('\t{} = {}'.format(k,mpar[k]))
	comp = getGridModel(mdls,mpar,spscl.wave,verbose=verbose)
#	comp.scale(mpar['scale'])
#	comp = splat.Spectrum(wave=wave,flux=numpy.array(mdls.loc[ibest,flux_name])*mdls.loc[ibest,'scale']*spscl.flux.unit)
	diff = spscl.flux.value-comp.flux.value
#	dof = numpy.count_nonzero(~numpy.isnan(spscl.flux.value))-1
	if verbose==True: print('\treduced chi2 = {}'.format(mpar['rchi']))
	# sclstd = numpy.nanstd(diff.flux.value,ddof=1)/numpy.nanmax(spscl.flux.value)
	# mpar['sclstd'] = sclstd

	if report == True:
# save parameters
		outfile = file_prefix+'_parameters.xlsx'
		mdls.drop(columns=[flux_name],inplace=True)
		mdls.to_excel(outfile,index=False)
# comparison plot		
		outfile = file_prefix+'_compare.pdf'
		label = '{} model '.format(mdls.loc[0,'model'])
		label+=r'$\chi^2_r$='+'{:.1f}\n'.format(mpar['rchi'])
		label+='T={:.0f} '.format(mpar['teff'])
		label+='logg={:.2f} '.format(mpar['logg'])
		label+='z={:.2f} '.format(mpar['z'])
		plotCompare(spscl,comp,outfile=outfile,clabel=label,absolute=absolute,xscale=xscale,yscale=yscale)
	return mpar


def fitMCMC(spc,models,p0={},constraints={},flux_name=DEFAULT_FLUX_NAME,output='all',
	pstep=DEFAULT_MCMC_STEPS,nstep=100,iterim=50,method='chidiff',threshhold=0.5,burn=0.25,
	quantscale=[0.25,0.5,0.75],nsample=0,absolute=False,report=True,xscale='linear',yscale='linear',
	file_prefix='mcmcfit_',verbose=ERROR_CHECKING):
#	radius=numpy.nan,e_radius=numpy.nan,report=True):
	'''
	Purpose
	-------

	Fit a spectrum to model grid using a simple Metropolis-Hastings Markov Chain Monte Carlo methor. Conducts the fit
	through model grid by interpolating (logarithmic) fluxes, and advances chains based on various chi-square comparison
	statistics. Returns a variety of outputs depending on the output and report keywords.

	Parameters
	----------

	ref : spc
		A Spectrum class object that contains the spectrum to be fit, assumed to be onthe same wavelength scale as the models

	models: pandas DataFrame
		A pandas DataFrame that is read in through `readModels()` or `getModelSet()` that contains the model parameters 
		and associated flux arrays

	p0 = {} : dict
		Dictionary containing the initial parameters; if not provided or some parameters missing, default parameters are
		used from the DEFINED_SPECTRAL_MODELS parameter

	contraints = {} : dict
		Optional parameter constraints, with the keys corresponding to the model parametes associated with 2-element
		arrays that specify the lower and upper limits. NOTE: currently only works with quantitative variables

	output = 'all' : str
		Specify what the function should return; options are:
			* 'best': return a dict of the best-fit model parameters
			* 'model': return the best fit model as a Spectrum class object
			* 'distribution': return a dict of the model parameter distributions (25%, 50%, and 75% quantiles)
			* 'chain': return the chain of parameter values as a pandas Dataframe
			* 'all' (DEFAULT): returns a dict containing all three of the above 

	pstep = DEFAULT_MCMC_STEPS : dict
		Dictionary containing the parameter step scales. New parameers are selected by a normal distribution with the step
		width, up to the parameter limits

	nstep = 100 : int
		Number of MCMC steps to take; ideally this should be 1000-10000+

	iterim = 50 : int
		If > 0 and report = True, this will iteratively save the chains and diagnostic plots on a cadence of iterim steps

	method = 'chidiff' : str
		The method by which the current (i) and previous (i-1) model fit chi2 values are compared; options are:
			* 'chidiff' (DEFAULT): compare difference in chis to overall miniumum chi2: (chi2[i]-chi2[i-1])/min(chi2)
			* 'survival': compute the F-test survival fraction for the ratio of chi2 values: SF(chi2[i]/chi2[i-1],dof,dof)
			 (see scipy.stats.f)
			* 'dofscale': compares the difference in chi2 values to the dof: dof/(0.5*dof+chi2[1]-chi2[-1])
	treshhold = 0.5 : float
		A scaling factor to determine the acceptance rate for new parameters, where the comparison statistic is compared to a
		uniform draw between [0,threshhold] or [threshhold,1] (depending on the statistic)

	burn = 0.25 : float
		The fraction of the initial chain to remove before determining the final parameter distributions. Set to a low fraction
		if you are starting with a good approximation of the model parameters

	quantscale = [0.25,0.5,0.75] : list
		List of three floats indicating the quantile scales to report for the parameter distributions

	nsample = 0 : int
		For the comparison plot, set to 0 to compare to the best fit model, or to a positive integer to compare to nsample models
		drawn from the chain

	xscale = 'linear' : str
		Scaling of the x-axis, based on the options in matplotlib.set_xscale()

	yscale = 'linear' : str
		Scaling of the y-axis, based on the options in matplotlib.set_xscale()

	absolute = True : bool
		Set to True if spectrum fluxes are in absolute flux units (flux at 10 pc), such that the optimal scaling factor 
		provides a realistic estimate of the radius

	report = True : bool
		Set to True to save both the chains and several diagnostic plots

	file_prefix = 'mcmcfit_' : str
		Prefix to append to output files if report = True

	verbose = ERROR_CHECKING : bool
		Set to True to return verbose output

	Outputs
	-------
	
	Determined by the `output` keyword; options are:
		* 'best': return a dict of the best-fit model parameters
		* 'model': return the best fit model as a Spectrum class object
		* 'distribution': return a dict of the model parameter distributions (25%, 50%, and 75% quantiles)
		* 'chain': return the chain of parameter values as a pandas Dataframe
		* 'all' (DEFAULT): returns a dict containing all three of the above 

	Example
	-------

	>>> import ucdmcmc
	>>> sp = ucdmcmc.getSample('JWST-NIRSPEC-PRISM')
	>>> models,wave = ucdmcmc.getModelSet('sonora21','JWST-NIRSPEC-PRISM')
	>>> spsm = ucdmcmc.resample(sp,wave)
	>>> par0 = ucdmcmc.fitGrid(spsm,models,verbose=False) # initial fit
	>>> par = ucdmcmc.fitMCMC(spsm,models,p0=par0,nstep=1000,burn=0.25,verbose=False) # MCMC
	>>> par['distributions']
	{'co': array([0.54139703, 0.88476726, 1.25166873]),
	  'kzz': array([2.17372842, 2.90277005, 3.99281996]),
	  'logg': array([4.693938  , 5.10931096, 5.41861495]),
	  'teff': array([ 976.68788945, 1033.00133369, 1090.21418499]),
	  'z': array([-0.35613309, -0.12806126,  0.12211709])}

	Dependencies
	------------
		
	`compareSpec()`
	`getGridModel()`
	`plotCopmare()`
	astropy.unit
	copy
	numpy
	pandas

	'''
# make sure object spectrum is sampled to same wavelength scale as models
	if len(spc.flux)!=len(models.loc[0,flux_name]):
		raise ValueError('Spectrum and models are not on same wavelength scale; be sure to resample observed spectrum onto model scale')
	spscl = copy.deepcopy(spc)

# constrain models if needed
	mdls = copy.deepcopy(models)
	for k in list(constraints.keys()):
		if k in list(mdls.columns):
			if isinstance(mdls.loc[0,k],str):
				par = list(set(list(dp[k])))
				if verbose==True: print('Constaining {} to within {}'.format(k,constraints[k]))
				for p in par:
					if p not in constraints[k]: 
						mdls = mdls[mdls[k]!=p]
						mdls.reset_index(inplace=True,drop=True)
			else:
				if verbose==True: print('Constaining {} to {}-{}'.format(k,constraints[k][0],constraints[k][1]))
				mdls = mdls[mdls[k]>=constraints[k][0]]
				mdls = mdls[mdls[k]<=constraints[k][1]]
				mdls.reset_index(inplace=True,drop=True)
	mset = mdls.loc[0,'model']
	mkys = list(mdls.keys())
	for x in ['model',flux_name]:
		if x in mkys: mkys.remove(x)

# if no or incomplete fit parameters, conduct an initial grid fit
	chk = True
	for k in mkys: chk=chk and (k in list(p0.keys()))
	if chk==False:
		if verbose==True: print('Running initial grid fit')
		p0 = fitGrid(spc,mdls,absolute=absolute,report=False,verbose=verbose)
		if flux_name in list(p0.keys()): del p0[flux_name]
		if verbose==True: print('\nGrid fit parameters: {}'.format(p0))

# validate steps
	if verbose==True: print('Fitting the following parameters:')
	mkysfit = copy.deepcopy(mkys)
	for k in mkys:
		vals = list(set(list(mdls[k])))
		vals.sort()
		if len(vals)<2: pstep[k] = 0.
		else:
			if k not in list(pstep.keys()):
				if isinstance(mdls.loc[0,k],str): pstep[k] = -1.
				else: pstep[k] = 0.5*numpy.nanmedian(numpy.absolute(numpy.array(vals)-numpy.roll(vals,1)))
		if pstep[k] == 0: mkysfit.remove(k)
		else:
			if verbose==True: print('\t{}: initial={} step={}'.format(k,p0[k],pstep[k]))
	nparam = len(mkysfit)

# continuous and discrete variables
	pfitc,pfitd = {},{}
	for k in mkysfit: 
		if k in list(p0.keys()):
			if isinstance(mdls.loc[0,k],str): pfitd[k] = p0[k]
			else: pfitc[k] = p0[k]
		else: 
			default = splat.SPECTRAL_MODELS[mset]['default'][k]
			if isinstance(default,str): pfitd[k] = default
			else: pfitc[k] = default

# some plotting set up
	ylabelpre = 'Scaled '
	if absolute==True: ylabelpre='Absolute '

# initialize MCMC
# SOMETHING IS WRONG HERE
	cmdl = getModel(mdls,p0,spscl.wave,scale=False,verbose=verbose)
	chi,scl,dof = compareSpec(spscl.flux.value,cmdl.flux.value,spscl.noise.value,verbose=verbose)
	dof = dof-nparam
	cmdl.scale(scl)
	chis = [chi]
	pvals = [p0]
	mdlflxs = [cmdl.flux.value]
	scales = [scl]

# run MCMC
	if verbose==True: print('Running MCMC for {:.0f} steps'.format(nstep))
	for i in tqdm(range(nstep)):
		pnew = copy.deepcopy(pvals[-1])
# continuous variables		
		for k in list(pfitc.keys()): 
			pnew[k] = numpy.random.normal(pvals[-1][k],pstep[k])
			pnew[k] = numpy.nanmin([pnew[k],numpy.nanmax(mdls[k])])
			pnew[k] = numpy.nanmax([pnew[k],numpy.nanmin(mdls[k])])
# discrete variables
		for k in list(pfitd.keys()): 
			vals = list(set(list(mdls[k])))
			pnew[k] = numpy.random.choice(vals)
		pnew = pnew | pfitd
		try:
			cmdl = getModel(mdls,pnew,spscl.wave,scale=False,verbose=verbose)
			if verbose==True: print(i,pnew)
			chinew,scl,_ = compareSpec(spscl.flux.value,cmdl.flux.value,spscl.noise.value,verbose=verbose)
			# if numpy.isnan(radius)==False and numpy.isnan(e_radius)==False:
			# 	chinew+=(((10.*u.pc*(scl**0.5)).to(u.Rsun).value-radius)/e_radius)**2
			#if 'scale' not in list(pnew.keys()): cmdl.scale(scl)

	# compute statistic
			if method=='chidiff': st,chst = (chinew-chis[-1])/numpy.nanmin(chis),numpy.random.uniform(0,threshhold)
			elif method=='survival': st,chst = 2*stats.f.sf(chinew/chis[-1],dof,dof),numpy.random.uniform(threshhold,1)
			elif method=='dofscale': st,chst = dof/(0.5*dof+chinew-chis[-1]),numpy.random.uniform(threshhold,1)
			else: raise ValueError('Do not recognize statistical comparison {}; try chidiff, survival, or dofscale'.format(method))
	#			if verbose==True: print(chinew,chis[-1],dof,st,chst)

			if st<chst:
	# reset if we've wandered off
				if chinew>(1+2*threshhold)*numpy.nanmin(chis):
					if verbose==True: print('RESETING TO BEST FIT')
					pvals.append(pvals[numpy.argmin(chis)])
					chis.append(chis[numpy.argmin(chis)])
					scales.append(scales[numpy.argmin(chis)])
					mdlflxs.append(mdlflxs[numpy.argmin(chis)])
	# criterion satisfied, make a move
				else:
					if verbose==True: print('CHANGED PARAMETERS!')
					pvals.append(pnew)
					chis.append(chinew)
					scales.append(scl)
					mdlflxs.append(cmdl.flux.value)
	# criterion not satisfied, stay in place
			else:
				pvals.append(pvals[-1])
				chis.append(chis[-1])
				scales.append(scales[-1])
				mdlflxs.append(mdlflxs[-1])
# model can't be read, stay in place
		except: 
			if verbose==True: print('Error reading in parameters {}'.format(pnew))
			pvals.append(pvals[-1])
			chis.append(chis[-1])
			scales.append(scales[-1])
			mdlflxs.append(mdlflxs[-1])
# iterim save
		if iterim>0 and i>0 and numpy.mod(i,iterim)==0 and report==True:
# save parameters
			dpfit = pandas.DataFrame()
			for k in mkys: 
				dpfit[k] = [p[k] for p in pvals]
			dpfit['chis'] = chis
			dpfit['dof'] = [dof]*len(dpfit)
			dpfit['scale'] = scales
			dpfit['radius'] = [(10.*u.pc*(x**0.5)).to(u.Rsun).value for x in dpfit['scale']]
			outfile = file_prefix+'_parameters.xlsx'
			dpfit.to_excel(outfile,index=False)
# plot comparison
			if verbose==True: print('Saving iterim plots')
			pbest = dict(dpfit.loc[numpy.argmin(dpfit['chis']),:])
#			pbest['radius'] = (10.*u.pc*(scales[numpy.argmin(chis)]**0.5)).to(u.Rsun).value
			cmdl = getModel(mdls,pbest,spscl.wave,scale=True,verbose=verbose)
			# print(scales[numpy.argmin(chis)],pbest['scale'],numpy.nanmax(cmdl.flux.value))
			# cmdl.scale(scales[numpy.argmin(chis)])
			# print(numpy.nanmax(cmdl.flux.value))
			label = '{} model '.format(mset)
			label+=r'$\chi^2_r$='+'{:.1f}\n'.format(numpy.nanmin(chis)/dof)
			label+='T={:.0f} '.format(pbest['teff'])
			label+='logg={:.2f} '.format(pbest['logg'])
			label+='z={:.2f} '.format(pbest['z'])
			outfile = file_prefix+'_compare.pdf'
			plotCompare(spscl,cmdl,outfile=outfile,clabel=label,absolute=absolute,verbose=verbose)
# plot cornerplot
			plotpars = copy.deepcopy(mkysfit)
			for k in plotpars:
				if isinstance(mdls.loc[0,k],str): plotpars.remove(k)
			if absolute==True: plotpars.append('radius')
			pltbest = [dpfit.loc[numpy.argmin(dpfit['chis']),x] for x in plotpars]
# NOTE: THIS IS ONE OPTION FOR WEIGHTING, COULD TRY OTHERS			
			weights = numpy.array(dof/(dof+dpfit['chis']-numpy.nanmin(dpfit['chis'])))
			outfile = file_prefix+'_corner.pdf'
			plotCorner(dpfit,plotpars,pbest,weights=weights,outfile=outfile,verbose=verbose)
# plot chains
			plotpars.append('chis')
			plotpars.append('scale')
			outfile = file_prefix+'_chains.pdf'
			plotChains(dpfit,plotpars,outfile=outfile,verbose=verbose)

# remove burn in
	pvalsb = pvals[int(burn*nstep):]
	dpfit = pandas.DataFrame()
	for k in mkys: 
		dpfit[k] = [p[k] for p in pvalsb]
	dpfit['chis'] = chis[int(burn*nstep):]
	dpfit['dof'] = [dof]*len(dpfit)
	dpfit['scale'] = scales[int(burn*nstep):]
	dpfit['radius'] = [(10.*u.pc*(x**0.5)).to(u.Rsun).value for x in dpfit['scale']]

# best fit parameters
	pbest = dict(dpfit.loc[numpy.argmin(dpfit['chis']),:])
	pvalsb[numpy.argmin(chis[int(burn*nstep):])]
	for x in [flux_name]:
		if x in list(pbest.keys()): del pbest[x]
	if verbose==True: print('Best parameters: {}'.format(pbest))
	cmdl = getModel(mdls,pbest,spscl.wave,verbose=verbose)
	if 'scale' not in list(pbest.keys()): cmdl.scale(scales[numpy.argmin(chis)])

# distribution of values
	pdist = {}
	dpars = copy.deepcopy(mkysfit)
	for k in dpars:
		if isinstance(pvalsb[0][k],str)==False: pdist[k] = numpy.nanquantile([p[k] for p in pvalsb],quantscale)
	if absolute==True: 
		pdist['radius'] = numpy.nanquantile([(10.*u.pc*(x**0.5)).to(u.Rsun).value for x in scales[int(burn*nstep):]],quantscale)

	if report == True:
# remove initial burn and save
		outfile = file_prefix+'_parameters.xlsx'
		if verbose==True: print('Saving database of prameters to {}'.format(outfile))
		dpfit.to_excel(outfile,index=False)
# plot comparison
		# cmdl = getModel(mdls,pbest,spscl.wave,verbose=verbose)
		# if 'scale' not in list(pbest.keys()): cmdl.scale(scales[numpy.argmin(chis)])
		label = '{} model '.format(mdls.loc[0,'model'])
		label+=r'$\chi^2_r$='+'{:.1f}\n'.format(chis[numpy.argmin(chis)]/dof)
		label+='T={:.0f} '.format(pbest['teff'])
		label+='logg={:.2f} '.format(pbest['logg'])
		label+='z={:.2f} '.format(pbest['z'])
		outfile = file_prefix+'_compare.pdf'
		if verbose==True: print('Plotting best fit comparison to {}'.format(outfile))
		if nsample<=0: 
			plotCompare(spscl,cmdl,outfile=outfile,clabel=label,absolute=absolute,verbose=verbose)
		else:
			plotCompareSample(spscl,mdls,dpfit,nsample=nsample,outfile=outfile,clabel=label,absolute=absolute,verbose=verbose)
# plot cornerplot
		plotpars = copy.deepcopy(mkysfit)
		for k in plotpars:
			if isinstance(mdls.loc[0,k],str): plotpars.remove(k)
		if absolute==True: plotpars.append('radius')
		pltbest = [dpfit.loc[numpy.argmin(dpfit['chis']),x] for x in plotpars]
# NOTE: THIS IS ONE OPTION FOR WEIGHTING, COULD TRY OTHERS			
		weights = numpy.array(dof/(dof+dpfit['chis']-numpy.nanmin(dpfit['chis'])))
		outfile = file_prefix+'_corner.pdf'
		if verbose==True: print('Plotting corner plot to {}'.format(outfile))
		plotCorner(dpfit,plotpars,pbest,weights=weights,outfile=outfile,verbose=verbose)
# plot chains
		plotpars.append('chis')
		plotpars.append('scale')
		outfile = file_prefix+'_chains.pdf'
		if verbose==True: print('Plotting chain plot to {}'.format(outfile))
		plotChains(dpfit,plotpars,outfile=outfile,verbose=verbose)

# return depending on output keyword
	if 'best' in output.lower(): return pbest
	elif 'spec' in output.lower(): return cmdl
	elif 'dist' in output.lower(): return pdist
	elif 'chain' in output.lower(): return pandas.DataFrame(pvalsb)
	else:
		return {'best': pbest, 'model': cmdl, 'distributions': pdist, 'chain': pandas.DataFrame(pvalsb)}


########################################################################
# PLOTTING FUNCTIONS
########################################################################

def plotCompare(sspec,cspec,outfile='',clabel='Comparison',absolute=False,xscale='linear',yscale='linear',
	figsize=[8,5],height_ratio=[5,1],scale=1.,fontscale=1,xlabel='Wavelength',ylabel='Flux',ylabel2='O-C',
	ylim=None,xlim=None,legend_loc=1,verbose=ERROR_CHECKING):

	sspec.scale(scale)
	cspec.scale(scale)
	diff = sspec.flux.value-cspec.flux.value

	# xlabel = r'Wavelength'+' ({:latex})'.format(sspec.wave.unit)
	# ylabel = r'F$_\lambda$'+' ({:latex})'.format(sspec.flux.unit)
	# if absolute==True: ylabel='Absolute '+ylabel
	strue = sspec.wave.value[numpy.isnan(sspec.flux.value)==False]
	wrng = [numpy.nanmin(strue),numpy.nanmax(strue)]

	plt.clf()
	fg, (ax1,ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': height_ratio}, sharex=True, figsize=figsize)
	ax1.step(sspec.wave.value,sspec.flux.value,'k-',linewidth=2,label=sspec.name)
	ax1.step(cspec.wave.value,cspec.flux.value,'m-',linewidth=4,alpha=0.5,label=clabel)
	ax1.legend(fontsize=12*fontscale,loc=legend_loc)
	ax1.plot([numpy.nanmin(sspec.wave.value),numpy.nanmax(sspec.wave.value)],[0,0],'k--')
	ax1.fill_between(sspec.wave.value,sspec.noise.value,-1.*sspec.noise.value,color='k',alpha=0.3)
	scl = numpy.nanmax(cspec.flux.value)
	scl = numpy.nanmax([scl,numpy.nanmax(sspec.flux.value)])
	if ylim==None: ax1.set_ylim([x*scl for x in [-0.1,1.3]])
	else: ax1.set_ylim(ylim)
	if yscale=='log':
#		ax1.set_ylim([x*scl for x in [1.e-2,2]])
		if ylim==None: ax1.set_ylim([numpy.nanmean(sspec.noise.value)/2.,2*scl])
	if xlim==None: xlim=wrng
	ax1.set_xlim(xlim)
	ax1.set_xscale(xscale)
	ax1.set_yscale(yscale)
	ax1.set_ylabel(ylabel,fontsize=12*fontscale)
	ax1.tick_params(axis="x", labelsize=0)
	ax1.tick_params(axis="y", labelsize=14*fontscale)

	ax2.step(sspec.wave.value,diff,'k-',linewidth=2)
	ax2.plot([numpy.nanmin(sspec.wave.value),numpy.nanmax(sspec.wave.value)],[0,0],'k--')
	ax2.fill_between(sspec.wave.value,sspec.noise.value,-1.*sspec.noise.value,color='k',alpha=0.3)
	scl = numpy.nanquantile(diff,[0.02,0.98])
	# ax2.set_ylim([2*sc for sc in scl])
	ax2.set_ylim([scl[0]-1.*(scl[1]-scl[0]),scl[1]+1.*(scl[1]-scl[0])])
	ax2.set_xlim(xlim)
	ax2.set_xscale(xscale)
	ax2.set_yscale(yscale)
	ax2.set_xlabel(xlabel,fontsize=16*fontscale)
	ax2.set_ylabel(ylabel2,fontsize=16*fontscale)
	ax2.tick_params(axis="x", labelsize=14*fontscale)
	ax2.tick_params(axis="y", labelsize=14*fontscale)
	plt.tight_layout()
	if outfile!='': plt.savefig(outfile)
	if verbose==True: plt.show()
	return

def plotCompareSample(spec,models,chain,nsample=50,relchi=1.2,method='samples',absolute=False,outfile='',
	clabel='Comparison',xlabel='Wavelength',ylabel='Flux',ylabel2='O-C',scale=1.,xscale='linear',yscale='linear',
	figsize=[8,5],height_ratio=[5,1],fontscale=1,ylim=None,xlim=None,legend_loc=1,verbose=ERROR_CHECKING):
# set up
	# xlabel = r'Wavelength'+' ({:latex})'.format(sspec.wave.unit)
	# ylabel = r'F$_\lambda$'+' ({:latex})'.format(sspec.flux.unit)
	# if absolute==True: ylabel='Absolute '+ylabel
	strue = spec.wave.value[numpy.isnan(spec.flux.value)==False]
	wrng = [numpy.nanmin(strue),numpy.nanmax(strue)]
	if nsample<0: nsample = int(len(chain)/10)

# first identify the best fit model
	pbest = dict(chain.loc[numpy.argmin(chain['chis']),:])
	cspec = getModel(models,pbest,spec.wave)
#	if 'scale' not in list(chain.columns): cspec.scale(scale)
#	cspec.scale(pbest['scale'])
# scale
	sspec = copy.deepcopy(spec)
	sspec.scale(scale)
#	print(numpy.nanmedian(sspec.flux.value))
	cspec.scale(scale)
	diff = sspec.flux.value-cspec.flux.value

# now identify the random sample
	chainsub = chain[chain['chis']/numpy.nanmin(chain['chis'])<relchi]
	chainsub.reset_index(inplace=True)
	nsamp = numpy.nanmin([nsample,len(chainsub)])
	fluxes = [getModel(models,dict(chainsub.loc[i,:]),sspec.wave).flux for i in numpy.random.randint(0,len(chainsub)-1,nsamp)]

# plot
	plt.clf()
	fg, (ax1,ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': height_ratio}, sharex=True, figsize=figsize)
	if method=='minmax':
		minflx = numpy.nanmin(fluxes,axis=0)*scale
		maxflx = numpy.nanmax(fluxes,axis=0)*scale
		# if 'scale' not in list(chainsub.columns):
		# 	minflx = minflx*scale 
		# 	maxflx = maxflx*scale 
		ax1.fill_between(sspec.wave.value,minflx,maxflx,color='m',alpha=0.2)
	elif method=='meanstd':
		meanflx = numpy.nanmean(fluxes,axis=0)*scale
		stdflx = numpy.nanstd(fluxes,axis=0)*scale
		# if 'scale' not in list(chainsub.columns):
		# 	meanflx = meanflx*scale 
			# stdflx = stdflx*scale 
		ax1.fill_between(sspec.wave.value,meanflx-stdflx,meanflx+stdflx,color='m',alpha=0.2)
	else:
		for f in fluxes: ax1.step(cspec.wave.value,f,'m-',linewidth=2,alpha=1/nsamp)
	ax1.step(sspec.wave.value,sspec.flux.value,'k-',linewidth=2,label=sspec.name)
	ax1.step(cspec.wave.value,cspec.flux.value,'m-',linewidth=2,alpha=0.7,label=clabel)
	ax1.legend(fontsize=12*fontscale,loc=legend_loc)
	ax1.plot([numpy.nanmin(sspec.wave.value),numpy.nanmax(sspec.wave.value)],[0,0],'k--')
	ax1.fill_between(sspec.wave.value,sspec.noise.value,-1.*sspec.noise.value,color='k',alpha=0.3)
	scl = numpy.nanmax(cspec.flux.value)
	scl = numpy.nanmax([scl,numpy.nanmax(sspec.flux.value)])
	if ylim==None: ax1.set_ylim([x*scl for x in [-0.1,1.3]])
	else: ax1.set_ylim(ylim)
	ax1.set_xscale(xscale)
	ax1.set_yscale(yscale)
	if yscale=='log':
		if ylim==None: ax1.set_ylim([x*scl for x in [1.e-2,2]])
	if xlim==None: xlim=wrng
	ax1.set_xlim(xlim)
	ax1.set_ylabel(ylabel,fontsize=12*fontscale)
	ax1.tick_params(axis="x", labelsize=0)
	ax1.tick_params(axis="y", labelsize=14*fontscale)

	ax2.step(sspec.wave.value,diff,'k-',linewidth=2)
	ax2.plot([numpy.nanmin(sspec.wave.value),numpy.nanmax(sspec.wave.value)],[0,0],'k--')
	ax2.fill_between(sspec.wave.value,sspec.noise.value,-1.*sspec.noise.value,color='k',alpha=0.3)
	scl = numpy.nanquantile(diff,[0.02,0.98])
	ax2.set_ylim([scl[0]-1.*(scl[1]-scl[0]),scl[1]+1.*(scl[1]-scl[0])])
	ax2.set_xlim(xlim)
	ax2.set_xscale(xscale)
	ax2.set_yscale(yscale)
	ax2.set_xlabel(xlabel,fontsize=16*fontscale)
	ax2.set_ylabel(ylabel2,fontsize=16*fontscale)
	ax2.tick_params(axis="x", labelsize=14*fontscale)
	ax2.tick_params(axis="y", labelsize=14*fontscale)
	plt.tight_layout()
	if outfile!='': plt.savefig(outfile)
	if verbose==True: plt.show()
	return

def plotChains(dpfit,plotpars,pbest={},outfile='',xlabel='Step',labeldict=PARAMETER_PLOT_LABELS,verbose=ERROR_CHECKING):
	nplot = int(len(plotpars))
	if nplot==0: 
		if verbose==True: print('WARNING: no parameters to plot')
		return
# set up plot
	plt.clf()
	fig = plt.figure(figsize=[2*6,numpy.ceil(nplot/2)*3])
	for i,l in enumerate(plotpars):	
		ax = plt.subplot(int(numpy.ceil(nplot/2)),2,i+1)
		ax.plot(dpfit[l],'k-')
# indicate current best fit parameter		
		if l in list(pbest.keys()): 
			ax.plot(numpy.zeros(len(dpfit[l]))+pbest[l],'b--')
# indicate best fit in chain		
		if 'chis' in list(dpfit.keys()):
#			print(l,dpfit.loc[numpy.argmin(dpfit['chis']),l])
			ax.plot(numpy.zeros(len(dpfit[l]))+dpfit.loc[numpy.argmin(dpfit['chis']),l],'m--')
			ax.plot([numpy.argmin(dpfit['chis']),numpy.argmin(dpfit['chis'])],[numpy.nanmin(dpfit[l]),numpy.nanmax(dpfit[l])],'m--')
			ax.set_title(dpfit.loc[numpy.argmin(dpfit['chis']),l])
# labels and ticks
		ax.set_xlabel(xlabel,fontsize=14)
		if l in list(labeldict.keys()): ax.set_ylabel(labeldict[l],fontsize=14)
		else: ax.set_ylabel(l,fontsize=14)
		ax.tick_params(axis="x", labelsize=14)
		ax.tick_params(axis="y", labelsize=14)
	plt.tight_layout()
	if outfile!='': fig.savefig(outfile)
	if verbose==True: plt.show()
	return

def plotCorner(dpfit,plotpars,pbest={},weights=[],outfile='',verbose=ERROR_CHECKING):
# choose plot columns
	ppars = copy.deepcopy(plotpars)
	for x in plotpars:
		if numpy.nanmin(dpfit[x])==numpy.nanmax(dpfit[x]): ppars.remove(x)
	if len(ppars)==0:
		if verbose==True: print('Warning: there are no parameters to plot!')
		return
# reorder
	ppars2 = []
	for k in list(PARAMETER_PLOT_LABELS.keys()):
		if k in ppars: ppars2.append(k)
	dpplot = dpfit[ppars2]
		
# weights
	if len(weights)<len(dpplot): weights=numpy.ones(len(dpplot))
	
# labels
	plabels=[]
	for k in ppars2:
		if k in list(PARAMETER_PLOT_LABELS.keys()): plabels.append(PARAMETER_PLOT_LABELS[k])
		else: plabels.append(k)

# best fit parameters
	truths = [numpy.nan for x in ppars2]
	if len(list(pbest.keys()))>0:
		for i,k in enumerate(ppars2):
			if k in list(pbest.keys()): truths[i]=pbest[k]

# generate plot
	plt.clf()
	fig = corner.corner(dpplot,quantiles=[0.16, 0.5, 0.84], labels=plabels, show_titles=True, weights=weights, \
						labelpad=0, title_kwargs={"fontsize": 14},label_kwargs={'fontsize': 14}, smooth=1,truths=truths, \
						truth_color='m',verbose=verbose)
	plt.tight_layout()
	if outfile!='': fig.savefig(outfile)
	if verbose==True: plt.show()
	return
