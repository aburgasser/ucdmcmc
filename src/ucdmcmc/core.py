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

# basic packages
import copy
import glob
import os
import sys

# external packages
#from astropy.coordinates import SkyCoord, EarthLocation, CartesianRepresentation, CartesianDifferential, Galactic, Galactocentric
import astropy.constants as const
from astropy.io import fits
import astropy.units as u
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
import pandas
import requests
from scipy.interpolate import griddata
import scipy.stats as stats
from statsmodels.stats.weightstats import DescrStatsW
from tqdm import tqdm


#######################################################
#######################################################
#################   INITIALIZATION  ###################
#######################################################
#######################################################


# code parameters
VERSION = '2025.08.20'
__version__ = VERSION
GITHUB_URL = 'http://www.github.com/aburgasser/ucdmcmc/'
ERROR_CHECKING = True
CODE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(CODE_PATH,'models/')
MODEL_URL = "http://spexarchive.coolstarlab.ucsd.edu/ucdmcmc"
ALT_MODEL_FOLDER = os.path.join(os.path.expanduser('~'),'.ucdmcmc_models','')
SPECTRA_FOLDER = os.path.join(CODE_PATH,'spectra/')
MODEL_FILE_PREFIX = 'models_'
WAVE_FILE_PREFIX = 'wave_'

# defaults
DEFAULT_WAVE_UNIT = u.micron
DEFAULT_FLUX_UNIT = u.erg/u.s/u.cm/u.cm/u.micron
DEFAULT_FLAM_UNIT = u.erg/u.s/u.cm/u.cm/u.micron
DEFAULT_FNU_UNIT = u.Jy
DEFAULT_WAVE_NAME = 'wave'
DEFAULT_FLUX_NAME = 'flux'
DEFAULT_NOISE_NAME = 'noise'
DEFAULT_PARAMETERS_NAME = 'noise'

# baseline wavelength grid
DEFAULT_WAVE_RANGE = [0.9,2.45]
DEFAULT_RESOLUTION = 300

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
	'EUCLID': {'instrument_name': 'EUCLID NISP', 'altname': [''], 'wave_range': [0.9,1.9]*u.micron, 'resolution': 350, 'bibcode': '', 'sample': '','sample_name': '', 'sample_bibcode': ''},
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
	'lacy23': {'instruments': {}, 'name': 'Lacy & Burrows (2023)', 'citation': 'Lacy & Burrows (2023)', 'bibcode': '2023ApJ...950....8L', 'altname': ['lacy2023','lac23','lacy'], 'default': {'teff': 500., 'logg': 4.5, 'z': 0., 'cld': 'NC', 'kzz': 0.}}, \
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

# check/setup ALT_MODEL_FOLDER
if os.path.exists(ALT_MODEL_FOLDER)==False:
	try: 
		os.mkdir(ALT_MODEL_FOLDER)
	except: 
		if ERROR_CHECKING==True: 
			print('Warning! Could not create model folder {}\nSet ucdmcmc.ALT_MODEL_FOLDER to your preferred model directory'.format(ALT_MODEL_FOLDER))



#######################################################
#######################################################
################  VARIOUS UTILITIES  ##################
#######################################################
#######################################################

# DATA DOWNLOADER
def downloadModel(file,url=MODEL_URL,targetdir=ALT_MODEL_FOLDER,verbose=ERROR_CHECKING):
# already have it?
	files = glob.glob(os.path.join(targetdir,file))
	if len(files)>0:
		if verbose==True: print('\n{} is already present in {}'.format(file,targetdir))
		return
# try to get it
	try:
		response = requests.get(os.path.join(url,file), stream=True)
		response.raise_for_status()  # Raise an exception for bad status codes
		with open(os.path.join(targetdir,file), 'wb') as f:
			for chunk in response.iter_content(chunk_size=8192): f.write(chunk)
		if verbose==True: print('\n{} downloaded from {}'.format(file,url))
	except requests.exceptions.RequestException as e:
		if verbose==True: print('\nError downloading {} from {}\n{}'.format(file,url,e))

	return


# GENERAL PURPOSE PROGRAM TO LOOK SOMETHING UP FROM A DICTIONARY
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


# CHECKS IF SOMETHING IS A UNIT
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


# CHECKS IF SOMETHING IS A NUMBER
def isNumber(s):
	'''
	:Purpose: Checks if something is a number.

	:param s: object to be checked
	:type s: required

	:Output: True or False

	:Example:
	>>> import splat
	>>> print splat.isNumber(3)
		True
	>>> print splat.isNumber('hello')
		False
	'''
	s1 = copy.deepcopy(s)
	if isinstance(s1,bool): return False
	if isinstance(s1,u.quantity.Quantity): s1 = s1.value
	if isinstance(s1,float): return (True and not np.isnan(s1))
	if isinstance(s1,int): return (True and not np.isnan(s1))
	try:
		s1 = float(s1)
		return (True and not np.isnan(s1))
	except:
		return False


# ROTATIONAL BROADENING KERNEL
def lsfRotation(vsini,vsamp,epsilon=0.6,verbose=ERROR_CHECKING):
	'''
	Purpose: 

		Generates a line spread function for rotational broadening, based on Gray (1992) 
		Ported over by Chris Theissen and Adam Burgasser from the IDL routine 
		`lsf_rotate <https://idlastro.gsfc.nasa.gov/ftp/pro/astro/lsf_rotate.pro>`_ writting by W. Landsman

	Required Inputs:  

		:param: **vsini**: vsini of rotation, assumed in units of km/s
		:param: **vsamp**: sampling velocity, assumed in unit of km/s. vsamp must be smaller than vsini or else a delta function is returned

	Optional Inputs:

		:param: **epsilon**: limb darkening parameter based on Gray (1992)

	Output:

		Line spread function kernel with length 2*vsini/vsamp (forced to be odd)

	:Example:
		>>> import splat
		>>> kern = lsfRotation(30.,3.)
		>>> print(kern)
			array([ 0.		,  0.29053574,  0.44558751,  0.55691445,  0.63343877,
			0.67844111,  0.69330989,  0.67844111,  0.63343877,  0.55691445,
			0.44558751,  0.29053574,  0.		])
	'''
# limb darkening parameters
	e1 = 2. * (1. - epsilon)
	e2 = np.pi * epsilon/2.
	e3 = np.pi * (1. - epsilon/3.)

# vsini must be > vsamp - if not, return a delta function
	if vsini <= vsamp:
		if verbose==True: print('\nWarning: velocity sampling {} is broader than vsini {}; returning delta function')  
		lsf = np.zeros(5)  
		lsf[2] = 1.
		return lsf

# generate LSF
	nsamp = np.ceil(2.*vsini/vsamp)
	if nsamp % 2 == 0:
		nsamp+=1
	x = np.arange(nsamp)-(nsamp-1.)/2.
	x = x*vsamp/vsini
	x2 = np.absolute(1.-x**2)

	return (e1*np.sqrt(x2) + e2*x2)/e3


# SPECTRUM READING ROUTINE
def readSpectrum(file,wave_unit=DEFAULT_WAVE_UNIT,flux_unit=DEFAULT_FLUX_UNIT,dimensionless=False,
	comment='#',file_type='',delimiter='',hdunum=0,waveheader=False,crval1='CRVAL1',cdelt1='CDELT1',
	wavelog=False,catch_sn=True,remove_nans=True,no_zero_noise=True,use_instrument_reader=True,
	instrument='',verbose=ERROR_CHECKING,wave=[],flux=[],noise=[],**kwargs):
	'''
	Purpose
	-------

	Reads in spectral data from a variety of formats

	'''

# prepare output
	output = {'wave':wave,'flux':flux,'noise':noise,'header':{}}

# check inputs and keyword parameters
	if file == '': raise NameError('\nNo filename passed to readSpectrum')
	if not(isUnit(wave_unit)):
		if verbose==True: print('Warning: wave_unit {} is not an astropy unit; using default {}'.format(wave_unit,DEFAULT_WAVE_UNIT))
		wave_unit = DEFAULT_WAVE_UNIT
	if not(isUnit(flux_unit)):
		if verbose==True: print('Warning: flux_unit {} is not an astropy unit; using default {}'.format(flux_unit,DEFAULT_FLUX_UNIT))
		flux_unit = DEFAULT_FLUX_UNIT
	if dimensionless==True: flux_unit = u.dimensionless_unscaled

# if a url, make sure it exists
	if file[:4]=='http':
		if requests.get(file).status_code!=requests.codes.ok:
			raise ValueError('Cannot find remote file {}; check URL or your online status'.format(file))

# if a local file, make sure it exists
	else:
		if os.path.exists(os.path.normpath(file)) == False: 
			raise ValueError('Cannot find file {}\n\n'.format(file))

# determine which type of file
	if file_type=='': file_type = file.split('.')[-1]

# zipped file - extract root
	for k in ['gz','bz2','zip']:
		if k in file_type:
			file_type = (file.replace('.'+k,'')).split('.')[-1]

# fits - can be done with fits.open as local or online and w/ or w/o gzip/bzip2/pkzip
	if 'fit' in file_type:
		with fits.open(os.path.normpath(file),ignore_missing_end=True,ignore_missing_simple=True,do_not_scale_image_data=True) as hdu:
			hdu.verify('silentfix+ignore')
			header = hdu[hdunum].header
			if 'NAXIS3' in list(header.keys()): d = np.copy(hdu[hdunum].data[0,:,:])
			else: d =  np.copy(hdu[hdunum].data)
# make sure file is oriented correctly
		if np.shape(d)[0]>np.shape(d)[1]: d = np.transpose(d)

# wavelength is in header 
		if waveheader==True and 'fit' in file_type and len(d[:,0])<3:
			flux = d[0,:]
			if crval1 in list(header.keys()) and cdelt1 in list(header.keys()):
				wave = np.polyval([float(header[cdelt1]),float(header[crval1])],np.arange(len(flux)))
			else: 
				raise ValueError('\nCannot find {} and {} keywords in header of fits file {}'.format(crval1,cdelt1,file))
# wavelength is explicitly in data array 
		else:
			wave = d[0,:]
			flux = d[1,:]
		if len(d[:,0]) > 2: noise = d[2,:]
		else: noise = [np.nan]*len(wave)

# ascii - can be done with pandas as local or online and w/ or w/o gzip/bzip2/pkzip
	else:
		if 'csv' in file_type and delimiter=='': delimiter = ','
		elif ('tsv' in file_type or 'txt' in file_type) and delimiter=='': delimiter = '\t'
		elif 'pipe' in file_type and delimiter=='': delimiter = '|'
		elif 'tex' in file_type and delimiter=='': delimiter = '&'
		else: delimiter = r'\s+'

# initial read
		dp = pandas.read_csv(file,delimiter=delimiter,comment=comment,header=0)
# if numbers in first row, replace with header
		if isNumber(dp.columns[0])==True:
			cnames = ['wave','flux']
			if len(dp.columns)>2: cnames.append('noise')
			if len(dp.columns)>3: 
				for i in range(len(dp.columns))-3: cnames.append('c{}'.format(i))
			dp = pandas.read_csv(file,delimiter=delimiter,comment=comment,names=cnames)
# assume order wave, flux, noise
		wave = np.array(dp[dp.columns[0]])
		flux = np.array(dp[dp.columns[1]])
		if len(dp.columns)>2: noise = np.array(dp[dp.columns[2]])
		else: noise = [np.nan]*len(dp)
# placeholder header
		header = fits.Header()	  # blank header

#  wavelength scale is logarithmic
	if 'wavelog'==True: wave = 10.**wave

# final output dictionary
	output['wave'] = np.array(wave)
	output['flux'] = np.array(flux) 
	output['noise'] = np.array(noise)
	output['header'] = header

# make sure arrays have units
	if not isUnit(output['wave']): output['wave'] = output['wave']*wave_unit
	output['wave'].to(wave_unit)
	if not isUnit(output['flux']): output['flux'] = output['flux']*flux_unit
	output['flux'].to(flux_unit)
	if not isUnit(output['noise']): output['noise'] = output['noise']*flux_unit
	output['noise'].to(flux_unit)

# remove all parts of spectrum that are nans
	if remove_nans==True:
		w = np.where(np.logical_and(np.isnan(output['wave']) == False,np.isnan(output['flux']) == False))
		output['wave'] = output['wave'][w]
		output['flux'] = output['flux'][w]
		output['noise'] = output['noise'][w]

# force places where noise is zero to be NaNs
	if no_zero_noise==True:
		output['noise'][np.where(output['noise'] == 0.)] = np.nan

	return output



#######################################################
#######################################################
###########  SPLAT-LIKE SPECTRUM CLASS  ###############
#######################################################
#######################################################

class Spectrum(object):
	'''
	:Description: 
		Class for containing spectral and source data from SpeX Prism Library.
		This is a temporary structure until astropy.specutils is completed

	'''

	def __init__(self, *args, verbose=ERROR_CHECKING, **kwargs):
		self.name = kwargs.get('name','')
		self.instrument = kwargs.get('instrument','')
		self.wave = kwargs.get('wave',np.array([]))
		self.flux = kwargs.get('flux',np.array([]))
		self.noise = kwargs.get('noise',np.array([]))
		self.filename = ''
		for x in ['file','filename','input']: self.filename = kwargs.get(x,self.filename)

# option 1: a filename is given
		if len(args) == 1:
			if isinstance(args[0],str)==True: self.filename = args[0]

# option 2: a pandas dataframe is given  
			elif isinstance(args[0],pandas.core.frame.DataFrame)==True:
				if 'wave' in list(args[0].columns) and 'flux' in list(args[0].columns):
					for k in list(args[0].columns): setattr(self,k,args[0][k])
				else: 
					if verbose==True: print('Passed a pandas array that is missing wave and/or flux columns')
					empty=True

# option 3: multiple lists or numpy arrays are given
# interpret as wave, flux, and optionally noise
		elif len(args) > 1:
			if (isinstance(args[0],list) or isinstance(args[0],np.ndarray)) and \
				(isinstance(args[1],list) or isinstance(args[1],np.ndarray)):
				self.wave = args[0]
				self.flux = args[1]
			else:
				if verbose==True: print('Multiple inputs need to be lists or numpy arrays')
				empty=True

			if len(args) > 2:
				if isinstance(args[2],list) or isinstance(args[2],np.ndarray):
					self.noise = args[2]
		else:
			pass

# read in file
		if len(self.wave)==0 and len(self.flux)==0 and self.filename != '':
			if self.name=='': self.name=self.filename

# read in spectrum, being careful not to overwrite specifically assigned quantities
			rs = readSpectrum(self.filename,**kwargs)
			if 'wave' in rs.keys():
				for k in list(rs.keys()): 
					if k not in list(kwargs.keys()): setattr(self,k.lower(),rs[k])

# None of this worked; create an empty Spectrum object (can be used for copying)
		if len(self.wave)==0 and len(self.flux)==0:
			if verbose==True: print('Warning: Creating an empty Spectrum object')
			return

# process spectral data
# convert to numpy arrays
		if not isinstance(self.wave,np.ndarray): self.wave = np.array(self.wave)
		if not isinstance(self.flux,np.ndarray): self.flux = np.array(self.flux)
#		if len(self.noise)==0: self.noise = [np.nan]*len(self.wave)
		if not isinstance(self.noise,np.ndarray): self.noise = np.array(self.noise)

# assure wave, flux, noise have units
		if not isUnit(self.wave): self.wave = np.array(self.wave)*self.wave.unit
		if not isUnit(self.flux): self.flux = np.array(self.flux)*self.flux.unit
		if not isUnit(self.noise): self.noise = np.array(self.noise)*self.flux.unit

# create a copy to store as the original
		self.original = copy.deepcopy(self)

		return

# copy function
	def __copy__(self):
		'''
		:Purpose: Make a copy of a Spectrum object
		'''
		s = type(self)()
		s.__dict__.update(self.__dict__)
		return s

# alt copy function
	def copy(self):
		'''
		:Purpose: Make a copy of a Spectrum object
		'''
		return self.__copy__()

# representation of spectrum
	def __repr__(self):
		'''
		:Purpose: A simple representation of the Spectrum object
		'''
		return '{} spectrum of {}'.format(self.instrument,self.name)

# map onto a given wavelength scale
	def toWavelengths(self,wave,force=True,verbose=False,**kwargs):
		'''
		:Purpose: 
			Maps a spectrum onto a new wavelength grid via interpolation or integral resampling

		:Required Inputs:
			:param wave: wavelengths to map to

		:Optional Inputs:
			:param force = True: proceed with conversion even if wavelength ranges are not perfectly in range
			:param verbose = False: provide verbose feedback
		
		:Outputs:
			None; Spectrum object is changed

		:Example:
		   TBD
		'''
		
		return resample(self,wave,verbose=verbose,**kwargs)


# addition
	def __add__(self,other):
		'''
		:Purpose: A representation of addition for Spectrum objects which correctly interpolates as a function of wavelength and combines variances

		:Output: a new Spectrum object equal to the spectral sum of the inputs

		:Example:
		   >>> import splat
		   >>> sp1 = splat.getSpectrum(lucky=True)[0]
		   >>> sp2 = splat.getSpectrum(lucky=True)[0]
		   >>> sp3 = sp1 + sp2
		   >>> sp3
			Spectrum of 2MASS J17373467+5953434 + WISE J174928.57-380401.6
		'''
# convert wavelength and flux units
		other.wave = other.wave.to(self.wave.unit)
		other.flux = other.flux.to(self.flux.unit,equivalencies=u.spectral_density(self.wave))
		other.noise = other.noise.to(self.flux.unit,equivalencies=u.spectral_density(self.wave))

# map to wavelength range
		other.toWavelengths(self.wave)

# # make a copy and identify wavelength range that is overlapping
		sp = copy.deepcopy(self)
#		 sp.wave = self.wave.value[np.where(np.logical_and(self.wave.value < np.nanmax(other.wave.value),self.wave.value > np.nanmin(other.wave.value)))]
#		 sp.wave=sp.wave*self.wave_unit

# # generate interpolated axes
#		 f1 = interp1d(self.wave.value,self.flux.value,bounds_error=False,fill_value=0.)
#		 f2 = interp1d(other.wave.value,other.flux.value,bounds_error=False,fill_value=0.)
#		 n1 = interp1d(self.wave.value,self.variance.value,bounds_error=False,fill_value=0.)
#		 n2 = interp1d(other.wave.value,other.variance.value,bounds_error=False,fill_value=0.)

# add & uncertainty
#		sp.flux = (f1(sp.wave.value)+f2(sp.wave.value))*self.flux.unit
		sp.flux = self.flux+other.flux
#		sp.noise = ((n1(sp.wave.value)+n2(sp.wave.value))*(self.flux.unit**2))**0.5
		sp.noise = ((self.noise.value**2+other.noise.value**2)**0.5)*self.flux.unit

# update information
		sp.name = self.name+' + '+other.name

# reset original
		sp.original = copy.deepcopy(sp)
		return sp

# subtraction
	def __sub__(self,other):
		'''
		:Purpose: A representation of subtraction for Spectrum objects which correctly interpolates as a function of wavelength and combines variances

		:Output: a new Spectrum object equal to the spectral difference of the inputs

		:Example:
		   >>> import splat
		   >>> sp1 = splat.getSpectrum(lucky=True)[0]
		   >>> sp2 = splat.getSpectrum(lucky=True)[0]
		   >>> sp3 = sp1 - sp2
		   >>> sp3
			Spectrum of 2MASS J17373467+5953434 - WISE J174928.57-380401.6
		'''
# convert wavelength and flux units
		other.wave = other.wave.to(self.wave.unit)
		other.flux = other.flux.to(self.flux.unit,equivalencies=u.spectral_density(self.wave))
		other.noise = other.noise.to(self.flux.unit,equivalencies=u.spectral_density(self.wave))

# map to wavelength range
		other.toWavelengths(self.wave)

# make a copy and fill in wavelength to be overlapping
		sp = copy.deepcopy(self)
#		 sp.wave = self.wave.value[np.where(np.logical_and(self.wave.value < np.nanmax(other.wave.value),self.wave.value > np.nanmin(other.wave.value)))]
# # this fudge is for astropy 1.*
#		 if not isUnit(sp.wave):
#			 sp.wave=sp.wave*self.wave.unit

# generate interpolated axes
		# f1 = interp1d(self.wave.value,self.flux.value,bounds_error=False,fill_value=0.)
		# f2 = interp1d(other.wave.value,other.flux.value,bounds_error=False,fill_value=0.)
		# n1 = interp1d(self.wave.value,self.variance.value,bounds_error=False,fill_value=0.)
		# n2 = interp1d(other.wave.value,other.variance.value,bounds_error=False,fill_value=0.)

# subtract & uncertainty
#		sp.flux = (f1(sp.wave.value)-f2(sp.wave.value))*self.flux.unit
		sp.flux = self.flux-other.flux
#		sp.noise = ((n1(sp.wave.value)+n2(sp.wave.value))*(self.flux.unit**2))**0.5
		sp.noise = ((self.noise.value**2+other.noise.value**2)**0.5)*self.flux.unit

# update information
		sp.name = self.name+' - '+other.name

# reset original
		sp.original = copy.deepcopy(sp)
		return sp


# multiplication
	def __mul__(self,other):
		'''
		:Purpose: A representation of multiplication for Spectrum objects which correctly interpolates as a function of wavelength and combines variances

		:Output: a new Spectrum object equal to the spectral product of the inputs

		:Example:
		   >>> import splat
		   >>> sp1 = splat.getSpectrum(lucky=True)[0]
		   >>> sp2 = splat.getSpectrum(lucky=True)[0]
		   >>> sp3 = sp1 * sp2
		   >>> sp3
			Spectrum of 2MASS J17373467+5953434 x WISE J174928.57-380401.6
		'''
# convert wavelength units
		other.wave = other.wave.to(self.wave.unit)

# map to wavelength range
		other.toWavelengths(self.wave)

# make a copy and fill in wavelength to be overlapping
		sp = copy.deepcopy(self)
		# sp.wave = self.wave.value[np.where(np.logical_and(self.wave.value < np.nanmax(other.wave.value),self.wave.value > np.nanmin(other.wave.value)))]
		# sp.wave=sp.wave*self.wave.unit

# generate interpolated axes
		# f1 = interp1d(self.wave.value,self.flux.value,bounds_error=False,fill_value=0.)
		# f2 = interp1d(other.wave.value,other.flux.value,bounds_error=False,fill_value=0.)
		# n1 = interp1d(self.wave.value,self.variance.value,bounds_error=False,fill_value=0.)
		# n2 = interp1d(other.wave.value,other.variance.value,bounds_error=False,fill_value=0.)

# multiply & uncertainty
#		sp.flux = np.multiply(np.array(f1(sp.wave.value)),np.array(f2(sp.wave.value)))*self.flux.unit*other.flux.unit
		sp.flux = np.multiply(sp.flux.value,sp.noise.value)*self.flux.unit*other.flux.unit
# uncertainty
		# sp.variance = np.multiply(sp.flux**2,((np.divide(n1(sp.wave.value),f1(sp.wave.value))**2)+(np.divide(n2(sp.wave.value),f2(sp.wave.value))**2)))
		# sp.variance=sp.variance*((self.flux.unit*other.flux.unit)**2)
		# sp.noise = sp.variance**0.5
		sp.noise = (np.multiply(sp.flux.value**2,((np.divide(self.noise.value,self.flux.value)**2)+(np.divide(other.noise.value,other.flux.value)**2)))**0.5)*self.flux.unit*other.flux.unit
		# sp.cleanNoise()

# update information
		sp.name = self.name+' x '+other.name

# reset original
		sp.original = copy.deepcopy(sp)
		return sp


# division
	def __div__(self,other):
		'''
		:Purpose: A representation of division for Spectrum objects which correctly interpolates as a function of wavelength and combines variances

		:Output: a new Spectrum object equal to the spectral ratio of the inputs

		:Example:
		   >>> import splat
		   >>> sp1 = splat.getSpectrum(lucky=True)[0]
		   >>> sp2 = splat.getSpectrum(lucky=True)[0]
		   >>> sp3 = sp1/sp2
		   >>> sp3
			Spectrum of 2MASS J17373467+5953434 + WISE J174928.57-380401.6
		'''
# convert wavelength units
		other.wave = other.wave.to(self.wave.unit)

# map to wavelength range
		other.toWavelengths(self.wave)

# make a copy and fill in wavelength to be overlapping
		sp = copy.deepcopy(self)
		# sp.wave = self.wave.value[np.where(np.logical_and(self.wave.value < np.nanmax(other.wave.value),self.wave.value > np.nanmin(other.wave.value)))]
		# sp.wave=sp.wave*self.wave.unit

# generate interpolated axes
		# f1 = interp1d(self.wave.value,self.flux.value,bounds_error=False,fill_value=0.)
		# f2 = interp1d(other.wave.value,other.flux.value,bounds_error=False,fill_value=0.)
		# n1 = interp1d(self.wave.value,self.variance.value,bounds_error=False,fill_value=0.)
		# n2 = interp1d(other.wave.value,other.variance.value,bounds_error=False,fill_value=0.)

# divide & uncertainty
#		sp.flux = np.divide(np.array(f1(sp.wave.value)),np.array(f2(sp.wave.value)))*(self.flux.unit/other.flux.unit)
		sp.flux = np.divide(sp.flux.value,sp.noise.value)*self.flux.unit/other.flux.unit
		# sp.variance = np.multiply(sp.flux**2,((np.divide(n1(sp.wave.value),f1(sp.wave.value))**2)+(np.divide(n2(sp.wave.value),f2(sp.wave.value))**2)))
		# sp.variance=sp.variance*((self.flux.unit/other.flux.unit)**2)
		# sp.noise = sp.variance**0.5
		sp.noise = (np.multiply(sp.flux.value**2,((np.divide(self.noise.value,self.flux.value)**2)+(np.divide(other.noise.value,other.flux.value)**2)))**0.5)*self.flux.unit*other.flux.unit

# clean up infinities
		sp.flux = (np.where(np.absolute(sp.flux.value) == np.inf, np.nan, sp.flux.value))*self.flux.unit/other.flux.unit
#		sp.cleanNoise()

# update information
		sp.name = self.name+' / '+other.name

# reset original
		sp.original = copy.deepcopy(sp)
		return sp


# division again
	def __truediv__(self,other):
		'''
		:Purpose: A representation of division for Spectrum objects which correctly interpolates as a function of wavelength and combines variances

		:Output: a new Spectrum object equal to the spectral ratio of the inputs

		:Example:
		   >>> import splat
		   >>> sp1 = splat.getSpectrum(lucky=True)[0]
		   >>> sp2 = splat.getSpectrum(lucky=True)[0]
		   >>> sp3 = sp1/sp2
		   >>> sp3
			Spectrum of 2MASS J17373467+5953434 + WISE J174928.57-380401.6
		'''
		return self/other


# export spectrum to file
	def export(self,filename='',clobber=True,delimiter='\t',save_header=True,save_noise=True,file_type='',comment='#',
		wave_name=DEFAULT_WAVE_NAME,flux_name=DEFAULT_FLUX_NAME,noise_name=DEFAULT_NOISE_NAME):
		'''
		:Purpose: 
			Exports a Spectrum object to either a fits or ascii file, depending on file extension given.  
			If no filename is explicitly given, the Spectrum.filename attribute is used. 
			If the filename does not include the full path, the file is saved in the current directory.  
			Spectrum.export and `Spectrum.save()`_ function in the same manner.

		.. _`Spectrum.save()` : api.html#splat.core.Spectrum.save

		:Required Inputs: 
			None

		:Optional Inputs: 
			:param filename: String specifying the filename to save; filename can also be included as an argument; if not provided, Spectrum.filename is used; alternate keywords: `file`
			:param clobber: Set to True to overwrite file, or False to raise flag if file exists (default = True) 
			:param csv: Set to True to write a CSV (comma-delimited) file (default = False) 
			:param tab: Set to True to write a tab-delimited file (default = True) 
			:param delimiter: character or string to specify as delimiter between columns (default = '\t'); alternate keywords: `sep` 
			:param save_header: set to True to add header to ascii files (default = True) 
			:param save_noise: set to True to save the noise column (default = True) 
			:param comment: use to specify comment character (default = '#') 

		:Output: 
			An ascii or fits file with the data and header

		:Example:
		   >>> import splat
		   >>> sp = splat.getSpectrum(lucky=True)[0]
		   >>> sp.export('/Users/adam/myspectrum.txt')
		   >>> from astropy.io import ascii
		   >>> data = ascii.read('/Users/adam/myspectrum.txt',format='tab')
		   >>> data
			<Table length=564>
			  wavelength		  flux		  uncertainty   
			   float64		  float64		   float64	 
			-------------- ----------------- -----------------
			0.645418405533			   0.0			   nan
			0.647664904594 6.71920214475e-16 3.71175052033e-16
			0.649897933006 1.26009925777e-15 3.85722895842e-16
			0.652118623257 7.23781818374e-16 3.68178778862e-16
			0.654327988625 1.94569566622e-15 3.21007116982e-16
			...
		'''

# prep inputs
		if len(args) > 0: filename = args[0]
		filename = kwargs.get('file',filename)
		if filename == '' and 'filename' in list(self.__dict__.keys()): filename = self.filename

		if filename == '':
			print('\nWarning! no filename provided, data were not saved')
			return

# determine which type of file
		if file_type == '': file_type = filename.split('.')[-1]

# fits file
		if 'fit' in file_type:
			data = np.vstack((self.wave.value,self.flux.value,self.noise.value))
			hdu = fits.PrimaryHDU(data)
			for k in list(self.header.keys()):
				if k.upper() not in ['HISTORY','COMMENT','BITPIX','NAXIS','NAXIS1','NAXIS2','EXTEND'] and k.replace('#','') != '': # and k not in list(hdu.header.keys()):
					hdu.header[k] = str(self.header[k])
			for k in list(self.__dict__.keys()):
				if isinstance(self.__getattribute__(k),str) == True or (isinstance(self.__getattribute__(k),float) == True and np.isnan(self.__getattribute__(k)) == False) or isinstance(self.__getattribute__(k),int) == True or isinstance(self.__getattribute__(k),bool) == True:
					hdu.header[k.upper()] = str(self.__getattribute__(k))
			hdu.writeto(filename,overwrite=clobber)

# ascii file - by default tab delimited
		else:
			if 'csv' in file_type: delimiter = ','
			if 'pipe' in file_type: delimiter = ' | '
			if 'tex' in file_type: delimiter = ' & '
			f = open(filename,'w')
			if save_header == True:
				for k in list(self.header.keys()):
					if k.upper() not in ['HISTORY','COMMENT'] and k.replace('#','') != '':
						f.write('{}{} = {}\n'.format(comment,k.upper(),self.header[k]))
				for k in list(self.__dict__.keys()):
					if isinstance(self.__getattribute__(k),str) == True or (isinstance(self.__getattribute__(k),float) == True and np.isnan(self.__getattribute__(k)) == False) or isinstance(self.__getattribute__(k),int) == True or isinstance(self.__getattribute__(k),bool) == True:
						f.write('{}{} = {}\n'.format(comment,k.upper(),self.__getattribute__(k)))
			if save_noise == True:
				f.write('{}{}{}{}{}{}\n'.format(comment,wave_name,delimiter,flux_name,delimiter,noise_name))
#				f.write('{}{}{}{}{}{}\n'.format(comment,self.wave.unit,delimiter,self.flux.unit,delimiter,self.noise.unit))
				for i in range(len(self.wave.value)): f.write('{}{}{}{}{}\n'.format(self.wave.value[i],delimiter,self.flux.value[i],delimiter,self.noise.value[i]))
			else:
				f.write('{}{}{}{}\n'.format(comment,wave_name,delimiter,flux_name))
#				f.write('{}{}{}{}\n'.format(comment,self.wave.unit,delimiter,self.flux.unit))
				for i in range(len(self.wave.value)): f.write('{}{}{}\n'.format(self.wave.value[i],delimiter,self.flux.value[i]))
			f.close()

		return


# convert to Fnu
	def toFnu(self,funit=DEFAULT_FNU_UNIT):
		'''
		:Purpose: 
			Converts flux density r'F\\_nu' in units of Jy.  
			There is no change if the spectrum is already in r'F\\_nu' units.

		:Required Inputs:
			None
		
		:Optional Inputs:
			None
		
		:Outputs:
			None; Spectrum object is changed
		
		:Example:
		   >>> import splat
		   >>> sp = splat.getSpectrum(lucky=True)[0]
		   >>> sp.toFnu()
		   >>> sp.flux.unit
			Unit("Jy")
		'''
		self.flux = self.flux.to(funit,equivalencies=u.spectral_density(self.wave))
		self.noise = self.noise.to(funit,equivalencies=u.spectral_density(self.wave))
		return

# convert to Flam
	def toFlam(self,funit=DEFAULT_FLAM_UNIT):
		'''
		:Purpose: 
			Converts flux density to r'F\\_lambda' in units of r'erg/s/cm\\^2/Hz'. 
			There is no change if the spectrum is already in r'F\\_lambda' units.
		
		:Required Inputs:
			None
		
		:Optional Inputs:
			None
		
		:Outputs:
			None; Spectrum object is changed
		
		:Example:
		   >>> import splat
		   >>> sp = splat.getSpectrum(lucky=True)[0]
		   >>> sp.toFnu()
		   >>> sp.flux.unit
			Unit("Jy")
		   >>> sp.toFlam()
		   >>> sp.flux.unit
			Unit("erg / (cm2 micron s)")
		'''
		self.flux = self.flux.to(funit,equivalencies=u.spectral_density(self.wave))
		self.noise = self.noise.to(funit,equivalencies=u.spectral_density(self.wave))
		return


# shift by pixel or wavelength
	def shift(self,s):
		'''
		:Purpose: Shifts the wavelength scale by a given radial velocity. This routine changes the underlying Spectrum object.
		
		:Example:
		   >>> import splat
		   >>> import astropy.units as u
		   >>> sp.rvShift(15*u.km/u.s)
		'''
		if isUnit(s): 
			s.to(self.wave.unit)
			self.wave = self.wave.value+s
		else: 
			self.wave = np.roll(self.wave,s)
			if s > 0: self.wave[:s] = np.nan
			if s < 0: self.wave[s:] = np.nan
		return


# shift by radial velocity
	def rvShift(self,rv):
		'''
		:Purpose: Shifts the wavelength scale by a given radial velocity. This routine changes the underlying Spectrum object.
		
		:Example:
		   >>> import splat
		   >>> import astropy.units as u
		   >>> sp.rvShift(15*u.km/u.s)
		'''
		if not isUnit(rv): rv=rv*(u.km/u.s)
		rv.to(u.km/u.s)
		self.wave = self.wave*((1.+(rv/const.c).to(u.m/u.m)))
		return


# rotational or gaussian broaden		
	def broaden(self,vbroad,kern=None,epsilon=0.6,method='rotation',verbose=False):
		'''
		:Purpose: 

			Broadens a spectrum in velocity space using a line spread function (LSF) either based on rotation or gaussian. 
			This routine changes the underlying Spectrum object.

		:Required Inputs:

			:param vbroad: broadening width, nominally in km/s
			
		:Optional Inputs:

			:param method: method of broadening, should be one of:

				- ``gaussian``: (default) Gaussian broadening, with vbroad equal to Gaussian sigma
				- ``rotation``: rotational broadening using splat.lsfRotation()
				- ``delta``: Delta function (no broadening)

			:param kern: input kernel, must be at least three elements wide (default = None)
			:param epsilon: epsilon parameter for limb darkening in rotational broadening (default = 0.6)
			:param verbose: provide extra feedback (default = False)

		:Outputs:

			None; Spectral flux is smoothed using the desired line spread function. No change is made to noise or other axes
			
		:Example:
		   >>> import splat
		   >>> sp = splat.Spectrum(10001)
		   >>> sp.broaden(30.,method='rotation')
		   >>> sp.info()
			History:
				SPEX_PRISM spectrum successfully loaded
				Rotationally broadened spectrum by 30.0 km/s
		'''
		report = ''
# determine velocity sampling
		if not isUnit(vbroad): vbroad=vbroad*(u.km/u.s)
		vbroad.to(u.km/u.s)
		samp = np.nanmedian(np.absolute(self.wave.value-np.roll(self.wave.value,1)) / self.wave.value)
		vsamp = (samp*const.c).to(u.km/u.s)

# velocity resolution is too low - use a delta function
		if kern != None:
			if len(kern) < 3:
				if verbose==True: print('\nWarning: input kernel {} must be at least three elements; setting to delta function'.format(kern))
				kern = None
				method = 'delta'

		if kern == None:
			if vsamp > vbroad:
				if verbose==True: print('\nWarning: velocity resolution {} is smaller than velocity broadening {}; setting to delta function'.format(vsamp,vbroad))
				method = 'delta'

# rotational broadening
			if 'rot' in method.lower():
				kern = lsfRotation(vbroad.value,vsamp.value,epsilon=epsilon)
				report = 'Rotationally broadened spectrum by {}'.format(vbroad)

# NOTE: THIS IS CURRENTLY NOT FUNCTIONAL
# gaussian ±10 sigma
			elif 'gauss' in method.lower():
				n = np.ceil(20.*vbroad.value/vsamp.value)
				if n%2==0: n+=1
				x = np.arange(n)-0.5*(n-1.)
				kern = np.exp(-0.5*(x**2))
				report = 'Broadened spectrum using a Gaussian with velocity width {}'.format(vbroad)

# delta function (no smoothing)
			else:
				kern = np.zeros(5)
				kern[2] = 1.
				report = 'Applying delta line spread function (no broadening)'

		else:
				report = 'Broadened spectrum using a input line spread function'

# normalize kernel
		kern = kern/np.nansum(kern)

# apply kernel
#		flux_unit = self.flux.unit
		a = (np.nanmax(self.wave.value)/np.nanmin(self.wave.value))**(1./len(self.wave))
		nwave = np.nanmin(self.wave.value)*(a**np.arange(len(self.wave)))
		nflux = self.flux.value*nwave
		ncflux = np.convolve(nflux, kern, 'same')
		self.flux = (ncflux/nwave)*self.flux.unit
		if verbose==True: print(report)

		return


# scale spectrum
	def scale(self,factor,noiseonly=False):
		'''
		:Purpose: 

			Scales a Spectrum object's flux and noise values by a constant factor. 


		:Required Inputs:

			:param factor: A floating point number used to scale the Spectrum object

		:Optional Inputs:

			:param noiseonly = False: scale only the noise and variance, useful when uncertainty is under/over estimated

		:Output: 

			None; spectrum is scaled

		:Example:
		   >>> import splat
		   >>> sp = splat.getSpectrum(lucky=True)[0]
		   >>> sp.fluxMax()
		   <Quantity 1.0577336634332284e-14 erg / (cm2 micron s)>
		   >>> sp.computeSN()
		   124.5198
		   >>> sp.scale(1.e15)
		   >>> sp.fluxMax()
		   <Quantity 1.0577336549758911 erg / (cm2 micron s)>
		   >>> sp.computeSN()
		   124.51981
		'''
		self.noise = self.noise*factor
		if noiseonly == False: self.flux = self.flux*factor
		return


# normalize
	def normalize(self,limits=[],method='median',verbose=ERROR_CHECKING):
		'''
		:Purpose: 
			Normalize a spectrum to a maximum value of 1 (in its current units) either at a 
			particular wavelength or over a wavelength range
		:Required Inputs: 
			None
		:Optional Inputs: 
			:param wave_range: choose the wavelength range to normalize; can be a list specifying minimum and maximum or a single wavelength (default = None); alternate keywords: `wave_range`, `range`
		:Output: 
			None; spectrum is normalized
		:Example:
		   >>> import splat
		   >>> sp = splat.getSpectrum(lucky=True)[0]
		   >>> sp.normalize()
		   >>> sp.fluxMax()
		   <Quantity 1.0 erg / (cm2 micron s)>
		   >>> sp.normalize(waverange=[2.25,2.3])
		   >>> sp.fluxMax()
		   <Quantity 1.591310977935791 erg / (cm2 micron s)>
		'''
		if len(limits) == 0:
			limits = [np.nanmin(self.wave.value),np.nanmax(self.wave.value)]
		elif len(limits) >= 2:
			if not isinstance(limits,list) and not isinstance(limits,np.ndarray):
				limits = [limits]
			if isUnit(limits[0]): limits = [r.to(self.wave.unit).value for r in limits]
			if isUnit(limits): limits = limits.to(self.wave.unit).value
			if np.nanmax(limits) > np.nanmax(self.wave.value) or np.nanmin(limits) < np.nanmin(self.wave.value):
				if verbose==True: print('\nWarning: normalization range {} is outside range of spectrum wave array: {}'.format(limits,[np.nanmin(self.wave.value),np.nanmax(self.wave.value)]))
# method
			if method in ['mean','average','ave']: scalefactor = np.nanmax(self.flux.value[np.where(np.logical_and(self.wave.value >= limits[0],self.wave.value <= limits[1]))])
			elif method in ['max','maximum']: scalefactor = np.nanmax(self.flux.value[np.where(np.logical_and(self.wave.value >= limits[0],self.wave.value <= limits[1]))])
			elif method in ['min','minimum']: scalefactor = np.nanmin(self.flux.value[np.where(np.logical_and(self.wave.value >= limits[0],self.wave.value <= limits[1]))])
			elif method in ['mode']: scalefactor = np.nanmode(self.flux.value[np.where(np.logical_and(self.wave.value >= limits[0],self.wave.value <= limits[1]))])
			else: scalefactor = np.nanmedian(self.flux.value[np.where(np.logical_and(self.wave.value >= limits[0],self.wave.value <= limits[1]))])
# single value
		else:
			f = interp1d(self.wave.value,self.flux.value)
			scalefactor = f(limits[0])
		if isUnit(scalefactor): scalefactor = scalefactor.value
		if scalefactor == 0. and verbose==True: print('\nWarning: normalize is attempting to divide by zero; ignoring')
		elif np.isnan(scalefactor) == True and verbose==True: print('\nWarning: normalize is attempting to divide by nan; ignoring')
		else: self.scale(1./scalefactor)
		return


# scale spectrum by filter magnitude
# NOTE: requires splat.photometry package
	def filterMag(self,filt,vebrose=ERROR_CHECKING,**kwargs):
		'''
		:Purpose: 

			Wrapper for `filterMag()`_ function in splat.photometry

		.. _`filterMag()` : api.html#splat.photometry.filterMag
		
		Required Inputs:

			**filter**: string specifiying the name of the filter

		Optional Inputs:

			See `filterMag()`_

		Outputs:

			Returns tuple containing filter-based spectrophotometic magnitude and its uncertainty

		:Example:
		   >>> import splat
		   >>> sp = splat.getSpectrum(lucky=True)[0]
		   >>> sp.fluxCalibrate('2MASS J',15.0)
		   >>> sp.filterMag(sp,'2MASS J')
			(15.002545668628173, 0.017635234089677564)
		'''

# for now leaning on splat.photometry package
		try: import splat.photometry as sphot
		except: 
			print('You must have the splat.photometry package installed for this filterMag()\nhttps://github.com/aburgasser/splat')
			return
		return sphot.filterMag(self,filt,**kwargs)


# scale spectrum by filter magnitude
# NOTE: requires splat.photometry package
	def fluxCalibrate(self,filt,mag,verbose=ERROR_CHECKING,**kwargs):
		'''
		:Purpose: Flux calibrates a spectrum given a filter and a magnitude. The filter must be one of those listed in `splat.FILTERS.keys()`. It is possible to specifically set the magnitude to be absolute (by default it is apparent).  This function changes the Spectrum object's flux, noise and variance arrays.
		
		Required Inputs:

		:param filt: string specifiying the name of the filter
		:param mag: number specifying the magnitude to scale to 

		Optional Inputs:

		:param absolute: set to True to specify that the given magnitude is an absolute magnitude, which sets the ``flux_label`` keyword in the Spectrum object to 'Absolute Flux' (default = False)
		:param apparent: set to True to specify that the given magnitude is an apparent magnitude, which sets the ``flux_label`` flag in the Spectrum object to 'Apparent Flux' (default = False)

		Outputs:

		None, Spectrum object is changed to a flux calibrated spectrum

		:Example:
		   >>> import splat
		   >>> sp = splat.getSpectrum(lucky=True)[0]
		   >>> sp.fluxCalibrate('2MASS J',15.0)
		   >>> splat.filterMag(sp,'2MASS J')
			(15.002545668628173, 0.017635234089677564)
		'''

# for now leaning on splat.photometry package
		try: import splat.photometry as sphot
		except: 
			print('You must have the splat.photometry package installed for this fluxCalibrate()\nhttps://github.com/aburgasser/splat')
			return
# get magnitude for filter
		apmag,apmag_e = sphot.filterMag(self,filt,**kwargs)
# NOTE: NEED TO INCORPORATE UNCERTAINTY INTO SPECTRAL UNCERTAINTY
		if np.isnan(apmag)==False:
			self.scale(10.**(0.4*(apmag-mag)))
		return

# simple plot routine
	def plot(self,outfile='',xscale='linear',yscale='linear',figsize=[8,5],fontscale=1,
		xlabel='Wavelength',ylabel='Flux',ylim=None,xlim=None,legend_loc=1,verbose=ERROR_CHECKING):
		'''
		:Purpose: 

			calls the `plotSpectrum()`_ function, by default showing the noise spectrum and zeropoints. 
			See the `plotSpectrum()`_ API listing for details.

		.. _`plotSpectrum()`: api.html#splat.plot.plotSpectrum

		:Output: A plot of the Spectrum object

		:Example:
		   >>> import splat
		   >>> sp = splat.getSpectrum(lucky=True)[0]
		   >>> sp.plot()
		'''
		strue = self.wave.value[np.isnan(self.flux.value)==False]
		wrng = [np.nanmin(strue),np.nanmax(strue)]

		plt.clf()
		plt.figure(figsize=figsize)
		plt.step(self.wave.value,self.flux.value,'k-',linewidth=2,label=self.name)
		plt.legend(fontsize=14*fontscale,loc=legend_loc)
		plt.plot([np.nanmin(self.wave.value),np.nanmax(self.wave.value)],[0,0],'k--')
		plt.fill_between(self.wave.value,self.noise.value,-1.*self.noise.value,color='k',alpha=0.3)
		plt.xscale(xscale)
		plt.yscale(yscale)
		if ylim==None:
			scl = np.nanmax(self.flux.value)
			if yscale=='log': ylim = [np.nanmean(self.noise.value)/2.,2*scl]
			else: ylim = [x*scl for x in [-0.1,1.3]]
		if xlim==None: xlim=wrng
		plt.ylim(xlim)
		plt.ylim(ylim)
		plt.xlabel(xlabel,fontsize=14*fontscale)
		plt.ylabel(ylabel,fontsize=14*fontscale)
		plt.xticks(fontsize=12*fontscale)
		plt.yticks(fontsize=12*fontscale)
		plt.tight_layout()
		if outfile!='': plt.savefig(outfile)
		else: plt.show()
		return


# reddening
	def redden(self, av=0.0, rv=3.1, normalize=False, verbose=ERROR_CHECKING,):
		'''
		:Purpose:

			Redden a spectrum using astandard interstellar profile
			from Cardelli, Clayton, and Mathis (1989 ApJ. 345, 245)

		:Required Inputs:

			None

		:Optional Inputs:

			:param av: Magnitude of reddening A_V (default = 0.)
			:param rv: Normalized extinction parameter, R_V = A(V)/E(B-V) (default = 3.1
			:param normalized: Set to True to normalize reddening function (default = False)

		:Outputs:

			None; spectral flux is changed

		:Example:

		   >>> import splat
		   >>> sp = splat.Spectrum(10001)				   # read in a source
		   >>> spr = splat.redden(sp,av=5.,rv=3.2)		  # redden to equivalent of AV=5

		'''
		w = (self.wave.to(u.micron)).value # micron assumed
		x = 1./w
		a = 0.574*(x**1.61)
		b = -0.527*(x**1.61)
		absfrac = 10.**(-0.4*av*(a+b/rv))
		report = 'Reddened following Cardelli, Clayton, and Mathis (1989) using A_V = {} and R_V = {}'.format(av,rv)

		if normalize == True:
			absfrac = absfrac/np.median(absfrac)
			report = report+' and normalized'

		self.flux = np.array(self.flux.value)*np.array(absfrac)*self.flux.unit
		self.noise = np.array(self.noise.value)*np.array(absfrac)*self.noise.unit
		if verbose==True: print(report)

		return


# reset to original form
	def reset(self):
		'''
		:Purpose: 

			Restores a Spectrum to its original read-in state, removing scaling and smoothing. 

		:Required Inputs:

			None
		
		:Optional Inputs:

			None

		:Output:

			Spectrum object is restored to original parameters
		
		:Example:
		   >>> import splat
		   >>> sp = splat.getSpectrum(lucky=True)[0]
		   >>> sp.fluxMax()
		   <Quantity 4.561630292384622e-15 erg / (cm2 micron s)>
		   >>> sp.normalize()
		   >>> sp.fluxMax()
		   <Quantity 0.9999999403953552 erg / (cm2 micron s)>
		   >>> sp.reset()
		   >>> sp.fluxMax()
		   <Quantity 4.561630292384622e-15 erg / (cm2 micron s)>
		'''
		for k in list(self.original.__dict__.keys()):
			if k != 'history':
				try: setattr(self,k,getattr(self.original,k))
				except: pass

		self.original = copy.deepcopy(self)
		return

# sample part of spectrum
	def sample(self,rng,method='median',verbose=ERROR_CHECKING):
		'''
		:Purpose: 
			Obtains a sample of spectrum over specified wavelength range

		:Required Inputs: 

			:param range: the range(s) over which the spectrum is sampled
			a single 2-element array or array of 2-element arrays

		:Optional Inputs: 

			None

		:Example:
			TBD
		'''

# single number = turn into small range
		if isinstance(rng,float):
			rng = [rng-0.01*(np.nanmax(self.wave.value)-np.nanmin(self.wave.value)),rng+0.01*(np.nanmax(self.wave.value)-np.nanmin(self.wave.value))]

		if not isinstance(rng,list): rng = list(rng)

		if isUnit(rng[0]):
			try: rng = [r.to(self.wave.unit).value for r in rng]
			except: raise ValueError('Could not convert trim range unit {} to spectrum wavelength unit {}'.format(rng.unit,self.wave.unit))

		w = np.where(np.logical_and(self.wave.value >= rng[0],self.wave.value <= rng[1]))
		if len(w[0])>0:
			if method.lower() in ['median','med']: val = np.nanmedian(self.flux.value[w])
			elif method.lower() in ['mean','average','ave']: val = np.nanmean(self.flux.value[w])
			elif method.lower() in ['max','maximum']: val = np.nanmax(self.flux.value[w])
			elif method.lower() in ['min','minimum']: val = np.nanmin(self.flux.value[w])
			elif method.lower() in ['std','stddev','stdev','rms']: val = np.nanstd(self.flux.value[w])
			elif method.lower() in ['unc','uncertainty','noise','error']: val = np.nanmedian(self.noise.value[w])
			elif method.lower() in ['sn','snr','signal-to-noise','s/n']: val = np.nanmedian(self.flux.value[w]/self.noise.value[w])
			else: raise ValueError('Did not recongize sampling method {}'.format(method))
			return val
		else:
			if verbose==True: print('Sampling range {} outside wavelength range of data'.format(rng))
			return np.nan


	def trim(self,rng,**kwargs):
		'''
		:Purpose: 
			Trims a spectrum to be within a certain wavelength range or set of ranges. 
			Data outside of these ranges are excised from the wave, flux and noise arrays. 
			The full spectrum can be restored with the reset() procedure.

		:Required Inputs: 

			:param range: the range(s) over which the spectrum is retained - a series of nested 2-element arrays

		:Optional Inputs: 

			None

		:Example:
		   >>> import splat
		   >>> sp = splat.getSpectrum(lucky=True)[0]
		   >>> sp.smoothfluxMax()
		   <Quantity 1.0577336634332284e-14 erg / (cm2 micron s)>
		   >>> sp.computeSN()
		   124.5198
		   >>> sp.scale(1.e15)
		   >>> sp.fluxMax()
		   <Quantity 1.0577336549758911 erg / (cm2 micron s)>
		   >>> sp.computeSN()
		   124.51981
		'''

		mask = np.zeros(len(self.wave))

# some code to deal with various possibilities, ultimately leading to [ [r1a,r1b], [r2a,r2b], ...]
# convert a unit-ed quantity
		if isUnit(rng):
			try: rng.to(self.wave.unit).value
			except: raise ValueError('Could not convert trim range unit {} to spectrum wavelength unit {}'.format(rng.unit,self.wave.unit))

# single number = turn into small range
		if isinstance(rng,float):
			rng = [rng-0.01*(np.nanmax(self.wave.value)-np.nanmin(self.wave.value)),rng+0.01*(np.nanmax(self.wave.value)-np.nanmin(self.wave.value))]

		if isUnit(rng[0]):
			try: rng = [r.to(self.wave.unit).value for r in rng]
			except: raise ValueError('Could not convert trim range unit {} to spectrum wavelength unit {}'.format(rng[0].unit,self.wave.unit))

		if not isinstance(rng[0],list): rng = [rng]

		for r in rng:
			if isUnit(r[0]):
				try: r = [x.to(self.wave.unit).value for x in r]
				except: raise ValueError('Could not convert trim range unit {} to spectrum wavelength unit {}'.format(r[0].unit,self.wave.unit))
			w = np.where(np.logical_and(self.wave.value > r[0],self.wave.value < r[1]))
		self.wave = self.wave[w]
		self.flux = self.flux[w]
		self.noise = self.noise[w]
		return




#######################################################
#######################################################
#################  MODELSET CLASS  ####################
#######################################################
#######################################################

class Modelset(object):
	'''
	:Description: 
		Class for containing spectral models including wavelength array
		Main elements are wavelength array, flux grid, and parameter grid
		Convenience functions for manipulating spectral fluxes
	'''

	def __init__(self, *args, **kwargs):
		'''
		Initializes a model set
		'''
		self.wave = kwargs.get('wave',np.array([]))
		self.flux = kwargs.get('flux',np.array([]))
		self.parameters = kwargs.get('parameters',pandas.DataFrame())
		self.modelname = ''
		for x in ['name','model','modelname']: self.modelname = kwargs.get(x,self.modelname)
		self.instrument = ''
		for x in ['instrument','inst','instr']: self.instrument = kwargs.get(x,self.instrument)
		self.filename = ''
		for x in ['file','filename','input','fluxfile']: self.filename = kwargs.get(x,self.filename)
		self.wavefile = kwargs.get('wavefile','')
		self.model_parameters = kwargs.get('model_parameters',{})
		self.instrument_parameters = kwargs.get('instrument_parameters',{})

		prefix = kwargs.get('prefix',MODEL_FILE_PREFIX)
		waveprefix = kwargs.get('waveprefix',WAVE_FILE_PREFIX)
		url = kwargs.get('url',MODEL_URL)
		wavecol = kwargs.get('wavecol',DEFAULT_WAVE_NAME)
		fluxcol = kwargs.get('fluxcol',DEFAULT_FLUX_NAME)
		waveunit = kwargs.get('waveunit',DEFAULT_WAVE_UNIT)
		fluxunit = kwargs.get('fluxunit',DEFAULT_FLUX_UNIT)

# READ IN FLUXES AND PARAMETERS
# one string argument - assume it is the filename
		if len(args)>0 and self.filename=='':
			if isinstance(args[0],str)==True:
				self.filename=args[0]

# no arguments - assume file name is constructed from model and instrument
		if self.modelname != '' and self.instrument != '' and self.filename=='':
			self.filename='{}{}_{}.h5'.format(prefix,self.modelname,self.instrument)

# we have a filename! read it in if present after searching relevant folders
		if self.filename!='':
			tmp = self.filename
			if os.path.exists(tmp)==False: tmp = os.path.join(MODEL_FOLDER,tmp)
			if os.path.exists(tmp)==False: tmp = tmp.replace(MODEL_FOLDER,ALT_MODEL_FOLDER)
# if still not present try to download from url
			if os.path.exists(tmp)==False: 
				downloadModel(self.filename,os.path.join(url,'models',''),ALT_MODEL_FOLDER,verbose=ERROR_CHECKING)
			if os.path.exists(tmp)==False: 
				raise ValueError('Could not locate model file {}'.format(self.filename))
			self.filename=tmp
# read in based on file type
			ftype = (self.filename.split('.'))[-1]
# default method: .h5 file with parameters and fluxes
			if ftype in ['h5']:
				dpm = pandas.read_hdf(self.filename)
				if fluxcol not in list(dpm.columns): raise ValueError('Flux column name {} not present in data array; specifiy the correct column name with keyword `fluxcol=`'.format(fluxcol))
				self.flux = np.array([np.array(x) for x in dpm[fluxcol]])*fluxunit
				for x in [fluxcol,'instrument','bibcode','model','modelname']:
					if x in list(dpm.columns): del dpm[x]
				self.parameters = dpm
			else: raise ValueError('Modelset initiation only set up for .h5 files; stay tuned')

# fluxes and parameters are passed
		elif len(self.flux) == 0 or len(self.parameters) == 0:
			raise ValueError('Modelset initiation requires filename or flux and parameter inputs, something is missing')

# READ IN WAVELENGTH ARRAY
		if len(self.wave)==0:
			if self.wavefile=='': 
				self.wavefile = '{}{}.csv'.format(waveprefix,self.instrument)
				tmp = self.wavefile
				if os.path.exists(tmp)==False: tmp = os.path.join(MODEL_FOLDER,tmp)
				if os.path.exists(tmp)==False: tmp = tmp.replace(MODEL_FOLDER,ALT_MODEL_FOLDER)
# if still not present try to download from url
				if os.path.exists(tmp)==False: 
					downloadModel(self.wavefile,os.path.join(url,'models',''),ALT_MODEL_FOLDER,verbose=ERROR_CHECKING)
				if os.path.exists(tmp)==False: 
					if ERROR_CHECKING==True:
						print('Cannot find wavefile {}; check filename or pass wavelength array')
				else:
					self.wavefile = tmp
# read in based on file type
					ftype = (self.wavefile.split('.'))[-1]
					delimiter = ''
# default method: .h5 file with parameters and fluxes
					if ftype in ['csv']:
						delimiter = ','
					elif ftype in ['txt','tsv']:
						delimiter = r'\s+'
					else: 
						if ERROR_CHECKING==True:
							print('Wave file read in limited to .csv, .txt, and .tsv files')
					if delimiter != '':
						dpw = pandas.read_csv(self.wavefile,delimiter=delimiter)
						cols = list(dpw.columns)
						if isNumber(cols[0]):
							dpw = pandas.read_csv(self.wavefile,delimiter=delimiter,names=wavecol)
							cols = list(dpw.columns)
						if wavecol in cols: self.wave = np.array(dpw[wavecol])*waveunit
						else: self.wave = np.array(dpw[cols[0]])*waveunit
		if len(self.wave)==0:
			raise ValueError('Warning: wave array was not included; include keyword `wavefile` or pass wavelength array')

# CHECK EVERYTHING LINES UP
		if isinstance(self.parameters,pandas.core.frame.DataFrame)==False:
			raise ValueError('Parameter array must be a pandas dataframe')
		if len(self.wave) != len(self.flux[0,:]):
			if ERROR_CHECKING==True: print('Warning! wavelength array has {:.0f} values but fluxes have {:.0f} values; you will have errors'.format(len(self.wave),len(self.flux[1])))
		if len(self.flux[:,0]) != len(self.parameters):
			if ERROR_CHECKING==True: print('Warning! flux array has {:.0f} spectra but there are {:.0f} parameter sets; you will have errors'.format(len(self.flux[1]),len(self.parameters)))
		if not isUnit(self.wave): self.wave = self.wave*waveunit
		if not isUnit(self.flux): self.flux = self.flux*fluxunit

# FILL IN RELEVANT INFORMATION FROM MODEL ARRAY
		tmp = checkName(self.instrument,DEFINED_INSTRUMENTS)
		if isinstance(tmp,bool)==False:
			self.instrument=tmp
			self.instrument_parameters = DEFINED_INSTRUMENTS[tmp]
		tmp = checkName(self.modelname,DEFINED_SPECTRAL_MODELS)
		if isinstance(tmp,bool)==False:
			self.modelname=tmp
			self.model_parameters = DEFINED_SPECTRAL_MODELS[tmp]
		return


	def info(self):
		'''
		Basic information
		'''
		print('\n{} models for {} instrument'.format(self.modelname,self.instrument))
		if len(self.model_parameters)>0:
			print('\nModel Information:')
			cols = list(self.model_parameters.keys())
			cols.sort()
			for x in cols: print('\t{}: {}'.format(x,str(self.model_parameters[x])))
		if len(self.instrument_parameters)>0:
			print('\nInstrument Information:')
			cols = list(self.instrument_parameters.keys())
			cols.sort()
			for x in cols: print('\t{}: {}'.format(x,str(self.instrument_parameters[x])))
		print('\nModel Parameters:')
		kys = list(self.parameters.columns)
		kys.sort()
		for x in ['file']:
			if x in kys: kys.remove(x)
		for k in kys:
			vals = list(set(list(self.parameters[k])))
			if isNumber(self.parameters.loc[0,k])==True:
				if len(vals)==1: print('\t{}: {}'.format(k,vals[0]))
				else: print('\t{}: {} to {}'.format(k,np.nanmin(vals),np.nanmax(vals)))
			else:
				f = vals[0]
				if len(vals) > 0:
					for i in vals[1:]: f=f+', {}'.format(i)
				print('\t{}: {}'.format(k,f))
		print('\nWavelength range: {:.2f} to {:.2f} {}'.format(np.nanmin(self.wave.value),np.nanmax(self.wave.value),str(self.wave.unit)))
		print('Fluxes in units of {}'.format(str(self.flux[0,:].unit)))
		return


	def toWavelengths(self,wave):
		'''
		resample all fluxes to an input wavelength grid
		'''
		pass

	def fluxConvert(self,unit):
		'''
		transfrom all fluxes to a new unit, including flam <-> fnu
		'''
		pass

	def model(self,parameters):
		'''
		retrieves an individual model from model grid based on parameters
		calls _gridModel and _interpolatedModel
		'''
		pass

	def _gridModel(self,parameters):
		'''
		retrieves one of the models in the model grid based on parameters
		'''
		pass

	def _interpolatedModel(self,parameters):
		'''
		computes an interpolated model among the model grid
		'''
		pass

#######################################################
#######################################################
#####  BASIC SPECTRAL MANIPULATION AND ANALYSIS  ######
#######################################################
#######################################################

# COMPARES TWO SPECTRA
def compareSpec(f1,f2,unc,weights=[],stat='chi-square',verbose=ERROR_CHECKING):
	'''
	
	Purpose
	-------

	Compares two flux vectors and corresponding uncertainty vector and returns a qualitative measure of agreement.
	Note that is assumed the  function, computes chi square with optimal scale factor

	Parameters
	----------

	f1 : np.array
		An array of floats corresponding to the first spectrum; this quantity should not have units

	f2 : np.array
		An array of floats corresponding to the second spectrum; this quantity should not have units

	unc : np.array
		An array of floats corresponding to the joint uncertainty; this quantity should not have units

	weights = [] : np.array
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
	if len(weights)!=len(f1): wt = np.ones(len(f1))
	else: wt=np.array(weights)
# mask out bad pixels in either spectrum or uncertainty
	w = np.where(np.logical_and(np.isnan(f1+f2+unc)==False,wt*unc!=0))
	dof = len(f1[w])
	if dof<=1: raise ValueError('Not enough flux or noise values are non-nan')
# compute chi-square - CURRENTLY ONLY OPTION
	scl = np.nansum(wt[w]*f1[w]*f2[w]/(unc[w]**2))/np.nansum(wt[w]*(f2[w]**2)/(unc[w]**2))
	chi = np.nansum(wt[w]*((f1[w]-scl*f2[w])**2)/(unc[w]**2))
	return chi, scl, dof-1



# RESAMPLE SPECTRUM ONTO A NEW WAVELENGTH SCALE
# NOTE: need to rework this using Johnson method
def resample(sp,wave,method='weighted integrate',wave_unit=DEFAULT_WAVE_UNIT,flux_unit=DEFAULT_FLUX_UNIT,default_noise=np.nan,smooth=1,verbose=ERROR_CHECKING):
	'''
	
	Purpose
	-------

	Resamples a spectrum onto a wavelength grid with optional smoothing

	Parameters
	----------

	sp : splat.Spectrum class
		splat Spectrum object to resample onto wave grid

	wave : np.ndarray or list
		wave grid to resample spectrum onto; if unitted, this is converted to units specified in `wave_unit`, 
		otherwise assumed to be in the units of `wave_unit`

	method = 'integrate' : str
		Method by which spectrum is resampled onto new wavelength grid; options are:
		* 'integrate': flux in integrated across wach wavelength grid point (also 'int')
		* 'weighted integrate' (default): weighted integration, where weights are equal to 1/uncertainty**2 (also 'wint')
		* 'mean': mean value in each wavelength grid point is used (also 'average', 'mn', 'ave')
		* 'weighted mean': weighted mean value with weights are equal to 1/uncertainty**2 (also 'wmn', 'weighted')
		* 'median': median value in each wavelength grid point is used (also 'wmn', 'weighted')

	default_noise = np.nan : int or float
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
#	 if isUnit(flux): flx0=flux.to(flux_unit).value
#	 else: flx0 = np.array(copy.deepcopy(flux))

# # prepare input uncertainty
#	 if isUnit(noise): unc0=noise.to(flux_unit).value
#	 else: unc0 = np.array(copy.deepcopy(noise))
#	 if len(noise)==0: 
#		 if isUnit(default_noise): dns=default_noise.to(flux_unit).value
#		 else: dns = np.array(copy.deepcopy(default_noise))
#		 unc0 = np.array([dns]*len(flx0))

# # prepare input wavelength grid
#	 if isUnit(wave0): wv0=wave0.to(wave_unit).value
#	 else: wv0 = np.array(copy.deepcopy(wave0))

# prepare output wavelength grid
	if isUnit(wave): wv=wave.to(sp.wave.unit).value
	else: wv = np.array(copy.deepcopy(wave))
	wshift = 2.*np.absolute(np.nanmedian(np.roll(wv,-1)-wv))


# trim if necessary
#	print(np.nanmin(sp.wave.value),np.nanmax(sp.wave.value),np.nanmin(wv),np.nanmax(wv))
	sp.trim([wv[0]-3.*wshift,wv[-1]+3.*wshift])
#	print(np.nanmin(sp.wave.value),np.nanmax(sp.wave.value),len(sp.wave))
	# wv0 = wv0[wtr]
	# flx0 = flx0[wtr]
	# unc0 = unc0[wtr]
	# wtr = np.where(np.logical_and(wv0>=wv[0]-3.*wshift,wv0<=wv[0]+3.*wshift))
	# if len(wv0[wtr])==0:
	#	 raise ValueError('Input wavelength grid {:.2f}-{:2f} does not overlap with new input wavelength grid {:.2f}-{:.2f}'.format(np.nanmin(wv0),np.nanmax(wv0),np.nanmin(wv),np.nanmax(wv)))

# prepare spectrum object
	# spc = copy.deepcopy(sp)
	# spc.trim([wv[0]-3.*wshift,wv[-1]+3.*wshift])

# run interpolation
	flx = [np.nan]*len(wv)
	unc = [np.nan]*len(wv)
	smind = int(smooth)
	for i,w in enumerate(wv):
		if i<smind: wrng = [w-(wv[smind]-w),wv[smind]]
		elif i>=len(wave)-smind: wrng = [wv[i-smind],w+(w-wv[i-smind])]
		else: wrng = [wv[i-smind],wv[i+smind]]
		wsel = np.where(np.logical_and(sp.wave.value>=wrng[0],sp.wave.value<=wrng[1]))
		cnt = len(sp.wave.value[wsel])
# expand range
		if cnt <= 1:
			wsel = np.where(np.logical_and(sp.wave.value>=wrng[0]-wshift,sp.wave.value<=wrng[1]+wshift))
			cnt = len(sp.wave.value[wsel])
		if cnt >= 1:
			flx0s = sp.flux.value[wsel]
			unc0s = sp.noise.value[wsel]
			wv0s = sp.wave.value[wsel]
			wn = np.where(~np.isnan(flx0s))
			if len(flx0s[wn])>0:
				if method.lower() in ['mean','mn','average','ave']:
					flx[i] = np.nanmean(flx0s[wn])
					if np.isfinite(np.nanmax(unc0s))==True: unc[i] = np.nanmean(unc0s[wn])/((len(unc0s[wn])-1)**0.5)
				elif method.lower() in ['weighted mean','wmn','weighted']:
					wts = 1./unc0s[wn]**2
					if np.isnan(np.nanmin(wts))==True: wts = np.ones(len(wv0s[wn]))
					flx[i] = np.nansum(wts*flx0s[wn])/np.nansum(wts)
					if np.isfinite(np.nanmax(unc0s))==True: unc[i] = (np.nansum(wts*unc0s[wn]**2)/np.nansum(wts))**0.5
				elif method.lower() in ['integrate','int']:
					wts = np.ones(len(wv0s[wn]))
					if cnt > 1: 
						flx[i] = np.trapz(wts*flx0s[wn],wv0s[wn])/np.trapz(wts,wv0s[wn])
						if np.isfinite(np.nanmax(unc0s))==True: unc[i] = (np.trapz(wts*unc0s[wn]**2,wv0s[wn])/np.trapz(wts,wv0s[wn]))**0.5
					else:
						flx[i] = np.nansum(wts*flx0s[wn])/np.nansum(wts)
						if np.isfinite(np.nanmax(unc0s))==True: unc[i] = (np.nansum(wts*unc0s[wn]**2)/np.nansum(wts))**0.5
				elif method.lower() in ['weighted integrate','wint']:
					wts = 1./unc0s[wn]**2
					if np.isnan(np.nanmin(wts))==True: wts = np.ones(len(wv0s[wn]))
					if cnt > 1: 
						flx[i] = np.trapz(wts*flx0s[wn],wv0s[wn])/np.trapz(wts,wv0s[wn])
						if np.isfinite(np.nanmax(unc0s))==True: unc[i] = (np.trapz(wts*unc0s[wn]**2,wv0s[wn])/np.trapz(wts,wv0s[wn]))**0.5
					else:
						flx[i] = np.nansum(wts*flx0s[wn])/np.nansum(wts)
						if np.isfinite(np.nanmax(unc0s))==True: unc[i] = (np.nansum(wts*unc0s[wn]**2)/np.nansum(wts))**0.5
					# unc[i] = (np.trapz(np.ones(len(wv0[wn])),wv0[wn])/np.trapz(1/unc0[wn]**2,wv0[wn]))**0.5
					# flx[i] = np.trapz(flx0[wn],wv0[wn])/np.trapz(np.ones(len(wv0[wn])),wv0[wn])
# median by default
				else:
					flx[i] = np.nanmedian(flx0s[wn])
					if np.isfinite(np.nanmax(unc0s))==True: unc[i] = flx[i]/np.nanmedian(flx0s[wn]/unc0s[wn])
		# else:
		#	 print('no wavepoints in {:.2f}-{:.2f}'.format(wrng[0],wrng[1]))
#					unc[i] = np.nanmedian(unc0[wn])/((len(unc0[wn])-1)**0.5)

# return flux
	# return flx*flux_unit

# return Spectrum object
	return Spectrum(wave=np.array(wv)*sp.wave.unit,flux=flx*sp.flux.unit,noise=unc*sp.flux.unit,name=sp.name)



# GET ONT OF THE SAMPLE SPECTRA PROVIDED
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
	sp = Spectrum(sfile,name=DEFINED_INSTRUMENTS[inst]['sample_name'],instrument=inst)
	if verbose==True: print('Reading in sample spectrum for instrument {} of source {}'.format(inst,sp.name))
	return sp



#######################################################
#######################################################
################   MODEL FUNCTIONS  ###################
#######################################################
#######################################################

# INFORMATION ON A MODEL
# UPDATE THIS WITH NEW MODEL STRUCTURE
# THIS WILL BE OBVIATED BY MODELSET CLASS
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
				else: print('\t\t{}: {} to {}'.format(k,np.nanmin(vals),np.nanmax(vals)))
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


# GENERATE A NEW WAVELENGTH ARRAY
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
	return np.array(wave)*wunit


# READ A PREDEFINED WAVELENGTH FILE
# THIS WILL BE OBVIATED BY MODELSET CLASS
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
	return np.array(dp[cname])*DEFAULT_WAVE_UNIT


# WRITE A WAVELENGTH FILE
# THIS WILL BE OBVIATED BY MODELSET CLASS
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


# READ IN A MODEL SET
# THIS WILL BE OBVIATED BY MODELSET CLASS
def readModelSet(file,verbose=ERROR_CHECKING):
	'''
	Reads in an h5 model set
	'''	
	if os.path.exists(file)==False:
		raise ValueError('WARNING: model set file {} cannot be found, check your file name'.format(file))
	return pandas.read_hdf(file)


# "GETS" A MODEL SET, INCLUDING THE WAVELENGTH FILE
# THIS WILL BE OBVIATED BY MODELSET CLASS
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


# GENERATE A MODEL SET
# note that this routine requires the splat.model package
def generateModelSet(modelset,wave=[],modelpars={},constraints={},initial_instrument='RAW',
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

	wave = DEFAULT_WAVE : str or list or np.ndarray
		Either the name of the instrument that will serve as the baseline wavelength array, or the array 
		of wavelengths to sample the spectra to, which can be of type list or np.ndarray and can be 
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

# check splat.model is available
	try: 
		import splat
		import splat.model as spmdl
	except: raise ValueError('The routine generateModelSet() requires the SPLAT package to be installed\nhttps://github.com/aburgasser/splat')

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
		#	 print('WARNING: No {} models for set {} are available in SPLAT, cannot run this'.format(initial_instrument,modelset))
		#	 return

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
		elif isinstance(wave,list) or isinstance(wave,np.ndarray):
			if len(wave)==0:
				wave = generateWave(DEFAULT_WAVE_RANGE,DEFAULT_RESOLUTION,method='resolution',verbose=ERROR_CHECKING)
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
	step = np.ceil(len(modelpars)/10.)
#	for i in tqdm(range(len(dp))):
	for i in range(len(modelpars)):
		if i!=0 and np.mod(i,step)==0 and verbose==True: print('\t{:.0f}% complete'.format(i/step*10),end='\r')
		par = dict(modelpars.loc[i,:])

# read in with splat.Spectrum
		mdl = splat.Spectrum(modelpars.loc[i,'file'])
		wv,flx = mdl.wave.value,mdl.flux.value
# read in with spmdl.loadModel
		if np.isfinite(np.nanmedian(flx))==False:
			par = dict(modelpars.loc[i,:])
			mdl = spmdl.loadModel(**par,force=True)
			wv,flx = mdl.wave.value,mdl.flux.value
# read in with splat.readSpectrum
		if np.isfinite(np.nanmedian(flx))==False:
			mdl = splat.readSpectrum(modelpars.loc[i,'file'])
			wv,flx = mdl['wave'].value,mdl['flux'].value
# read in with pandas
		if np.isfinite(np.nanmedian(flx))==False:
			if '.txt' in modelpars.loc[i,'file']: delim='\t'
			elif '.csv' in modelpars.loc[i,'file']: delim=','
			else: delim=r'\s+'
			dp = pandas.read_csv(modelpars.loc[i,'file'],delimiter=delim,names=['wave','flux'],comment='#')
			wv,flx = dp['wave'],dp['flux']
# don't know what to do
#		print(modelpars.loc[i,'file'],len(flx))
		try: md = np.isfinite(np.nanmedian(flx))
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


# GET ONE OF THE GRID MODELS
# THIS WILL BE OBVIATED BY MODELSET CLASS
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

	wave = [] : list or np.ndarray
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
	mdl = Spectrum(wave=wave,flux=flx*DEFAULT_FLUX_UNIT,name=name)
	if 'scale' in list(par.keys()) and scale==True: mdl.scale(par['scale'])
	mdl.parameters = dict(smdls.loc[0,:])
	for x in [flux_name,'file']:
		if x in list(mdl.parameters.keys()): del mdl.parameters[x]
	return mdl


# GET AN INTERPOLATED GRID MODEL
# THIS WILL BE OBVIATED BY MODELSET CLASS
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

	wave = [] : list or np.ndarray
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
#				print(par[k],np.nanmin(vals),np.nanmax(vals)) 
			if par0[k] in vals:
				smdls = smdls[smdls[k]==par0[k]]
				smdls.reset_index(inplace=True,drop=True)
			else:
				valstep = np.absolute(np.array(vals)-np.roll(vals,1))
				step = np.nanmedian(valstep[1:])				
				limits[k] = [np.nanmax([np.nanmin(smdls[k]),par0[k]-step]),np.nanmin([np.nanmax(smdls[k]),par0[k]+step])]
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
			fitvals+=tuple([[np.log10(x) for x in smdls[k]]])
			parvals.append(np.log10(par0[k]))
		else:
			fitvals+=tuple([list(smdls[k])])
			parvals.append(par0[k])
	parvals = np.array([parvals])
	fitvals = np.transpose(np.vstack(fitvals))

# run interpolation
	flx = []
	for i in range(len(smdls.loc[0,flux_name])):
		fs = [np.log10(x[i]) for x in smdls[flux_name]]
		try: flx.append(griddata(fitvals,tuple(fs),parvals,method='linear',rescale=True)[0])
		except: 
			if verbose==True: print('getInterpModel failed for values '.format)
			raise ValueError('Insufficient model coverage; try reducing parameter constraints')
	flx = np.array(flx)
	flx = 10.**flx
	if np.isnan(np.nanmedian(flx))==True: raise ValueError('Could not interpolate {} over grid, possibly due to grid gaps'.format(par0))
#	print(truepar)

# turn into Spectrum and scale if desired
	name = '{} model: '.format(models.loc[0,'model'])
# NEED TO ADD - FORMAT STRING FROM PARAMETERS
	for x in list(par0.keys()): 
		name=name+'{}={} '.format(x,par0[x])
	mdl = Spectrum(wave=wave,flux=flx*DEFAULT_FLUX_UNIT,name=name)
	if 'scale' in list(par.keys()) and scale==True: mdl.scale(par['scale'])
	mdl.parameters = par0
	return mdl



# WRAPPER TO GET A GRID OR INTERPOLATED MODEL
# THIS WILL BE OBVIATED BY MODELSET CLASS
def getModel(mdls,par,wave,scale=True,verbose=ERROR_CHECKING):
	try: sp = getGridModel(mdls,par,wave,scale=scale,verbose=verbose)
	except: sp = getInterpModel(mdls,par,wave,scale=scale,verbose=verbose)
#	if 'scale' in par and rescale==True: sp.scale(par['scale'])
	return sp


#####################################################
#####################################################
#####################################################
################## FITTING METHODS ################## 
#####################################################
#####################################################
#####################################################


# FIT SPECTRUM TO A GRID OF MODELS
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
	for x in ['scale','chis','radius','dof']: mdls[x] = [np.nan]*len(mdls)
	for jjj in range(len(mdls)):
		chi,scl,dof = compareSpec(spscl.flux.value,np.array(mdls.loc[jjj,flux_name]),spscl.noise.value,verbose=verbose)
		mdls.loc[jjj,'chis'] = chi
		mdls.loc[jjj,'scale'] = scl
		mdls.loc[jjj,'dof'] = dof
# radius scaling assuming spectrum is in absolute flux units
		mdls.loc[jjj,'radius'] = (10.*u.pc*(scl**0.5)).to(u.Rsun).value
#	mdls['model'] = [mset]*len(mdls)

# best fit
	mpar = dict(mdls.loc[np.argmin(mdls['chis']),:])
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
#	comp = splat.Spectrum(wave=wave,flux=np.array(mdls.loc[ibest,flux_name])*mdls.loc[ibest,'scale']*spscl.flux.unit)
	diff = spscl.flux.value-comp.flux.value
#	dof = np.count_nonzero(~np.isnan(spscl.flux.value))-1
	if verbose==True: print('\treduced chi2 = {}'.format(mpar['rchi']))
	# sclstd = np.nanstd(diff.flux.value,ddof=1)/np.nanmax(spscl.flux.value)
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


# MCMC FIT OF A SPECTRUM TO AN INTERPOLATED GRID OF MODELS
def fitMCMC(spc,models,p0={},constraints={},flux_name=DEFAULT_FLUX_NAME,output='all',
	pstep=DEFAULT_MCMC_STEPS,nstep=100,iterim=50,method='chidiff',threshhold=0.5,burn=0.25,
	quantscale=[0.25,0.5,0.75],nsample=0,absolute=False,report=True,xscale='linear',yscale='linear',
	file_prefix='mcmcfit_',verbose=ERROR_CHECKING):
#	radius=np.nan,e_radius=np.nan,report=True):
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
				else: pstep[k] = 0.5*np.nanmedian(np.absolute(np.array(vals)-np.roll(vals,1)))
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
			default = DEFINED_SPECTRAL_MODELS[mset]['default'][k]
#			default = splat.SPECTRAL_MODELS[mset]['default'][k]
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
			pnew[k] = np.random.normal(pvals[-1][k],pstep[k])
			pnew[k] = np.nanmin([pnew[k],np.nanmax(mdls[k])])
			pnew[k] = np.nanmax([pnew[k],np.nanmin(mdls[k])])
# discrete variables
		for k in list(pfitd.keys()): 
			vals = list(set(list(mdls[k])))
			pnew[k] = np.random.choice(vals)
		pnew = pnew | pfitd
		try:
			cmdl = getModel(mdls,pnew,spscl.wave,scale=False,verbose=verbose)
			if verbose==True: print(i,pnew)
			chinew,scl,_ = compareSpec(spscl.flux.value,cmdl.flux.value,spscl.noise.value,verbose=verbose)
			# if np.isnan(radius)==False and np.isnan(e_radius)==False:
			#	 chinew+=(((10.*u.pc*(scl**0.5)).to(u.Rsun).value-radius)/e_radius)**2
			#if 'scale' not in list(pnew.keys()): cmdl.scale(scl)

	# compute statistic
			if method=='chidiff': st,chst = (chinew-chis[-1])/np.nanmin(chis),np.random.uniform(0,threshhold)
			elif method=='survival': st,chst = 2*stats.f.sf(chinew/chis[-1],dof,dof),np.random.uniform(threshhold,1)
			elif method=='dofscale': st,chst = dof/(0.5*dof+chinew-chis[-1]),np.random.uniform(threshhold,1)
			else: raise ValueError('Do not recognize statistical comparison {}; try chidiff, survival, or dofscale'.format(method))
	#			if verbose==True: print(chinew,chis[-1],dof,st,chst)

			if st<chst:
	# reset if we've wandered off
				if chinew>(1+2*threshhold)*np.nanmin(chis):
					if verbose==True: print('RESETING TO BEST FIT')
					pvals.append(pvals[np.argmin(chis)])
					chis.append(chis[np.argmin(chis)])
					scales.append(scales[np.argmin(chis)])
					mdlflxs.append(mdlflxs[np.argmin(chis)])
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
		if iterim>0 and i>0 and np.mod(i,iterim)==0 and report==True:
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
			pbest = dict(dpfit.loc[np.argmin(dpfit['chis']),:])
#			pbest['radius'] = (10.*u.pc*(scales[np.argmin(chis)]**0.5)).to(u.Rsun).value
			cmdl = getModel(mdls,pbest,spscl.wave,scale=True,verbose=verbose)
			# print(scales[np.argmin(chis)],pbest['scale'],np.nanmax(cmdl.flux.value))
			# cmdl.scale(scales[np.argmin(chis)])
			# print(np.nanmax(cmdl.flux.value))
			label = '{} model '.format(mset)
			label+=r'$\chi^2_r$='+'{:.1f}\n'.format(np.nanmin(chis)/dof)
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
			pltbest = [dpfit.loc[np.argmin(dpfit['chis']),x] for x in plotpars]
# NOTE: THIS IS ONE OPTION FOR WEIGHTING, COULD TRY OTHERS			
			weights = np.array(dof/(dof+dpfit['chis']-np.nanmin(dpfit['chis'])))
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
	pbest = dict(dpfit.loc[np.argmin(dpfit['chis']),:])
	pvalsb[np.argmin(chis[int(burn*nstep):])]
	for x in [flux_name]:
		if x in list(pbest.keys()): del pbest[x]
	if verbose==True: print('Best parameters: {}'.format(pbest))
	cmdl = getModel(mdls,pbest,spscl.wave,verbose=verbose)
	if 'scale' not in list(pbest.keys()): cmdl.scale(scales[np.argmin(chis)])

# distribution of values
	pdist = {}
	dpars = copy.deepcopy(mkysfit)
	for k in dpars:
		if isinstance(pvalsb[0][k],str)==False: pdist[k] = np.nanquantile([p[k] for p in pvalsb],quantscale)
	if absolute==True: 
		pdist['radius'] = np.nanquantile([(10.*u.pc*(x**0.5)).to(u.Rsun).value for x in scales[int(burn*nstep):]],quantscale)

	if report == True:
# remove initial burn and save
		outfile = file_prefix+'_parameters.xlsx'
		if verbose==True: print('Saving database of prameters to {}'.format(outfile))
		dpfit.to_excel(outfile,index=False)
# plot comparison
		# cmdl = getModel(mdls,pbest,spscl.wave,verbose=verbose)
		# if 'scale' not in list(pbest.keys()): cmdl.scale(scales[np.argmin(chis)])
		label = '{} model '.format(mdls.loc[0,'model'])
		label+=r'$\chi^2_r$='+'{:.1f}\n'.format(chis[np.argmin(chis)]/dof)
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
		pltbest = [dpfit.loc[np.argmin(dpfit['chis']),x] for x in plotpars]
# NOTE: THIS IS ONE OPTION FOR WEIGHTING, COULD TRY OTHERS			
		weights = np.array(dof/(dof+dpfit['chis']-np.nanmin(dpfit['chis'])))
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


# emcee FIT OF A SPECTRUM TO AN INTERPOLATED GRID OF MODELS
# PLACEHOLDER
def fitemcee(spc,models,p0={},constraints={},output='all',
	pstep=DEFAULT_MCMC_STEPS,nstep=100,iterim=50,method='chidiff',threshhold=0.5,burn=0.25,
	quantscale=[0.25,0.5,0.75],nsample=0,absolute=False,report=True,xscale='linear',yscale='linear',
	file_prefix='emceefit_',verbose=ERROR_CHECKING):

	pass
	return

####################################################
####################################################
####################################################
################ PLOTTING FUNCTIONS ################ 
####################################################
####################################################
####################################################

# PLOT COMPARISON OF TWO SPECTRA
def plotCompare(sspec,cspec,outfile='',clabel='Comparison',absolute=False,xscale='linear',yscale='linear',
	figsize=[8,5],height_ratio=[5,1],scale=1.,fontscale=1,xlabel='Wavelength',ylabel='Flux',ylabel2='O-C',
	ylim=None,xlim=None,legend_loc=1,verbose=ERROR_CHECKING):

	sspec.scale(scale)
	cspec.scale(scale)
	diff = sspec.flux.value-cspec.flux.value

	# xlabel = r'Wavelength'+' ({:latex})'.format(sspec.wave.unit)
	# ylabel = r'F$_\lambda$'+' ({:latex})'.format(sspec.flux.unit)
	# if absolute==True: ylabel='Absolute '+ylabel
	strue = sspec.wave.value[np.isnan(sspec.flux.value)==False]
	wrng = [np.nanmin(strue),np.nanmax(strue)]

	plt.clf()
	fg, (ax1,ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': height_ratio}, sharex=True, figsize=figsize)
	ax1.step(sspec.wave.value,sspec.flux.value,'k-',linewidth=2,label=sspec.name)
	ax1.step(cspec.wave.value,cspec.flux.value,'m-',linewidth=4,alpha=0.5,label=clabel)
	ax1.legend(fontsize=12*fontscale,loc=legend_loc)
	ax1.plot([np.nanmin(sspec.wave.value),np.nanmax(sspec.wave.value)],[0,0],'k--')
	ax1.fill_between(sspec.wave.value,sspec.noise.value,-1.*sspec.noise.value,color='k',alpha=0.3)
	scl = np.nanmax(cspec.flux.value)
	scl = np.nanmax([scl,np.nanmax(sspec.flux.value)])
	if ylim==None: ax1.set_ylim([x*scl for x in [-0.1,1.3]])
	else: ax1.set_ylim(ylim)
	if yscale=='log':
#		ax1.set_ylim([x*scl for x in [1.e-2,2]])
		if ylim==None: ax1.set_ylim([np.nanmean(sspec.noise.value)/2.,2*scl])
	if xlim==None: xlim=wrng
	ax1.set_xlim(xlim)
	ax1.set_xscale(xscale)
	ax1.set_yscale(yscale)
	ax1.set_ylabel(ylabel,fontsize=12*fontscale)
	ax1.tick_params(axis="x", labelsize=0)
	ax1.tick_params(axis="y", labelsize=14*fontscale)

	ax2.step(sspec.wave.value,diff,'k-',linewidth=2)
	ax2.plot([np.nanmin(sspec.wave.value),np.nanmax(sspec.wave.value)],[0,0],'k--')
	ax2.fill_between(sspec.wave.value,sspec.noise.value,-1.*sspec.noise.value,color='k',alpha=0.3)
	scl = np.nanquantile(diff,[0.02,0.98])
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

# PLOT COMPARISON OF SPECTRUM AND BEST MCMC FIT, ALONG WITH SAMPLING OF CHAIN
def plotCompareSample(spec,models,chain,nsample=50,relchi=1.2,method='samples',absolute=False,outfile='',
	clabel='Comparison',xlabel='Wavelength',ylabel='Flux',ylabel2='O-C',scale=1.,xscale='linear',yscale='linear',
	figsize=[8,5],height_ratio=[5,1],fontscale=1,ylim=None,xlim=None,legend_loc=1,verbose=ERROR_CHECKING):
# set up
	# xlabel = r'Wavelength'+' ({:latex})'.format(sspec.wave.unit)
	# ylabel = r'F$_\lambda$'+' ({:latex})'.format(sspec.flux.unit)
	# if absolute==True: ylabel='Absolute '+ylabel
	strue = spec.wave.value[np.isnan(spec.flux.value)==False]
	wrng = [np.nanmin(strue),np.nanmax(strue)]
	if nsample<0: nsample = int(len(chain)/10)

# first identify the best fit model
	pbest = dict(chain.loc[np.argmin(chain['chis']),:])
	cspec = getModel(models,pbest,spec.wave)
#	if 'scale' not in list(chain.columns): cspec.scale(scale)
#	cspec.scale(pbest['scale'])
# scale
	sspec = copy.deepcopy(spec)
	sspec.scale(scale)
#	print(np.nanmedian(sspec.flux.value))
	cspec.scale(scale)
	diff = sspec.flux.value-cspec.flux.value

# now identify the random sample
	chainsub = chain[chain['chis']/np.nanmin(chain['chis'])<relchi]
	chainsub.reset_index(inplace=True)
	nsamp = np.nanmin([nsample,len(chainsub)])
	fluxes = [getModel(models,dict(chainsub.loc[i,:]),sspec.wave).flux for i in np.random.randint(0,len(chainsub)-1,nsamp)]

# plot
	plt.clf()
	fg, (ax1,ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': height_ratio}, sharex=True, figsize=figsize)
	if method=='minmax':
		minflx = np.nanmin(fluxes,axis=0)*scale
		maxflx = np.nanmax(fluxes,axis=0)*scale
		# if 'scale' not in list(chainsub.columns):
		#	 minflx = minflx*scale 
		#	 maxflx = maxflx*scale 
		ax1.fill_between(sspec.wave.value,minflx,maxflx,color='m',alpha=0.2)
	elif method=='meanstd':
		meanflx = np.nanmean(fluxes,axis=0)*scale
		stdflx = np.nanstd(fluxes,axis=0)*scale
		# if 'scale' not in list(chainsub.columns):
		#	 meanflx = meanflx*scale 
			# stdflx = stdflx*scale 
		ax1.fill_between(sspec.wave.value,meanflx-stdflx,meanflx+stdflx,color='m',alpha=0.2)
	else:
		for f in fluxes: ax1.step(cspec.wave.value,f,'m-',linewidth=2,alpha=1/nsamp)
	ax1.step(sspec.wave.value,sspec.flux.value,'k-',linewidth=2,label=sspec.name)
	ax1.step(cspec.wave.value,cspec.flux.value,'m-',linewidth=2,alpha=0.7,label=clabel)
	ax1.legend(fontsize=12*fontscale,loc=legend_loc)
	ax1.plot([np.nanmin(sspec.wave.value),np.nanmax(sspec.wave.value)],[0,0],'k--')
	ax1.fill_between(sspec.wave.value,sspec.noise.value,-1.*sspec.noise.value,color='k',alpha=0.3)
	scl = np.nanmax(cspec.flux.value)
	scl = np.nanmax([scl,np.nanmax(sspec.flux.value)])
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
	ax2.plot([np.nanmin(sspec.wave.value),np.nanmax(sspec.wave.value)],[0,0],'k--')
	ax2.fill_between(sspec.wave.value,sspec.noise.value,-1.*sspec.noise.value,color='k',alpha=0.3)
	scl = np.nanquantile(diff,[0.02,0.98])
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

# PLOT CHAINS OF MCMC FIT
def plotChains(dpfit,plotpars,pbest={},outfile='',xlabel='Step',labeldict=PARAMETER_PLOT_LABELS,verbose=ERROR_CHECKING):
	nplot = int(len(plotpars))
	if nplot==0: 
		if verbose==True: print('WARNING: no parameters to plot')
		return
# set up plot
	plt.clf()
	fig = plt.figure(figsize=[2*6,np.ceil(nplot/2)*3])
	for i,l in enumerate(plotpars):	
		ax = plt.subplot(int(np.ceil(nplot/2)),2,i+1)
		ax.plot(dpfit[l],'k-')
# indicate current best fit parameter		
		if l in list(pbest.keys()): 
			ax.plot(np.zeros(len(dpfit[l]))+pbest[l],'b--')
# indicate best fit in chain		
		if 'chis' in list(dpfit.keys()):
#			print(l,dpfit.loc[np.argmin(dpfit['chis']),l])
			ax.plot(np.zeros(len(dpfit[l]))+dpfit.loc[np.argmin(dpfit['chis']),l],'m--')
			ax.plot([np.argmin(dpfit['chis']),np.argmin(dpfit['chis'])],[np.nanmin(dpfit[l]),np.nanmax(dpfit[l])],'m--')
			ax.set_title(dpfit.loc[np.argmin(dpfit['chis']),l])
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


# PLOT PARAMETER DISTRIBUTIONS (CORNER PLOT) OF MCMC FIT
def plotCorner(dpfit,plotpars,pbest={},weights=[],outfile='',verbose=ERROR_CHECKING):
# choose plot columns
	ppars = copy.deepcopy(plotpars)
	for x in plotpars:
		if np.nanmin(dpfit[x])==np.nanmax(dpfit[x]): ppars.remove(x)
	if len(ppars)==0:
		if verbose==True: print('Warning: there are no parameters to plot!')
		return
# reorder
	ppars2 = []
	for k in list(PARAMETER_PLOT_LABELS.keys()):
		if k in ppars: ppars2.append(k)
	dpplot = dpfit[ppars2]
		
# weights
	if len(weights)<len(dpplot): weights=np.ones(len(dpplot))
	
# labels
	plabels=[]
	for k in ppars2:
		if k in list(PARAMETER_PLOT_LABELS.keys()): plabels.append(PARAMETER_PLOT_LABELS[k])
		else: plabels.append(k)

# best fit parameters
	truths = [np.nan for x in ppars2]
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
