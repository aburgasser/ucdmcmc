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
#import scipy.stats as stats
from tqdm import tqdm
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation, CartesianRepresentation, CartesianDifferential, Galactic, Galactocentric
import splat
import splat.model as spmdl
import astropy.units as u
from statsmodels.stats.weightstats import DescrStatsW


#######################################################
#######################################################
#################   INITIALIZATION  ###################
#######################################################
#######################################################


# code parameters
VERSION = '5 July 2024'
GITHUB_URL = 'http://www.github.com/aburgasser/ucdmcmc/'
ERROR_CHECKING = True
CODE_PATH = os.path.dirname(os.path.abspath(__file__))+'/../'
MODEL_FOLDER = os.path.join(CODE_PATH,'models/')
MODEL_FILE_PREFIX = 'models_'
WAVE_FILE_PREFIX = 'wave_'

# defaults
DEFAULT_FLUX_UNIT = u.erg/u.s/u.cm/u.cm/u.micron
DEFAULT_WAVE_UNIT = u.micron

# baseline wavelength grid
DEFAULT_WAVE_RANGE = [0.9,2.45]
DEFAULT_RESOULTION = 300

# other parameters
PARAMETER_PLOT_LABELS = {
	'teff':r'T$_{eff}$ (K)',
	'logg':r'$\log{g}$ (cm/s$^2$)',
	'z':'[M/H]',
	'enrich':r'[$\alpha$/Fe]',
	'co':'C/O',
	'kzz':r'$\log\kappa_{zz}$ (cm$^2$/s)',
	'cld':'Cloud Model',
	'ad':r'$\gamma$',
	'radius':r'R (R$_\odot$)',
	'chis':r'$\chi^2$',
}

DEFINED_INSTRUMENTS = {
	'SPEX-PRISM': {'instrument_name': 'IRTF SpeX prism', 'wave_range': [0.7,2.5]*u.micron, 'resolution': 150, 'norders': 1, 'readnoise': 12, 'darkcurrent': 0.2, 'gain': 12, 'altname': ['SPEX','PRISM'], 'bibcode': '2003PASP..115..362R'},
	'JWST-NIRSPEC-PRISM': {'instrument_name': 'JWST NIRSpec (prism mode)', 'pixelscale': 0.43*u.arcsec, 'disperser': 'prism', 'wave_range': [0.6,6.0]*u.Angstrom, 'slitwidth': 2.*u.arcsec, 'resolution': 100, 'norders': 1, 'readnoise': 3.8, 'darkcurrent': 0., 'gain': 1.9, 'altname': ['JWST-NIRSPEC'], 'wave_unit': u.micron, 'flux_unit': u.erg/u.s/u.cm/u.cm/u.micron, 'reader': ''},
	'KECK-NIRES': {'instrument_name': 'Keck NIRES', 'pixelscale': 0.123*u.arcsec, 'wave_range': [0.94,2.45]*u.micron, 'slitwidth': 0.55*u.arcsec, 'resolution': 2700, 'norders': 5, 'orders': [3,4,5,6,7], 'order_wave_range': [[0.94,1.06],[0.95,1.23],[1.13,1.48],[1.42,1.85],[1.88,2.46]],'readnoise': 15., 'darkcurrent': 0.13, 'gain': 3.8, 'altname': ['NIRES'], 'wave_unit': u.micron, 'flux_unit': u.erg/u.s/u.cm/u.cm/u.micron, 'instrument_bibcode': '2000SPIE.4008.1048M', 'reader': ''},
}

DEFINED_SPECTRAL_MODELS = {\
	'atmo20': {'instruments': {}, 'name': 'ATMO2020', 'citation': 'Phillips et al. (2020)', 'bibcode': '', 'altname': ['atmos','phillips','phi20','atmos2020','atmos20','atmo2020','atmo20'], 'default': {'teff': 1500., 'logg': 5.0, 'z': 0.0,'kzz': 'CE','cld': 'LC','broad': 'A','ad': 1.0,'logpmin': -8, 'logpmax': 4}}, \
	'btdusty16': {'instruments': {}, 'name': 'BT Dusty 2016', 'citation': 'TBD', 'bibcode': '', 'altname': ['btdusty2016','dusty16','dusty2016','dusty-bt','bt-dusty','bt-dusty2016','btdusty','bt-dusty16','btd'], 'default': {'teff': 2000., 'logg': 5.0, 'z': 0.0, 'enrich': 0.0}}, \
	'btsettl08': {'instruments': {}, 'name': 'BT Settl 2008', 'citation': 'Allard et al. (2012)', 'bibcode': '2012RSPTA.370.2765A', 'altname': ['allard','allard12','allard2012','btsettl','btsettled','btsettl08','btsettl2008','BTSettl2008','bts','bts08'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'enrich': 0.}}, \
	'burrows06': {'instruments': {}, 'name': 'Burrows et al. (2006)', 'citation': 'Burrows et al. (2006)', 'bibcode': '2006ApJ...640.1063B', 'altname': ['burrows','burrows2006'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'cld': 'nc'}}, \
	'dback24': {'instruments': {}, 'name': 'Sonora Diamondback', 'citation': 'Morley et al. (2024)', 'bibcode': '2024arXiv240200758M', 'altname': ['diamondback','sonora-diamondback','sonora-dback','dback24','diamondback24','morley24','mor24'], 'default': {'teff': 1200., 'logg': 5.0, 'z': 0., 'fsed': 'f2'}}, \
	'elfowl24': {'instruments': {}, 'name': 'Sonora Elfowl', 'citation': 'Mukherjee et al. (2024)', 'bibcode': '2024ApJ...963...73M', 'altname': ['elfowl','sonora-elfowl','elfowl24','mukherjee','mukherjee24','muk24'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'co': 1, 'kzz': 2}}, \
	'lowz': {'instruments': {}, 'name': 'LowZ models', 'citation': 'Meisner et al. (2021)', 'bibcode': '2021ApJ...915..120M', 'altname': ['meisner2021','mei21','line21','line2021'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'kzz': '2.0', 'co': 0.85}}, \
	'saumon12': {'instruments': {}, 'name': 'Saumon et al. 2012', 'citation': 'Saumon et al. (2012)', 'bibcode': '2012ApJ...750...74S', 'altname': ['saumon','sau12','saumon2012'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.}}, \
	'sonora21': {'instruments': {}, 'name': 'Sonora Bobcat', 'citation': 'Marley et al. (2021)', 'bibcode': '2021ApJ...920...85M', 'altname': ['marley2021','sonora','sonora2021','bobcat','sonora-bobcat'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'co': 1}}, \
	'sand24': {'instruments': {}, 'name': 'SAND', 'citation': 'Alvarado et al. (2024)', 'bibcode': '', 'altname': ['sand','san24','sand2024'], 'default': {'teff': 1500., 'logg': 5.0, 'z': 0.1, 'enrich': 0.0}}, \
	# 'btcond': {'instruments': {}, 'name': 'BT Cond', 'citation': 'Allard et al. (2012)', 'bibcode': '2012RSPTA.370.2765A', 'altname': ['dusty-cond','bt-cond','btc'], 'default': {'teff': 1500., 'logg': 5.0, 'z': 0.0, 'enrich': 0.0}}, \
	# 'btnextgen': {'instruments': {}, 'name': 'BT NextGen', 'citation': 'Allard et al. (2012)', 'bibcode': '2012RSPTA.370.2765A', 'altname': ['nextgen-bt','btnextgen','btn'], 'default': {'teff': 3000., 'logg': 5.0, 'z': 0.0, 'enrich': 0.}}, \
	# 'btsettl15': {'instruments': {}, 'name': 'BT Settl 2015', 'citation': 'Allard et al. (2015)', 'bibcode': '2015A&A...577A..42B', 'altname': ['allard15','allard2015','btsettl015','btsettl2015','BTSettl2015','bts15'],  'default': {'teff': 1500., 'logg': 5.0, 'z': 0.}}, \
	# 'cond01': {'instruments': {}, 'name': 'AMES Cond', 'citation': 'Allard et al. (2001)', 'bibcode': '2001ApJ...556..357A', 'altname': ['cond','cond-ames','amescond'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.0}}, \
	# 'drift': {'instruments': {}, 'name': 'Drift', 'citation': 'Witte et al. (2011)', 'bibcode': '2011A&A...529A..44W', 'altname': ['witte','witte11','witte2011','helling'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.}}, \
	# 'dusty01': {'instruments': {}, 'name': 'AMES Dusty', 'citation': 'Allard et al. (2001)', 'bibcode': '2001ApJ...556..357A', 'altname': ['dusty','dusty-ames','amesdusty'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.0}}, \
	# 'lacy23': {'instruments': {}, 'name': 'Lacy & Burrows (2023)', 'citation': 'Lacy & Burrows (2023)', 'bibcode': '2021ApJ...915..120M', 'altname': ['meisner2021','mei21','line21','line2021'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'kzz': '2.0', 'co': 0.85}}, \
	# 'madhusudhan11': {'instruments': {}, 'name': 'Madhusudhan et al. (2011)', 'citation': 'Madhusudhan et al. (2011)', 'bibcode': '2011ApJ...737...34M', 'altname': ['madhu','madhusudhan','madhu11','madhu2011','madhusudhan2011'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.,'cld': 'ae60', 'kzz': 'eq','fsed': 'eq'}}, \
	# 'morley12': {'instruments': {}, 'name': 'Morley et al. (2012)', 'citation': 'Morley et al. (2012)', 'bibcode': '2012ApJ...756..172M', 'altname': ['morley','morley2012'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'fsed': 'f5'}}, \
	# 'morley14': {'instruments': {}, 'name': 'Morley et al. (2014)', 'citation': 'Morley et al. (2014)', 'bibcode': '2014ApJ...787...78M', 'altname': ['morley2014'], 'default': {'teff': 300., 'logg': 5.0, 'z': 0., 'fsed': 'f5', 'cld': 'h50'}}, \
	# 'saumon08': {'instruments': {}, 'name': 'Saumon & Marley 2008', 'citation': 'Saumon & Marley 2008', 'bibcode': '2008ApJ...689.1327S', 'altname': ['sau08','saumon2008'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.}}, \
	# 'sonora18': {'instruments': {}, 'name': 'Sonora Alpha', 'citation': 'Marley et al. (2018)', 'bibcode': 'marley_mark_2018_1309035', 'altname': ['marley','marley18','marley2018','sonora2018'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'cld': 'nc'}}, \
	# 'karalidi21': {'instruments': {}, 'name': 'Sonora Cholla', 'citation': 'Karalidi et al. (2021)', 'bibcode': '2021ApJ...923..269K', 'altname': ['karalidi2021','karalidi','sonora-cholla','cholla'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0., 'kzz': 4}}, \
	# 'gerasimov20': {'instruments': {}, 'name': 'Gerasimov et al. 2020', 'citation': 'Gerasimov et al. (2020)', 'bibcode': '2020RNAAS...4..214G', 'altname': ['phxlowz','ger20'], 'default': {'teff': 1000., 'logg': 5.0, 'z': 0.}}, \
	# 'tremblin16': {'instruments': {}, 'name': 'Tremblin et al. 2016', 'citation': 'Tremblin et al. 2016', 'bibcode': '2016ApJ...817L..19T', 'altname': ['tremblin','tre16','tremblin2016'], 'default': {'teff': 1300., 'logg': 5.0, 'z': 0.1, 'kzz': 6.0, 'ad': 1.05}}, \
	# 'veyette': {'instruments': {}, 'name': 'Veyette et al. 2017', 'citation': 'Veyette et al. 2017', 'bibcode': '2017ApJ...851...26V', 'altname': ['veyette17','veyette2017'], 'default': {'teff': 3000., 'logg': 5.0, 'z': 0.0, 'enrich': 0.0, 'carbon': 0.0, 'oxygen': 0.0}}, \
}

# welcome message on load in
print('\n\nWelcome to the UCDMCMC spectral fitting code!')
#print('This code is designed to conduct both grid and MCMC fitting of spectral data of ultracool dwarfs')
print('You are currently using version {}\n'.format(VERSION))
# print('If you make use of any features of this toolkit for your research, please remember to cite the SPLAT paper:')
# print('\n{}; Bibcode: {}\n'.format(CITATION,BIBCODE))
# print('If you make use of any spectra or models in this toolkit, please remember to cite the original source.')
#print('Please report any errors are feature requests to our github page, {}\n\n'.format(GITHUB_URL))
if ERROR_CHECKING==True: print('Currently running in error checking mode')


#######################################################
#######################################################
################  VARIOUS UTILITIES  ##################
#######################################################
#######################################################

def checkName(ref,refdict,altref='altname',output=False,verbose=ERROR_CHECKING):
	'''
	Purpose: 
		General usage program to check if a key is present in a dictionary, with the option to look through alternate names

	Required Inputs:
		:param ref: A string containing the reference for lumiosity/SpT relation, should be among the keys and alternate names in refdict
		:param refdict: dictionary containing empirical relation information

	Optional Inputs:
		None

	Output:
		A string containing SPLAT's default name for a given reference set, or False if that reference is not present

	Example:

	>>> import splat
	>>> print(splat.checkDict('filippazzo',splat.SPT_LBOL_RELATIONS))
		filippazzo2015
	>>> print(splat.checkDict('burgasser',splat.SPT_BC_RELATIONS))
		False
	'''
	refc = copy.deepcopy(ref)

# check reference	
	if not isinstance(refc,str): return output
	for k in list(refdict.keys()):
		if refc==k: output = k
		if altref in list(refdict[k].keys()):
			if refc in [x for x in list(refdict[k][altref])]: output = k

# return correct key or indicate an error
	if output == False:
		if verbose==True: print('\nCould not find item {} in input dictionary; try: {}'.format(ref,list(refdict.keys())))
	return output


#######################################################
#######################################################
#####  BASIC SPECTRAL MANIPULATION AND ANALYSIS  ######
#######################################################
#######################################################

def compareSpec(f1,f2,unc,weights=[],verbose=ERROR_CHECKING):
	'''
	Main fitting function, computes chi square with optimal scale factor
	'''
# weighting - can be used to mask bad pixels or weight specific regions
	if len(weights)!=len(f1): wt = numpy.ones(len(f1))
	else: wt=numpy.array(weights)
# mask out bad pixels in either spectrum or uncertainty
	w = numpy.where(numpy.logical_and(numpy.isnan(f1+f2+unc)==False,unc!=0))
	dof = len(f1[w])
	if dof<=1: raise ValueError('Not enough flux or noise values are non-nan')
# compute chi-square
	scl = numpy.nansum(wt[w]*f1[w]*f2[w]/(unc[w]**2))/numpy.nansum(wt[w]*(f2[w]**2)/(unc[w]**2))
	chi = numpy.nansum(wt[w]*((f1[w]-scl*f2[w])**2)/(unc[w]**2))
	return chi, scl, dof-1


def resample(sp,wave,method='integrate',smooth=1,verbose=ERROR_CHECKING):
	'''
	Resamples a spectrum to a given wavelength grid with optional smoothing
	'''
# prepare wavelength grid
	if splat.isUnit(wave): wv=wave.to(sp.wave.unit).value
	else: wv = copy.deepcopy(wave)
	wshift = 2.*numpy.nanmedian(numpy.roll(wv,-1)-wv)

# prepare spectrum object
	spc = copy.deepcopy(sp)
	spc.trim([wv[0]-3.*wshift,wv[-1]+3.*wshift])

# run interpolation
	flx = [numpy.nan]*len(wv)
	unc = [numpy.nan]*len(wv)
	smind = int(smooth)
	for i,w in enumerate(wv):
		if i<smind: wrng = [w-(wv[smind]-w),wv[smind]]
		elif i>=len(wave)-smind: wrng = [wv[i-smind],w+(w-wv[i-smind])]
		else: wrng = [wv[i-smind],wv[i+smind]]
		wsel = numpy.where(numpy.logical_and(sp.wave.value>=wrng[0],sp.wave.value<=wrng[1]))
		cnt = len(sp.flux[wsel])
# expand range
		if cnt <= 1:
			wsel = numpy.where(numpy.logical_and(sp.wave.value>=wrng[0]-wshift,sp.wave.value<=wrng[1]+wshift))
			cnt = len(sp.flux[wsel])
		if cnt >= 1:
			flx0 = sp.flux.value[wsel]
			unc0 = sp.noise.value[wsel]
			wv0 = sp.wave.value[wsel]			
			wn = numpy.where(~numpy.isnan(flx0))
			if len(flx0[wn])>0:
				if method.lower() in ['mean','mn','average','ave']:
					flx[i] = numpy.nanmean(flx0[wn])
					unc[i] = numpy.nanmean(unc0[wn])/((len(unc0[wn])-1)**0.5)
				if method.lower() in ['weighted mean','wmn','weighted']:
					wts = 1./unc0[wn]**2
					if numpy.isnan(numpy.nanmin(wts))==True: wts = numpy.ones(len(wv0[wn]))
					flx[i] = numpy.nansum(wts*flx0[wn])/numpy.nansum(wts)
					unc[i] = (numpy.nansum(wts*unc0[wn]**2)/numpy.nansum(wts))**0.5
				elif method.lower() in ['integrate','int']:
					wts = 1./unc0[wn]**2
					if numpy.isnan(numpy.nanmin(wts))==True: wts = numpy.ones(len(wv0[wn]))
					if cnt > 1: 
						flx[i] = numpy.trapz(wts*flx0[wn],wv0[wn])/numpy.trapz(wts,wv0[wn])
						unc[i] = (numpy.trapz(wts*unc0[wn]**2,wv0[wn])/numpy.trapz(wts,wv0[wn]))**0.5
					else:
						flx[i] = numpy.nansum(wts*flx0[wn])/numpy.nansum(wts)
						unc[i] = (numpy.nansum(wts*unc0[wn]**2)/numpy.nansum(wts))**0.5
					# unc[i] = (numpy.trapz(numpy.ones(len(wv0[wn])),wv0[wn])/numpy.trapz(1/unc0[wn]**2,wv0[wn]))**0.5
					# flx[i] = numpy.trapz(flx0[wn],wv0[wn])/numpy.trapz(numpy.ones(len(wv0[wn])),wv0[wn])
# median by default
				else:
					flx[i] = numpy.nanmedian(flx0[wn])
					unc[i] = flx[i]/numpy.nanmedian(flx0[wn]/unc0[wn])
#					unc[i] = numpy.nanmedian(unc0[wn])/((len(unc0[wn])-1)**0.5)

# return Spectrum object
	return splat.Spectrum(wave=numpy.array(wv)*sp.wave.unit,flux=flx*sp.flux.unit,noise=unc*sp.flux.unit,name=sp.name)



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
	>>> ucdmcmc.info('btsettl')

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
		for x in ['model','flux']:
			if x in kys: kys.remove(x)
		for k in kys:
			vals = list(set(list(mpars[k])))
			vals.sort()
			if isinstance(mpars[k].iloc[0],float)==True:
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
	Generates a wavelength array by specifying range and either a resolution or constant step size
	'''
# prepare wavelength range
	wunit = DEFAULT_WAVE_UNIT
	if len(wave_range) != 2: raise ValueError('input wave length range must be a 2-element list or numpy array, you passed {}'.format(wave_range))
	if splat.isUnit(wave_range): 
		wunit = wave_range.unit
		wv=wave_range.value
	if splat.isUnit(wave_range[0]):
		wunit = wave_range.unit
		wv=[x.value for x in wave_range]
	else: wv = copy.deepcopy(wave_range)

# generate wavelength grid based on different methods
	if method in ['resolution','res']:
		if verbose==True: print('Generate wavelength grid from {} to {} at constant resolution {}'.format(wv[0]*wunit,wv[1]*wunit,wstep))
		wave = [wv[0]]
		while wave[-1] <= wv[1]: wave.append(wave[-1]*(1+1/wstep))
#	elif method.lower() in ['step','linear']:
	else:
		if verbose==True: print('Generate wavelength grid from {} to {} at constant step size {}'.format(wv[0]*wunit,wv[1]*wunit,wstep*wunit))
		wave = [wv[0]]
		while wave[-1] <= wv[1]: wave.append(wave[-1]+wstep)

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

def writeWave(wave,file='wave.csv',overwrite=True,verbose=ERROR_CHECKING):
	'''
	Writes wavelength array to file
	'''	
	if os.path.exists(file)==True:
		if overwrite==False: raise ValueError('WARNING: wave file {} is already in place; set overwrite=True to overwrite'.format(file))
		else:
			if verbose==True: print('WARNING: overwriting wave file {}'.format(file))
	dp = pandas.DataFrame()
	if splat.isUnit(wave): dp['wave'] = wave.value
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


def getModelSet(setname='',instrument='SPEX-PRISM',wavefile='',prefix=MODEL_FILE_PREFIX,wprefix=WAVE_FILE_PREFIX,info=False,verbose=ERROR_CHECKING):
	'''
	gets model set that is already saved and wavelength grid
	'''	
# any models available?
	# allfiles = glob.glob(os.path.join(MODEL_FOLDER,'{}*.h5'.format(prefix)))
	# allwfiles = glob.glob(os.path.join(MODEL_FOLDER,'{}*.csv'.format(prefix)))
	# if len(allfiles)==0: 
	# 	print('No stored model files available in ucdmcmc folder {}'.format(MODEL_FOLDER))
	# 	return

# list what provided models are available	
	if info==True or setname=='':
		modelInfo(model=setname)
		return

# construct expected file name and check it's there
	if os.path.exists(setname)==True: 
		file=copy.deepcopy(setname)
		instrument=((os.path.basename(file).split('_'))[-1]).replace('.h5','')
	elif os.path.exists(os.path.join(MODEL_FOLDER,setname))==True: 
		file=os.path.join(MODEL_FOLDER,setname)
		instrument=((os.path.basename(file).split('_'))[-1]).replace('.h5','')
	elif os.path.exists('{}{}_{}.h5'.format(prefix,setname,instrument))==True: 
		file = '{}{}_{}.h5'.format(prefix,setname,instrument)
	else: file = os.path.join(MODEL_FOLDER,'{}{}_{}.h5'.format(prefix,setname,instrument))
	if os.path.exists(file)==False:
		print('WARNING: model set file for {} cannot be found, check your file name'.format(setname))
		modelInfo()
		raise ValueError

# read in appropriate wave file
	if wavefile!='':
		if os.path.exists(wavefile)==True: wfile=copy.deepcopy(wavefile)
		elif os.path.exists(os.path.join(MODEL_FOLDER,wavefile))==True: wfile=os.path.join(MODEL_FOLDER,wavefile)
		else: raise ValueError('Could not located wavelength file {}'.format(wavefile))
	else: wfile = os.path.join(MODEL_FOLDER,'{}{}.csv'.format(wprefix,instrument))
	if os.path.exists(wfile)==False:
		print('Could not located wavelength file {}'.format(wfile))
		files = glob.glob(os.path.join(MODEL_FOLDER,'{}.csv'.format(wprefix)))
		if len(files) > 0: 
			print('Available wavelength grids:')
			for f in files: print('\t{}'.format(os.path.basename(f)))
		raise ValueError

# finally read in models and wave grid
	models = readModelSet(file,verbose=verbose)
	wave = readWave(wfile,verbose=verbose)
	return models, wave


def generateModelSet(modelset,wave=DEFAULT_WAVE,output_prefix=MODEL_FILE_PREFIX,constraints={},smooth=1,save_wave=False,verbose=ERROR_CHECKING):
	''' 
	Generates the h5 files containing resampled models
	'''
# check name
	mset = spmdl.checkSpectralModelName(modelset)
	if isinstance(mset,bool):
		print('WARNING: Model set {} is not contained in SPLAT, cannot run this'.format(modelset))
		return
	outfile = output_prefix+'.h5'
	if verbose==True: print('Processing {} models'.format(mset))

# prepare wavelength grid
	if splat.isUnit(wave): wv = wave.to(DEFAULT_WAVE_UNIT).value
	else: wv = copy.deepcopy(wave)

# load up RAW model parameters
	mpars = spmdl.loadModelParameters(mset,instrument='RAW')['parameter_sets']
	dp = pandas.DataFrame(mpars)
	if dp['instrument'].iloc[0]!='RAW':
		print('WARNING: No RAW models for set {} are available in SPLAT, cannot run this'.format(modelset))
		return

# make constraints if needed
	for k in list(constraints.keys()):
		if k in list(spmdl.SPECTRAL_MODEL_PARAMETERS.keys()) and k in list(dp.columns):
			if spmdl.SPECTRAL_MODEL_PARAMETERS['teff']['type'] == 'continuous':
				if verbose==True: print('Constaining {} to {}-{}'.format(k,constraints[k][0],constraints[k][1]))
				dp = dp[dp[k]>=constraints[k][0]]
				dp = dp[dp[k]<=constraints[k][1]]
			else:
				par = list(set(list(dp[k])))
				if verbose==True: print('Constaining {} to within {}'.format(k,constraints[k]))
				for p in par:
					if p not in constraints[k]: dp = dp[dp[k]!=p]
# run the models
	if verbose==True: print('Processing {:.0f} {} models'.format(len(dp),mset))
	pars = []
# using a very dumbed down tqdm to save on memory issues	
	step = int(len(dp)/10.)
#	for i in tqdm(range(len(dp))):
	for i in range(len(dp)):
		if i!=0 and numpy.mod(i,step)==0 and verbose==True: print('\t{:.0f}% complete'.format(i/step*10),end='\r')
		par = dict(dp.iloc[i])
		mdl = spmdl.loadModel(**par)
		mdlsm = resample(mdl,wv,smooth=smooth,method='integrate')
		par['flux'] = mdlsm.flux.value
		pars.append(par)

# save the models
	dpo = pandas.DataFrame(pars)
	del dpo['instrument']
	if verbose==True: print('Saving {} models to {}'.format(mset,outfile))
	dpo.to_hdf(outfile,'models','w',complevel=4,index=False)
	if save_wave==True: 
		dpw = pandas.DataFrame()
		dpw['wave'] = wv
		outfile = output_prefix+'_wave.csv'
		dpw.to_csv(outfile,index=False)
		if verbose==True: print('Saving wavelength array to {}'.format(outfile))
	return True


def getGridModel(mdls,par,wave,verbose=ERROR_CHECKING):
	'''
	Gets a specific model from a model set
	'''	
# prep wavelegngth array
	if splat.isUnit(wave)==False: wv = wave*DEFAULT_WAVE_UNIT
	else: wv = wave.to(DEFAULT_WAVE_UNIT)

# prep downselect
	kys = list(mdls.keys())
	for x in ['model','flux']:
		if x in kys: kys.remove(x)
	smdls = copy.deepcopy(mdls)

# prep downselect
	for k in kys:
		if k in list(par.keys()): 
			if k=='kzz': smdls = smdls[smdls[k]==str(par[k])]
			else: smdls = smdls[smdls[k]==par[k]]
	if len(smdls)==0: raise ValueError('No models match parameters {}'.format(par))
	elif len(smdls)>1: 
		if verbose==True: print('{:.0f} models statisfy criteria, returning the first one'.format(len(smdls)))
	flx = smdls['flux'].iloc[0]
	mdl = splat.Spectrum(wave=wave,flux=flx*DEFAULT_FLUX_UNIT,name='{} model'.format(mdls['model'].iloc[0]))
#	if 'scale' in list(par.keys()): mdl.scale(par['scale'])
	return mdl


def getInterpModel(mdls,par0,wave,verbose=ERROR_CHECKING):
	'''
	Generates an interpolated model from 
	'''
# prep wavelength array
	if splat.isUnit(wave)==False: wv = wave*DEFAULT_WAVE_UNIT
	else: wv = wave.to(DEFAULT_WAVE_UNIT)

# get model information
	mset = spmdl.checkSpectralModelName(mdls['model'].iloc[0])
	if not isinstance(mset,bool): defaults = splat.SPECTRAL_MODELS[mset]['default']
	else: raise ValueError('Model set {} not set up in splat'.format(mdls['model'].iloc[0]))

# prep partial downslelect
	smdls = copy.deepcopy(mdls)
	mkys = list(mdls.keys())
	for x in ['model','flux']:
		if x in mkys: mkys.remove(x)
	par = copy.deepcopy(par0)
	limits,steps = {},{}

# downselect or bail out if we're outside parameter range
	for k in mkys:
		vals = list(set(list(smdls[k])))
		vals.sort()
		if k in list(par.keys()):
			# if k=='kzz' and isinstance(truepar[k],str)==False: 
			# 	smdls = smdls[smdls[k]=='{:.1f}'.format(truepar[k])]
			if isinstance(smdls[k].iloc[0],str)==True: 
				if par[k] not in vals:
					if verbose==True: print('WARNING: input value {} for parameter {} not in model set, using default value {} instead'.format(par[k],k,defaults[k]))
					par[k] = default[k]
				smdls = smdls[smdls[k]==par[k]]
			else:
#				print(par[k],numpy.nanmin(vals),numpy.nanmax(vals)) 
				if par[k]<numpy.nanmin(vals) or par[k]>numpy.nanmax(vals):
					raise ValueError('Parameter {} = {} outside range of models: {} to {}'.format(k,par[k],numpy.nanmin(vals),numpy.nanmax(vals)))
				valstep = numpy.absolute(numpy.array(vals)-numpy.roll(vals,1))
				step = numpy.nanmedian(valstep[1:])				
				limits[k] = [numpy.nanmax([numpy.nanmin(smdls[k]),par[k]-step]),numpy.nanmin([numpy.nanmax(smdls[k]),par[k]+step])]
				if step>0:
					smdls = smdls[smdls[k]>=limits[k][0]]
					smdls = smdls[smdls[k]<=limits[k][1]]													 
		else: 
			smdls = smdls[smdls[k]==defaults[k]]
			par[k] = defaults[k]
#		if verbose==True: print(k,list(set(list(smdls[k]))),len(smdls))
		if len(smdls)==0: raise ValueError('No model satisfies parameter selection (failed at {} = {})'.format(k,par[k]))
#	mkys = list(limits.keys())
#	print(smdls)
	
# eliminate degenerate parameters
	mkys0 = copy.deepcopy(mkys)
	for k in mkys:
		if len(set(list(smdls[k])))<2: 
			mkys0.remove(k)
			par[k] = smdls[k].iloc[0]
	mkys = copy.deepcopy(mkys0)
	
# prep models for griddata interpolation
	fitvals,parvals = (),[]
	for k in mkys:
		if k=='teff': 
			fitvals+=tuple([[numpy.log10(x) for x in smdls[k]]])
			parvals.append(numpy.log10(par[k]))
		else:
			fitvals+=tuple([list(smdls[k])])
			parvals.append(par[k])
	parvals = numpy.array([parvals])
	fitvals = numpy.transpose(numpy.vstack(fitvals))

# run interpolation
	flx = []
	for i in range(len(smdls['flux'].iloc[0])):
		fs = [numpy.log10(x[i]) for x in smdls['flux']]
		try: flx.append(griddata(fitvals,tuple(fs),parvals,method='linear',rescale=True)[0])
		except: raise ValueError('Insufficient model coverage; try reducing parameter constraints')
	flx = numpy.array(flx)
	flx = 10.**flx
	if numpy.isnan(numpy.nanmedian(flx))==True: raise ValueError('Could not interpolate {} over grid, possibly due to grid gaps'.format(par))
#	print(truepar)
	return splat.Spectrum(wave=wave,flux=flx*DEFAULT_FLUX_UNIT,name='{} model'.format(mdls['model'].iloc[0]))


def getModel(mdls,par,wave,verbose=ERROR_CHECKING):
	try: sp = getGridModel(mdls,par,wave,verbose=verbose)
	except: sp = getInterpModel(mdls,par,wave,verbose=verbose)
	return sp


########################################################################
# FITTING METHODS
########################################################################


def fitGrid(spc,omdls,constraints={},report=True,output_prefix='gridfit_',absolute=False,verbose=ERROR_CHECKING):
	'''
	Fit spectrum to model grid
	assumes spectrum is already resampled to correct wavelength grid
	'''
# make sure object spectrum is sampled to same wavelength scale as models
	if len(spc.flux)!=len(omdls['flux'].iloc[0]):
		raise ValueError('Spectrum and models are not on same wavelength scale; be sure to resample observed spectrum onto model scale')
	spscl = copy.deepcopy(spc)

# constrain models if needed
	mdls = copy.deepcopy(omdls)
	for k in list(constraints.keys()):
		if k in list(mdls.columns):
			if isinstance(mdls[k].iloc[0],str):
				par = list(set(list(dp[k])))
				if verbose==True: print('Constaining {} to within {}'.format(k,constraints[k]))
				for p in par:
					if p not in constraints[k]: mdls = mdls[mdls[k]!=p]
			else:
				if verbose==True: print('Constaining {} to {}-{}'.format(k,constraints[k][0],constraints[k][1]))
				mdls = mdls[mdls[k]>=constraints[k][0]]
				mdls = mdls[mdls[k]<=constraints[k][1]]
	
# run through each grid point
	for x in ['scale','chi','radius','dof']: mdls[x] = [numpy.nan]*len(mdls)
	for jjj in range(len(mdls)):
		chi,scl,dof = compareSpec(spscl.flux.value,numpy.array(mdls['flux'].iloc[jjj]),spscl.noise.value,verbose=verbose)
		mdls['chi'].iloc[jjj] = chi
		mdls['scale'].iloc[jjj] = scl
		mdls['dof'].iloc[jjj] = dof
# radius scaling assuming spectrum is in absolute flux units
		mdls['radius'].iloc[jjj] = (10.*u.pc*(scl**0.5)).to(u.Rsun).value
#	mdls['model'] = [mset]*len(mdls)

# best fit
	mpar = {}
	ibest = numpy.argmin(mdls['chi'])
	if verbose==True: print('Best fit model:')
	for k in list(mdls.keys()): 
		mpar[k] = mdls[k].iloc[ibest]
		if verbose==True and k not in ['flux']: print('\t{} = {}'.format(k,mpar[k]))
	comp = getGridModel(mdls,mpar,spscl.wave,verbose=verbose)
	comp.scale(mpar['scale'])
#	comp = splat.Spectrum(wave=wave,flux=numpy.array(mdls['flux'].iloc[ibest])*mdls['scale'].iloc[ibest]*spscl.flux.unit)
	diff = spscl.flux.value-comp.flux.value
#	dof = numpy.count_nonzero(~numpy.isnan(spscl.flux.value))-1
	rchi = mdls['chi'].iloc[ibest]/mdls['dof'].iloc[ibest]
	mpar['rchi'] = rchi
	if verbose==True: print('\treduced chi2 = {}'.format(rchi))
	# sclstd = numpy.nanstd(diff.flux.value,ddof=1)/numpy.nanmax(spscl.flux.value)
	# mpar['sclstd'] = sclstd

	if report == True:
# save parameters
		outfile = output_prefix+'_parameters.xlsx'
		mdls.drop(columns=['flux'],inplace=True)
		mdls.to_excel(outfile,index=False)
# comparison plot		
		outfile = output_prefix+'_compare.pdf'
		label = '{} model '.format(mdls['model'].iloc[0])
		label+=r'$\chi^2_r$='+'{:.1f}\n'.format(rchi)
		label+='T={:.0f} '.format(mpar['teff'])
		label+='logg={:.2f} '.format(mpar['logg'])
		label+='z={:.2f} '.format(mpar['z'])
		plotCompare(spscl,comp,outfile=outfile,clabel=label,absolute=absolute)
	return mpar


DEFAULT_MCMC_STEPS = {'teff': 25, 'logg': 0.1, 'z': 0.1, 'enrich': 0.05}
def fitMCMC(spc,omdls,p0={},constraints={},nstep=100,interim=50,burn=0.25,threshhold=0.5,pstep=DEFAULT_MCMC_STEPS,absolute=False,report=True,output_prefix='mcmcfit_',verbose=ERROR_CHECKING):
#	radius=numpy.nan,e_radius=numpy.nan,report=True):
	'''
	Fit spectrum to model grid using MCMC interpolation
	assumes spectrum is already resampled to correct wavelength grid
	'''

# make sure object spectrum is sampled to same wavelength scale as models
	if len(spc.flux)!=len(omdls['flux'].iloc[0]):
		raise ValueError('Spectrum and models are not on same wavelength scale; be sure to resample observed spectrum onto model scale')
	spscl = copy.deepcopy(spc)

# constrain models if needed
	mdls = copy.deepcopy(omdls)
	for k in list(constraints.keys()):
		if k in list(mdls.columns):
			if isinstance(mdls[k].iloc[0],str):
				par = list(set(list(dp[k])))
				if verbose==True: print('Constaining {} to within {}'.format(k,constraints[k]))
				for p in par:
					if p not in constraints[k]: mdls = mdls[mdls[k]!=p]
			else:
				if verbose==True: print('Constaining {} to {}-{}'.format(k,constraints[k][0],constraints[k][1]))
				mdls = mdls[mdls[k]>=constraints[k][0]]
				mdls = mdls[mdls[k]<=constraints[k][1]]
	mset = mdls['model'].iloc[0]
	mkys = list(mdls.keys())
	for x in ['model','flux']:
		if x in mkys: mkys.remove(x)

# if no or incomplete fit parameters, conduct an initial grid fit
	chk = True
	for k in mkys: chk=chk and (k in list(p0.keys()))
	if chk==False:
		if verbose==True: print('Running initial grid fit')
		p0 = fitGrid(spc,mdls,absolute=absolute,report=False,verbose=verbose)
		if 'flux' in list(p0.keys()): del p0['flux']
#		if verbose==True: print('\nGrid fit parameters: {}'.format(p0))

# validate steps
	if verbose==True: print('Fitting the following parameters:')
	mkysfit = copy.deepcopy(mkys)
	for k in mkys:
		vals = list(set(list(mdls[k])))
		vals.sort()
		if len(vals)<2: pstep[k] = 0.
		else:
			if k not in list(pstep.keys()):
				if isinstance(mdls[k].iloc[0],str): pstep[k] = -1.
				else: pstep[k] = 0.5*numpy.nanmedian(numpy.absolute(numpy.array(vals)-numpy.roll(vals,1)))
		if pstep[k] == 0: mkysfit.remove(k)
		else:
			if verbose==True: print('\t{}: initial={} step={}'.format(k,p0[k],pstep[k]))
	nparam = len(mkysfit)

# continuous and discrete variables
	pfitc,pfitd = {},{}
	for k in mkysfit: 
		if k in list(p0.keys()):
			if isinstance(mdls[k].iloc[0],str): pfitd[k] = p0[k]
			else: pfitc[k] = p0[k]
		else: 
			default = splat.SPECTRAL_MODELS[mset]['default'][k]
			if isinstance(default,str): pfitd[k] = default
			else: pfitc[k] = default

# some plotting set up
	ylabelpre = 'Scaled '
	if absolute==True: ylabelpre='Absolute '

# initialize MCMC
	cmdl = getModel(mdls,p0,spscl.wave,verbose=verbose)
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
			cmdl = getModel(mdls,pnew,spscl.wave,verbose=verbose)
			if verbose==True: print(i,pnew)
			chinew,scl,_ = compareSpec(spscl.flux.value,cmdl.flux.value,spscl.noise.value,verbose=verbose)
			# if numpy.isnan(radius)==False and numpy.isnan(e_radius)==False:
			# 	chinew+=(((10.*u.pc*(scl**0.5)).to(u.Rsun).value-radius)/e_radius)**2
			cmdl.scale(scl)

# compute statistic
			st,chst = (chinew-chis[-1])/numpy.nanmin(chis),numpy.random.uniform(0,threshhold)
#			st,chst = 2*scipy.stats.f.sf(chinew/chis[-1],dof,dof),numpy.random.uniform(0.5,1)
#			st,chst = dof/(0.5*dof+chinew-chis[-1]),numpy.random.uniform(threshhold,1)
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
# interim save
		if interim>0 and i>0 and numpy.mod(i,interim)==0:
# save parameters
			dpfit = pandas.DataFrame()
			for k in mkys: 
				vals = [p[k] for p in pvals]
				dpfit[k] = vals
			dpfit['chis'] = chis
			dpfit['dof'] = [dof]*len(dpfit)
			dpfit['scale'] = scales
			dpfit['radius'] = [(10.*u.pc*(x**0.5)).to(u.Rsun).value for x in scales]
			outfile = output_prefix+'_parameters.xlsx'
			dpfit.to_excel(outfile,index=False)
# plot comparison
			if verbose==True: print('Saving interim plots')
			pbest = pvals[numpy.argmin(chis)]
#			pbest['radius'] = (10.*u.pc*(scales[numpy.argmin(chis)]**0.5)).to(u.Rsun).value
			cmdl = getModel(mdls,pbest,spscl.wave,verbose=verbose)
			cmdl.scale(scales[numpy.argmin(chis)])
			outfile = output_prefix+'_compare.pdf'
			plotCompare(spscl,cmdl,outfile=outfile,clabel='Best {} model\n'.format(mset)+r'$\chi^2_r$='+'{:.1f}'.format(numpy.nanmin(chis)/dof),absolute=absolute,verbose=verbose)
# plot cornerplot
			plotpars = copy.deepcopy(mkysfit)
			if absolute==True: plotpars.append('radius')
			pltbest = [dpfit[x].iloc[numpy.argmin(dpfit['chis'])] for x in plotpars]
			weights = numpy.array(dof/(dof+dpfit['chis']-numpy.nanmin(dpfit['chis'])))
			outfile = output_prefix+'_corner.pdf'
			plotCorner(dpfit,plotpars,pbest,weights=weights,outfile=outfile,verbose=verbose)
# plot chains
			plotpars.append('chis')
			outfile = output_prefix+'_chains.pdf'
			plotMCMCChains(dpfit,plotpars,outfile=outfile,verbose=verbose)

# best fit after revmoving burn
	pvalsb = pvals[int(burn*nstep):]	
	pbest = pvalsb[numpy.argmin(chis[int(burn*nstep):])]
	if verbose==True: print('Best parameters: {}'.format(pbest))

	if report == True:
# remove initial burn and save
		dpfit = pandas.DataFrame()
		for k in mkys: 
			vals = [p[k] for p in pvals]
			dpfit[k] = vals[int(burn*nstep):]
		dpfit['chis'] = chis[int(burn*nstep):]
		dpfit['dof'] = [dof]*len(dpfit)
		dpfit['scale'] = scales[int(burn*nstep):]
		dpfit['radius'] = [(10.*u.pc*(x**0.5)).to(u.Rsun).value for x in dpfit['scale']]
		outfile = output_prefix+'_parameters.xlsx'
		if verbose==True: print('Saving database of prameters to {}'.format(outfile))
		dpfit.to_excel(outfile,index=False)
# plot comparison
		cmdl = getModel(mdls,pbest,spscl.wave,verbose=verbose)
		cmdl.scale(scales[numpy.argmin(chis)])
		label = '{} model '.format(mdls['model'].iloc[0])
		label+=r'$\chi^2_r$='+'{:.1f}\n'.format(chis[numpy.argmin(chis)]/dof)
		label+='T={:.0f} '.format(pbest['teff'])
		label+='logg={:.2f} '.format(pbest['logg'])
		label+='z={:.2f} '.format(pbest['z'])
		outfile = output_prefix+'_compare.pdf'
		if verbose==True: print('Plotting best fit comparison to {}'.format(outfile))
		plotCompare(spscl,cmdl,outfile=outfile,clabel=label,absolute=absolute,verbose=verbose)
# plot cornerplot
		plotpars = copy.deepcopy(mkysfit)
		for k in plotpars:
			if isinstance(mdls[k].iloc[0],str): plotpars.remove(k)
		if absolute==True: plotpars.append('radius')
		pltbest = [dpfit[x].iloc[numpy.argmin(dpfit['chis'])] for x in plotpars]
		weights = numpy.array(dof/(dof+dpfit['chis']-numpy.nanmin(dpfit['chis'])))
		outfile = output_prefix+'_corner.pdf'
		if verbose==True: print('Plotting corner plot to {}'.format(outfile))
		plotCorner(dpfit,plotpars,pbest,weights=weights,outfile=outfile,verbose=verbose)
# plot chains
		plotpars.append('chis')
		outfile = output_prefix+'_chains.pdf'
		if verbose==True: print('Plotting chain plot to {}'.format(outfile))
		plotMCMCChains(dpfit,plotpars,outfile=outfile,verbose=verbose)

# return - might want to vary this up		
	return pbest		


########################################################################
# PLOTTING FUNCTIONS
########################################################################

def plotCompare(sspec,cspec,outfile='',clabel='Comparison',absolute=False,verbose=ERROR_CHECKING):
	diff = sspec.flux.value-cspec.flux.value

	xlabel = r'Wavelength'+' ({:latex})'.format(sspec.wave.unit)
	ylabel = r'F$_\lambda$'+' ({:latex})'.format(sspec.flux.unit)
	if absolute==True: ylabel='Absolute '+ylabel

	plt.clf()
	plt.figure(figsize=[8,7])
	fg, (ax1,ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4,1]}, sharex=True)
	ax1.step(sspec.wave.value,sspec.flux.value,'k-',linewidth=2,label=sspec.name)
	ax1.step(cspec.wave.value,cspec.flux.value,'m-',linewidth=4,alpha=0.5,label=clabel)
	ax1.legend(fontsize=12)
	ax1.plot([numpy.nanmin(sspec.wave.value),numpy.nanmax(sspec.wave.value)],[0,0],'k--')
	ax1.fill_between(sspec.wave.value,sspec.noise.value,-1.*sspec.noise.value,color='k',alpha=0.3)
	scl = numpy.nanmax(cspec.flux.value)
	scl = numpy.nanmax([scl,numpy.nanmax(sspec.flux.value)])
	ax1.set_ylim([x*scl for x in [-0.1,1.3]])
	ax1.set_xlim([numpy.nanmin(sspec.wave.value),numpy.nanmax(sspec.wave.value)])
	ax1.set_ylabel(ylabel,fontsize=12)
	ax1.tick_params(axis="x", labelsize=0)
	ax1.tick_params(axis="y", labelsize=14)

	ax2.step(sspec.wave.value,diff,'k-',linewidth=2)
	ax2.plot([numpy.nanmin(sspec.wave.value),numpy.nanmax(sspec.wave.value)],[0,0],'k--')
	ax2.fill_between(sspec.wave.value,sspec.noise.value,-1.*sspec.noise.value,color='k',alpha=0.3)
	scl = numpy.nanquantile(diff,[0.02,0.98])
	ax2.set_ylim([2*sc for sc in scl])
	ax2.set_xlim([numpy.nanmin(sspec.wave.value),numpy.nanmax(sspec.wave.value)])
	ax2.set_xlabel(xlabel,fontsize=16)
	ax2.set_ylabel(r'$\Delta$',fontsize=16)
	ax2.tick_params(axis="x", labelsize=14)
	ax2.tick_params(axis="y", labelsize=14)
	plt.tight_layout()
	if outfile!='': plt.savefig(outfile)
	if verbose==True: plt.show()
	return

def plotMCMCChains(dpfit,plotpars,pbest={},outfile='',verbose=ERROR_CHECKING):
	nplot = int(len(plotpars))
	if nplot==0: 
		if verbose==True: print('WARNING: no parameters to plot')
		return
	plt.clf()
	fig = plt.figure(figsize=[2*6,numpy.ceil(nplot/2)*3])
	for i,l in enumerate(plotpars):	
		ax = plt.subplot(int(numpy.ceil(nplot/2)),2,i+1)
		ax.plot(dpfit[l],'k-')
		if l in list(pbest.keys()): 
			ax.plot(numpy.zeros(len(dpfit[l]))+pbest[l],'m--')
		ax.set_xlabel('Step',fontsize=14)
		if l in list(PARAMETER_PLOT_LABELS.keys()): ax.set_ylabel(PARAMETER_PLOT_LABELS[l],fontsize=14)
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
						title_kwargs={"fontsize": 12},label_kwargs={'fontsize': 12}, smooth=1,truths=truths, \
						truth_color='m',verbose=verbose)
	plt.tight_layout()
	if outfile!='': fig.savefig(outfile)
	if verbose==True: plt.show()
	return
