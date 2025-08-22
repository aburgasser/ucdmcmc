import os
import glob
import pandas
import numpy
import requests
from ucdmcmc import MODEL_FOLDER,ALT_MODEL_FOLDER,MODEL_FILE_PREFIX,DEFINED_SPECTRAL_MODELS,DEFINED_INSTRUMENTS,WAVE_FILE_PREFIX,MODEL_URL
from ucdmcmc import checkName,isUnit,readWave,readModelSet,getModelSet,getGridModel,getInterpModel,downloadModel
from ucdmcmc import Spectrum

VERBOSE = True

# check the provided models
def test_input_models():
# are there models in MODEL_FOLDER?	
	allfiles = glob.glob(os.path.join(MODEL_FOLDER,'{}*.h5'.format(MODEL_FILE_PREFIX)))
	assert len(allfiles) > 0

# check reading in of these files	
	for f in allfiles:
		assert isinstance(readModelSet(f),type(pandas.DataFrame()))

# check relevant wave files are present	
	for instr in list(DEFINED_INSTRUMENTS.keys()):
		assert checkName(instr,DEFINED_INSTRUMENTS,verbose=VERBOSE)==instr
		assert os.path.exists(os.path.join(MODEL_FOLDER,'{}{}.csv'.format(WAVE_FILE_PREFIX,instr)))
		wave = readWave(instr,verbose=VERBOSE)
		assert isUnit(wave)
		assert isinstance(wave.value,numpy.ndarray)

# check relevant models files are present and correctly formatted	
	for mdl in list(DEFINED_SPECTRAL_MODELS.keys()):
		for instr in list(DEFINED_INSTRUMENTS.keys()):
			mfile = os.path.join(MODEL_FOLDER,'{}{}_{}.h5'.format(MODEL_FILE_PREFIX,mdl,instr))
			if os.path.exists(mfile):
				fluxes,wave = getModelSet(mdl,instr,verbose=VERBOSE)
				assert isUnit(wave)
				assert isinstance(wave.value,numpy.ndarray)
				assert isinstance(fluxes,type(pandas.DataFrame()))
# check these are the same size
				assert len(wave)==len(fluxes.loc[0,'flux'])				
				for x in ['model','flux','teff','logg']: assert x in list(fluxes.columns)
# get one of the grid models (default parameters)
				par = DEFINED_SPECTRAL_MODELS[mdl]['default']
				msp = getGridModel(fluxes,par,wave,verbose=VERBOSE)
				assert isinstance(msp,Spectrum)
				assert len(msp.wave)==len(wave)
				for x in list(DEFINED_SPECTRAL_MODELS[mdl]['default'].keys()):
					print(x,msp.parameters[x],DEFINED_SPECTRAL_MODELS[mdl]['default'][x])
					assert msp.parameters[x]==DEFINED_SPECTRAL_MODELS[mdl]['default'][x]
# get one of the interpolated models 
# SKIPPING FOR NOW AS THIS DOESN'T ALWAYS WORK



# check the provided models
def test_remote():
	response = requests.get(MODEL_URL)
	assert response.status_code == 200
	mfile = os.path.join(MODEL_URL,'models','{}{}_{}.h5'.format(MODEL_FILE_PREFIX,'btsettl08','SPEX-PRISM'))
	print(mfile)
	response = requests.get(mfile)
	assert response.status_code == 200
