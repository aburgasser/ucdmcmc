import os
import glob
import pandas
import numpy
#import splat
from ucdmcmc import MODEL_FOLDER,MODEL_FILE_PREFIX,DEFINED_SPECTRAL_MODELS,DEFINED_INSTRUMENTS,WAVE_FILE_PREFIX
from ucdmcmc import checkName,isUnit,readWave,readModelSet,getModelSet,getGridModel,getInterpModel
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
				for x in ['model','flux','teff','logg']: assert x in list(fluxes.columns)
				if instr=='SPEX-PRISM': print(mdl,instr,len(fluxes.loc[0,'flux'])==len(wave))
### UNCOMMENT FOLLOWING WHEN FIXED
#				assert len(fluxes.loc[0,'flux'])==len(wave)


# check the provided models
def test_models():
	for mdl in list(DEFINED_SPECTRAL_MODELS.keys()):
		for instr in list(DEFINED_INSTRUMENTS.keys()):
			mfile = os.path.join(MODEL_FOLDER,'{}{}_{}.h5'.format(MODEL_FILE_PREFIX,mdl,instr))
			if os.path.exists(mfile):
				fluxes,wave = getModelSet(mdl,instr,verbose=VERBOSE)
# check these are the same size
				assert len(wave)==len(fluxes.loc[0,'flux'])				
# get one of the grid models (default parameters)
				par = DEFINED_SPECTRAL_MODELS[mdl]['default']
				msp = getGridModel(fluxes,par,wave,verbose=VERBOSE)
				assert isinstance(msp,Spectrum)
				assert len(msp.wave)==len(wave)
				for x in list(DEFINED_SPECTRAL_MODELS[mdl]['default'].keys()):
					print(x,msp.parameters[x],DEFINED_SPECTRAL_MODELS[mdl]['default'][x])
					assert msp.parameters[x]==DEFINED_SPECTRAL_MODELS[mdl]['default'][x]
# get an interpolated model (may not work?)
				# par['teff'] = par['teff']+25
				# msp = getInterpModel(fluxes,par,wave,verbose=VERBOSE)
				# assert isinstance(msp,type(splat.core.Spectrum))
				# assert len(msp.wave)==len(wave)
				# for x in list(DEFINED_SPECTRAL_MODELS[mdl]['default'].keys()):
				# 	assert msp.parameters[x]==par[x]

