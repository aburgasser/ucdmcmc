import os
from ucdmcmc import getSample,fitGrid, fitMCMC, fitEMCEE, DEFINED_SPECTRAL_MODELS, DEFINED_INSTRUMENTS, MODEL_FOLDER, MODEL_FILE_PREFIX

VERBOSE = True

# test gridfit
# TBD: check parameters are returned, fits conducted for each of the baseline models
def test_gridfit():
	instr = 'JWST-NIRSPEC-PRISM'
	sp = getSample(instr,verbose=VERBOSE)
# check relevant models files are present and correctly formatted	
	for mdl in list(DEFINED_SPECTRAL_MODELS.keys()):
		for instr in list(DEFINED_INSTRUMENTS.keys()):
			mfile = os.path.join(MODEL_FOLDER,'{}{}_{}.h5'.format(MODEL_FILE_PREFIX,mdl,instr))
			if os.path.exists(mfile):
				pass
	return

# test MCMCfit
# TBD: check parameters are returned, fits conducted for each of the baseline models
def test_mcmcfit():
	instr = 'JWST-NIRSPEC-PRISM'
	sp = getSample(instr,verbose=VERBOSE)
# check relevant models files are present and correctly formatted	
	for mdl in list(DEFINED_SPECTRAL_MODELS.keys()):
		for instr in list(DEFINED_INSTRUMENTS.keys()):
			mfile = os.path.join(MODEL_FOLDER,'{}{}_{}.h5'.format(MODEL_FILE_PREFIX,mdl,instr))
			if os.path.exists(mfile):
				pass
	return

# test emcee fit
# TBD: check parameters are returned, fits conducted for each of the baseline models
def test_emceefit():
	instr = 'JWST-NIRSPEC-PRISM'
	sp = getSample(instr,verbose=VERBOSE)
# check relevant models files are present and correctly formatted	
	for mdl in list(DEFINED_SPECTRAL_MODELS.keys()):
		for instr in list(DEFINED_INSTRUMENTS.keys()):
			mfile = os.path.join(MODEL_FOLDER,'{}{}_{}.h5'.format(MODEL_FILE_PREFIX,mdl,instr))
			if os.path.exists(mfile):
				pass
	return
