import os
import numpy as np
import astropy.units as u
from ucdmcmc import CODE_PATH,MODEL_FOLDER,SPECTRA_FOLDER,ALT_MODEL_FOLDER
from ucdmcmc import isUnit,isNumber,isArray,checkName
from ucdmcmc import PARAMETERS,DEFINED_INSTRUMENTS,DEFINED_SPECTRAL_MODELS

VERBOSE = True

# check folders are present
def test_folders():
	assert os.path.exists(CODE_PATH)
	assert os.path.exists(MODEL_FOLDER)
	assert os.path.exists(ALT_MODEL_FOLDER)
	assert os.path.exists(SPECTRA_FOLDER)

# check checkname
def test_checkname():
	for var in [PARAMETERS,DEFINED_INSTRUMENTS,DEFINED_SPECTRAL_MODELS]:
		for x in list(var.keys()): 
			assert checkName(x,var)==x
			assert checkName('asdbaisubdiau',var)==False
			if 'altname' in list(var[x].keys()):
				for y in var[x]['altname']:
					assert checkName(y,var)==x

# check isunit
def test_isunit():
	assert isUnit(u.m)
	assert isUnit(20*u.m)
	assert isUnit(20)==False
	assert isUnit([5,10,15,20]*u.m)
	assert isUnit([5,10,15,20])==False

# check isnumber
def test_isnumber():
	assert isNumber(20)
	assert isNumber('20')
	assert isNumber(20*u.m)
	assert isNumber(u.m)==False
	assert isNumber([5,10,15,20]*u.m)==False
	assert isNumber([5,10,15,20])==False

# check isnumber
def test_isarray():
	assert isArray(20)==False
	assert isArray([1,2])
	assert isArray(np.array([1,2]))
	assert isArray((1,2))
	assert isArray({1,2})
	assert isArray([1,2]*u.m)
	assert isArray(np.array([1,2])*u.m)





