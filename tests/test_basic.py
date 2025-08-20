import os
import astropy.units as u
from ucdmcmc import CODE_PATH,MODEL_FOLDER,SPECTRA_FOLDER
from ucdmcmc import isUnit,isNumber

VERBOSE = True

# check folders are present
def test_folders():
# isunit
	assert os.path.exists(CODE_PATH)
	assert os.path.exists(MODEL_FOLDER)
	assert os.path.exists(SPECTRA_FOLDER)

# check isunit
def test_isunit():
# isunit
	assert isUnit(u.m)
	assert isUnit(20*u.m)
	assert isUnit(20)==False
	assert isUnit([5,10,15,20]*u.m)
	assert isUnit([5,10,15,20])==False

# check isnumber
def test_isunit():
# isunit
	assert isNumber(20)
	assert isNumber('20')
	assert isNumber(20*u.m)
	assert isNumber(u.m)==False
	assert isNumber([5,10,15,20]*u.m)==False
	assert isNumber([5,10,15,20])==False



