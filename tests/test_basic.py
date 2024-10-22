import os
import astropy.units as u
from ucdmcmc import CODE_PATH,MODEL_FOLDER,SPECTRA_FOLDER
from ucdmcmc import isUnit

VERBOSE = True

# check folders are present
def test_folders():
# isunit
	assert os.path.exists(CODE_PATH)
	assert os.path.exists(MODEL_FOLDER)
	assert os.path.exists(SPECTRA_FOLDER)

# check basic functions
def test_isunit():
# isunit
	assert isUnit(u.m)
	assert isUnit(20*u.m)
	assert isUnit(20)==False
	assert isUnit([5,10,15,20]*u.m)
	assert isUnit([5,10,15,20])==False


