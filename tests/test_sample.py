import os
import glob
import splat
from ucdmcmc import DEFINED_INSTRUMENTS,getSample,readWave,resample

VERBOSE=True

# check the provided sample spectra
def test_sample():
	for instr in list(DEFINED_INSTRUMENTS.keys()):
		sp = getSample(instr,verbose=VERBOSE)
		assert isinstance(sp,splat.core.Spectrum)

# check spectral re-sampling
def test_resample():
	sp = getSample('SPEX-PRISM',verbose=VERBOSE)
	wave = readWave('NIR',verbose=VERBOSE)
	spc = resample(sp,wave,verbose=VERBOSE)
	assert len(spc.wave)==len(wave)

