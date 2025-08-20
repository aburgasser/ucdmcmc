import os
#import glob
#import splat
from ucdmcmc import DEFINED_INSTRUMENTS,getSample,readWave,resample, Spectrum

VERBOSE=True

# check the provided sample spectra
def test_sample():
	for instr in list(DEFINED_INSTRUMENTS.keys()):
		if DEFINED_INSTRUMENTS[instr]['sample']!='':
			sp = getSample(instr,verbose=VERBOSE)
			assert isinstance(sp,Spectrum)

# check spectral re-sampling
def test_resample():
	sp = getSample('SPEX-PRISM',verbose=VERBOSE)
	wave = readWave('NIR',verbose=VERBOSE)
	spc = resample(sp,wave,verbose=VERBOSE)
	assert len(spc.wave)==len(wave)

