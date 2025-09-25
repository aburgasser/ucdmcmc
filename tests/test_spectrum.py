# test spectrum class functions
# import os
# import glob
# import pandas
import numpy as np
import copy
import astropy.units as u
import astropy.constants as const
from ucdmcmc import Spectrum,getSample,DEFAULT_FLUX_UNIT,DEFINED_INSTRUMENTS,DEFAULT_FLAM_UNIT,DEFAULT_FNU_UNIT

VERBOSE = True

# check the provided sample spectra
def test_sample():
	for instr in list(DEFINED_INSTRUMENTS.keys()):
		if DEFINED_INSTRUMENTS[instr]['sample']!='':
			sp = getSample(instr,verbose=VERBOSE)
			assert isinstance(sp,Spectrum)
	return

# test toWavelengths
def test_basic():
	sp = getSample('JWST-NIRSPEC-PRISM',verbose=VERBOSE)
	wave = [1.]
	while wave[-1]<3: wave.append(wave[-1]*(1+1/50))
	sp.toWavelengths(wave)
	assert len(sp.wave)==len(wave)
	assert list(sp.wave.value)==wave
	return

# test spectrum math
def test_math():
	sp1 = getSample('JWST-NIRSPEC-PRISM',verbose=VERBOSE)
	sp2 = getSample('SPEX-PRISM',verbose=VERBOSE)
	sp3 = sp1.copy()
	assert isinstance(sp3,Spectrum)
	assert sp3.flux.unit == sp1.flux.unit
	assert sp3.noise.unit == sp1.noise.unit
	# assert list(sp3.flux.value) == list(sp1.flux.value)
	# assert list(sp3.noise.value) == list(sp1.noise.value)
	sp3 = sp1+sp2
	assert isinstance(sp3,Spectrum)
	assert len(sp3.wave)==len(sp1.wave)
	assert sp3.flux.unit == sp1.flux.unit
	assert sp3.noise.unit == sp1.noise.unit
	sp3 = sp2+sp1
	assert isinstance(sp3,Spectrum)
	assert len(sp3.wave)==len(sp2.wave)
	assert sp3.flux.unit == sp1.flux.unit
	assert sp3.noise.unit == sp1.noise.unit
	sp3 = sp1-sp2
	assert isinstance(sp3,Spectrum)
	assert len(sp3.wave)==len(sp1.wave)
	assert sp3.flux.unit == sp1.flux.unit
	assert sp3.noise.unit == sp1.noise.unit
	sp3 = sp2-sp1
	assert isinstance(sp3,Spectrum)
	assert len(sp3.wave)==len(sp2.wave)
	assert sp3.flux.unit == sp1.flux.unit
	assert sp3.noise.unit == sp1.noise.unit
	sp3 = sp1*sp2
	assert isinstance(sp3,Spectrum)
	assert len(sp3.wave)==len(sp1.wave)
	assert sp3.flux.unit == sp1.flux.unit**2
	assert sp3.noise.unit == sp1.noise.unit**2
	sp3 = sp2*sp1
	assert isinstance(sp3,Spectrum)
	assert len(sp3.wave)==len(sp2.wave)
	assert sp3.flux.unit == sp1.flux.unit**2
	assert sp3.noise.unit == sp1.noise.unit**2
	sp3 = sp1/sp2
	assert isinstance(sp3,Spectrum)
	assert len(sp3.wave)==len(sp1.wave)
	assert sp3.flux.unit == u.dimensionless_unscaled
	assert sp3.noise.unit == u.dimensionless_unscaled
	sp3 = sp2/sp1
	assert isinstance(sp3,Spectrum)
	assert len(sp3.wave)==len(sp2.wave)
	assert sp3.flux.unit == u.dimensionless_unscaled
	assert sp3.noise.unit == u.dimensionless_unscaled
	return

# check masking
def test_mask():
	sp = getSample('JWST-NIRSPEC-PRISM',verbose=VERBOSE)
	sp1 = copy.deepcopy(sp)
# by wavelength
	sp.mask([1.2,2.3],action='remove')
	assert len(sp.wave) < len(sp1.wave)
	sp = copy.deepcopy(sp1)
	sp.mask([1.2,2.3],action='replace',replace_value=np.nan)
	assert len(sp.wave) == len(sp1.wave)
	assert np.nanmedian(sp.flux.value) != np.nanmedian(sp1.flux.value)
	sp = copy.deepcopy(sp1)
# with array
	mask = np.zeros(len(sp.wave))
	mask[:30] = 1
	sp.mask(mask,action='remove')
	assert len(sp.wave) < len(sp1.wave)
	sp = copy.deepcopy(sp1)
	sp.mask(mask,action='replace',replace_value=np.nan)
	assert len(sp.wave) == len(sp1.wave)
	assert np.nanmedian(sp.flux.value) != np.nanmedian(sp1.flux.value)
	sp = copy.deepcopy(sp1)
# by S/N
	sp.maskSN(15,action='remove')
	assert len(sp.wave) < len(sp1.wave)
	sp = copy.deepcopy(sp1)
	sp.maskSN(15,action='replace',replace_value=np.nan)
	assert len(sp.wave) == len(sp1.wave)
	assert np.nanmedian(sp.flux.value) != np.nanmedian(sp1.flux.value)
	sp = copy.deepcopy(sp1)
# trim
	sp.trim([2,3])
	assert len(sp.wave) < len(sp1.wave)
	return

# test conversions
def test_convert():
	sp = getSample('JWST-NIRSPEC-PRISM',verbose=VERBOSE)
	assert sp.flux.unit == DEFAULT_FLAM_UNIT
	sp.fluxConvert(u.Jy)
	assert sp.flux.unit == u.Jy
	sp1 = copy.deepcopy(sp)
	sp.fluxConvert(u.mJy)
	assert sp.flux.unit == u.mJy
	assert np.nanmedian(sp.flux.value) != np.nanmedian(sp1.flux.value)
	return

# test scaling
def test_scale():
	sp = getSample('SPEX-PRISM',verbose=VERBOSE)
	sp.normalize()
	assert np.nanmax(sp.flux.value) == 1.
	sp.normalize(method='median')
	assert np.nanmedian(sp.flux.value) == 1.
	sp.scale(2)
	assert np.nanmedian(sp.flux.value) == 2.
	sp.normalize(1.2)
	smp = sp.sample([1.18,1.22])
	assert np.absolute(smp-1.)<0.1

# SKIPPING FLUX CAL TESTS AS THESE REQUIRE SPLAT
	sp.fluxConvert(u.Jy)
	assert sp.flux.unit == u.Jy
	sp1 = copy.deepcopy(sp)
	sp.fluxConvert(u.mJy)
	assert sp.flux.unit == u.mJy
	assert np.nanmedian(sp.flux.value) > np.nanmedian(sp1.flux.value)
	return

# test shifting
def test_shift():
	sp = getSample('SPEX-PRISM',verbose=VERBOSE)
	pshft = 0.5
	wshft = np.nanmedian(sp.wave.value-np.roll(sp.wave.value,1))*pshft*sp.wave.unit
	rvshft = (wshft.value/np.nanmedian(sp.wave.value))*const.c
	sp1 = copy.deepcopy(sp)
	sp1.shift(pshft)
	sp2 = copy.deepcopy(sp)
	sp2.shift(wshft)
	sp3 = copy.deepcopy(sp)
	sp3.shift(rvshft)
	assert sp1.wave.value[np.argmax(sp1.flux.value)]>sp.wave.value[np.argmax(sp.flux.value)]
	assert sp1.wave.value[np.argmax(sp1.flux.value)]>sp2.wave.value[np.argmax(sp2.flux.value)]
	assert sp2.wave.value[np.argmax(sp2.flux.value)]>sp3.wave.value[np.argmax(sp3.flux.value)]

