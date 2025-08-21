import os
from ucdmcmc import generateWave,writeWave,readWave

VERBOSE = True

# check wave generation, writing and reading
def test_wave():
	wrng = [0.5,5.0]
	res = 1000
	step = 0.01
	wave = generateWave(wrng,res,method='resolution',verbose=VERBOSE)
	assert len(wave) == 2305
	wave = generateWave(wrng,step,method='step',verbose=VERBOSE)
	assert len(wave) == 452
# save and read
	fname = 'test_wave.csv'	
	writeWave(wave,fname,verbose=VERBOSE)
	assert os.path.exists(fname)
	wavein = readWave(fname,verbose=VERBOSE)
	assert len(wavein) == 452
# clean up
	os.remove(fname)
