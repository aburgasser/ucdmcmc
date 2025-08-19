#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

#try:
#  from setuptools import setup, find_packages
#  setup
#except ImportError:
#  from distutils.core import setup
#  setup

#from distutils.core import setup

from setuptools import setup

setup()

# from __future__ import unicode_literals
# from setuptools import setup, find_packages
# from ucdmcmc.core import VERSION

# setup(
#   name = "UCDMCMC",
#   version = VERSION,
#   packages = find_packages(),
# #  packages = find_packages(exclude=['docs','tests']),

#   # Project uses reStructuredText, so ensure that the docutils get
#   # installed or upgraded on the target machine
#   install_requires = [
#     'astropy',
#     'corner',
#     'h5py',
#     'matplotlib',
#     'numpy',
#     'pandas',
#     'scipy',
#     'splat',
#     'statsmodels',
#     'tqdm',
#   ],

#   package_dir = {'ucdmcmc': 'ucdmcmc'},    
#   package_data = {'ucdmcmc': ['models/*','spectra/*','tests/*']},
# #      'reference/Filters': [
# #        'reference/Filters/*.txt',
# #      ],
# #      'reference/EvolutionaryModels': [
# #        'reference/EvolutionaryModels/Baraffe/*.txt',
# #        'reference/EvolutionaryModels/Burrows/*.txt',
# #        'reference/EvolutionaryModels/Saumon/*.txt',
# #      ],
# #      'reference/Spectra': [
# #        'reference/Spectra/*.fits',
# #      ],
# #      'reference/SpectralModels': [
# #        'reference/SpectralModels/BTSettl2008/*.txt',
# #        'reference/SpectralModels/BTSettl2015/*.txt',
# #        'reference/SpectralModels/burrows06/*.txt',
# #        'reference/SpectralModels/drift/*.txt',
# #        'reference/SpectralModels/morley12/*.txt',
# #        'reference/SpectralModels/morley14/*.txt',
# #        'reference/SpectralModels/saumon12/*.txt',
# #      ],
# #  },
#   include_package_data=True,

#   zip_safe = True,
#   use_2to3 = False,
#   classifiers=[
#       'Development Status :: 3 - Alpha',
#       'Intended Audience :: Science/Research',
#       'License :: OSI Approved :: MIT License',
#       'Operating System :: OS Independent',
#       'Programming Language :: Python :: 3',
#       'Programming Language :: Python :: 3.9',
#       'Topic :: Scientific/Engineering :: Astronomy',
#       'Topic :: Scientific/Engineering :: Physics'
#   ],

#   # metadata for upload to PyPI
#   author = "Adam Burgasser",
#   author_email = "aburgasser@ucsd.edu",
#   description = "UCD MCMC fitting program",
# #  long_description = long_description,
#   license = "MIT",
# #    download_url='%s/astropy-%s.tar.gz' % (DOWNLOAD_BASE_URL, VERSION),
#   keywords = ['ultracool dwarfs','spectral fitting'],
#   #url = "http://www.browndwarfs.org/splat/",   # project home page, if any


# )
