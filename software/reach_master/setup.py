"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='reach_master',
    version="0.0.1",
    description='robotic system for rat reaching',
    long_description=long_description,
    author='Brian Gereke',
    author_email='bgereke@utexas.edu',
    url='https://github.com/BouchardLab/ReachMaster',    
    packages=find_packages(exclude=('tests','bin','temp')),
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'License :: OSI Approved :: BSD License',
        'Operating System :: Linux Ubuntu 16.04',
    ],
    python_requires='~=2.7',
    	)