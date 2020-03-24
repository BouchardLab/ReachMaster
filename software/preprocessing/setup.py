"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
# Get package dependencies from the requirements file
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name='preprocessing',
    version="0.0.1",
    description='process data produced by ReachMaster',
    long_description=long_description,
    author='Brian Gereke',
    author_email='bgereke@utexas.edu',
    url='https://github.com/BouchardLab/ReachMaster',   
    packages=find_packages(exclude=()),
    install_requires=requirements,
    classifiers=[
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'License :: OSI Approved :: BSD License',
    'Operating System :: Linux Ubuntu 16.04',
    ],
    python_requires='~=3.6'
    )
