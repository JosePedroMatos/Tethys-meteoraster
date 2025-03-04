'''
python setup.py bdist_wheel
'''

from setuptools import setup, find_packages

setup(
    name='MeteoRaster',  # Replace with your module's name
    version='0.2',
    packages=find_packages(),
    description='Module for handling distributed ensemble and probabilistic forecast data',
    author='Jose Pedro Matos',
    author_email='jose.matos@tecnico.ulisboa.pt',
)