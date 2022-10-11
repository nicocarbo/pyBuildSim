# -*- coding: utf-8 -*-


import os, setuptools
from codecs import open

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "requirements.txt")) as f:
    required_packages = f.read().splitlines()
    
setuptools.setup(  
    name='pyBuildSim', 
    version='0.0.0',  
    description='Python package for performing simplified building simulation',
    url='https://github.com/nicocarbo/pyBuildSim',  
    author='Nicolas Carbonare', 
    author_email='nicocarbonare@gmail.com',  
    packages=setuptools.find_packages(),
    install_requires=required_packages,
    include_package_data=True
    )
