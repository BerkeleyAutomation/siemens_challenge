"""
Setup of Surface Decluttering with Mobile Robots
Author: Ajay Tanwani
"""
from setuptools import setup

setup(name='tpc',
      version='0.1.dev0',
      description='surface decluttering with mobile robots',
      author='Ajay Tanwani',
      author_email='ajay.tanwani@berkeley.edu',
      package_dir={'': 'src'},
      packages=['tpc', 'tpc.config', 'tpc.perception', 'tpc.manipulation', 'tpc.data', 'tpc.offline', 'tpc.detection']
     )
