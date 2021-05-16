#!/usr/bin/env python

from setuptools import setup

setup(name='motion_learning_direction_space',
      version='1.0',
      description='Motion Learning in Direction Space',
      author='Lukas Huber',
      author_email='lukas.huber@epfl.ch',
      packages=['motion_learning_direction_space',
                'motion_learning_direction_space.example_folder'],
      scripts=['scripts/test_gmm.py',
               'scripts/test_Amatrix.py'],
      package_dir={'': 'src'}
     )
