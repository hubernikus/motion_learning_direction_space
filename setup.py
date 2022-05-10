#!/usr/bin/env python

from setuptools import setup

setup(
    name='motion_learning_direction_space',
    version='0.0.1',
    description='Motion Learning in Direction Space',
    author='Lukas Huber',
    author_email='lukas.huber@epfl.ch',
    packages=['motion_learning_direction_space',
              'motion_learning_direction_space.example_folder'],
    scripts=[],
    package_dir={'': 'src'}
)
