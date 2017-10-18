#!/usr/bin/env python

from setuptools import setup

setup(
    name='powerflow',
    version='0.1',
    author='gallar.r.g@gmail.com',
    author_email='',
    packages=['powerflow'],
    entry_points={
          'console_scripts': [
              'powerflow = powerflow.cli:run_cli'
          ]
           },
    scripts='',
    url='',
    license='',
    description='',
    long_description='',
    install_requires=[],
)
