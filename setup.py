#!/usr/bin/env python

from setuptools import setup

setup(name='git_template',
      version='0.1',
      description='Git template.',
      author='Anastasia Dmitrienko',
      author_email='ad3473@columbia.edu',
      install_requires=['numpy', 
                       'keras',
                       'tensorflow',
                        'matplotlib',
                        'gzip',
                        'random',
                        'randtensor',
                        'math',
                        'sklearn',
                        'itertools',
                        'scipy'
                       ],
      packages=['git_template']
     )
