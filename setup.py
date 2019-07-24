#!/usr/bin/env python

from setuptools import setup

setup(name='MNIST_Abstraction_Testing',
      version='0.1',
      description='Git template.',
      author='Anastasia Dmitrienko',
      author_email='ad3473@columbia.edu',
      install_requires=['numpy', 
                       'keras',
                       'tensorflow',
                        'matplotlib',
                        'gzip',
                        'randtensor',
                        'sklearn',
                        'scipy'
                       ],
      packages=['git_template']
     )
