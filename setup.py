#!/usr/bin/env python

from setuptools import setup

setup(name='MNIST_Abstraction_Testing',
      version='0.1',
      description='Git template.',
      author='Anastasia Dmitrienko',
      author_email='ad3473@columbia.edu',
      dependency_links=['https://github.com/gamaleldin/rand_tensor/tarball/master#egg=package-1.0'],
      install_requires=['numpy', 
                       'keras',
                       'tensorflow',
                        'matplotlib',
                        'gzip',
                        'sklearn',
                        'scipy',
                        'rand_tensor'
                       ]

     )
