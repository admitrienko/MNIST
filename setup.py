
#!/usr/bin/env python

from setuptools import setup

setup(name='MNIST-Abstraction-Testing',
      version='0.1',
      description='MNIST abstraction testing project.',
      author='Anastasia Dmitrienko',
      author_email='ad3473@columbia.edu',
      dependency_links=['https://github.com/gamaleldin/rand_tensor/tarball/master#egg=package-1.0'],
      install_requires=['numpy', 
                       'keras',
                       'tensorflow',
                        'matplotlib',
                        'sklearn',
                        'scipy',
                        'setuptools>=41.0.0'
                       ],
      packages = ['MNIST-Abstraction-Testing'],
     )
