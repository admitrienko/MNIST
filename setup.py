
#!/usr/bin/env python

from setuptools import setup

setup(name='MNIST_Abstraction_Testing',
      version='0.1',
      description='MNIST abstraction testing project.',
      author='Anastasia Dmitrienko',
      author_email='ad3473@columbia.edu',
      #dependency_links=['https://github.com/gamaleldin/rand_tensor/tarball/master#egg=package-1.0'], 
      dependency_links=[
        'https://github.com/admitrienko/rand_tensor.git#egg=rand_tensor-1.0.0'
           ],
      install_requires=['numpy', 
                       'keras',
                       'tensorflow',
                        'matplotlib',
                        'sklearn',
                        'scipy',
                        'setuptools>=41.0.0',
                        'rand_tensor',
                       ],

      packages = ['MNIST_Abstraction_Testing'],
     )
