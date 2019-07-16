# MNIST Abstraction Testing 

<p align="center">
<a href="https://travis-ci.org/admitrienko/MNIST-Abstraction-Testing"><img alt="Build Status" src="https://travis-ci.org/admitrienko/MNIST-Abstraction-Testing.svg?branch=master"></a>

The code in this repository is part of a project investigating the properties of neural representations that encode multiple variables in an abstract format simultaneously, drawing from the findings in "The geometry of abstraction in hippocampus and prefrontal cortex" found [here](https://www.biorxiv.org/content/biorxiv/early/2018/12/09/408633.full.pdf).

This code recreates the simulated multi-layer neural network trained with back-propagation from the paper, using [MNIST](http://yann.lecun.com/exdb/mnist/) data.
We then use the tensor maximum entropy [TME](https://github.com/gamaleldin/rand_tensor) method to generate surrogate data and investigate the properties of this data that shares the same tensor marginal covariances and the tensor marginal means as the original MNIST data.
