from setuptools import setup, find_packages

setup(
    name='torchregression',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
    ],
    author='Sébastien Fabbro',
    author_email='sebfabbro@gmail.com',
    description='A collection of regression loss functions and utilities for PyTorch.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sfabbro/torchregression',
    classifiers=[ 
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)