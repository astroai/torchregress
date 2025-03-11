from setuptools import setup, find_packages
import os

# Read the contents of README.md file
try:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = 'A collection of regression loss functions and utilities for PyTorch.'

setup(
    name='torchregression',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
    ],
    extras_require={
        'test': [
            'pytest',
            'pytest-cov',
        ],
        'dev': [
            'black',
            'flake8',
            'mypy',
        ],
        'docs': [
            'sphinx',
        ],
    },
    python_requires='>=3.9',
    author='Sébastien Fabbro',
    author_email='sebfabbro@gmail.com',
    description='A collection of regression loss functions and utilities for PyTorch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sfabbro/torchregression',
    project_urls={
        'Bug Tracker': 'https://github.com/sfabbro/torchregression/issues',
    },
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
    keywords='pytorch, regression, machine learning, deep learning, loss functions',
)