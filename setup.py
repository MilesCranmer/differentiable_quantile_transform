from setuptools import setup, find_packages

setup(
    name='diffquantiles',  # Required
    version='1.0.0',  # Required
    description='Differentiable quantile transform',  # Optional
    url='https://github.com/MilesCranmer/differentiable_quantile_transform',  # Optional
    author='Miles Cranmer',  # Optional
    packages=find_packages(),  # Required
    install_requires=['torch']  # Optional
)
