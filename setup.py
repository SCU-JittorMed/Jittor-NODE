# setup.py
from setuptools import setup, find_packages

setup(
    name='jittornode',
    version='0.1',
    packages=find_packages(),  # Finds the `jittornode/` package
    install_requires=[
        'jittor',  # Add other dependencies here
    ],
    python_requires='>=3.6',
)