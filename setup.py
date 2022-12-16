# setup.py
from setuptools import setup
from setuptools import find_packages

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
    requirements = [x.strip() for x in content]

setup(name='future-proofing-cities',
    version='1.0.0',
    description="prediction of delta T for city",
    packages=find_packages(), # find packages automatically
    install_requires=requirements)
