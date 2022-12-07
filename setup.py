# setup.py
from setuptools import setup
from setuptools import find_packages

# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
    requirements = [x.strip() for x in content]

setup(name='preprocessing',
    description="preprocessing LST data",
    packages=find_packages(), # find packages automatically
    install_requires=requirements)
