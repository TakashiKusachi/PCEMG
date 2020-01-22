#! python

from setuptools import setup, find_packages

with open("./description.txt") as f:
    desctiprion = f.read()

setup(
    name="pcemg",
    version="0.0.1",
    author="TakashiKusachi",
    description=desctiprion,
    install_requires=[],
    packages=find_packages(exclude=['example']),
    entry_points={
        'console_scripts':[
        ]
    }
)