#! python

from setuptools import setup, find_packages

with open("./description.txt") as f:
    desctiprion = f.read()

setup(
    name="pcemg",
    version="0.0.1",
    author="TakashiKusachi",
    description=desctiprion,
    install_requires=[
        'torchvision',
        'torch',
        'torch-jtnn >= 0.0.1'
    ],
    extras_require={
        'example':[
            'gitpython','scikit-learn','tqdm'
            ],
        'doc':[
            'sphinx'
            ],
    },
    packages=find_packages(exclude=['example','tests']),
    entry_points={
        'console_scripts':[
        ]
    },
    test_requires=['GitPython'],
    test_suite = 'tests',
)