#! python

from setuptools import setup, find_packages

with open("./description.txt") as f:
    desctiprion = f.read()

setup(
    name="pcemg",
    version="1.0.0",
    author="TakashiKusachi",
    description=desctiprion,
    install_requires=[
        'torchvision',
        'torch',
        'torch-jtnn >= 0.0.1'
    ],
    package_data ={
        'pcemg':[
            'data/*.ini'
            'scripts/data/*.ini',
            ]
        },
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