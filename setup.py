from setuptools import setup

import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='GCNfold',
      py_modules=['GCNfold'],
      install_requires=[
          'torch'
      ],
)
