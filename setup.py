#!/bin/env python 

import sys

from setuptools import setup
from setuptools_rust import RustExtension

setup(rust_extensions=[RustExtension("mem_psd.rust_ext")],
      packages=["mem_psd"],
      include_package_data=True)

