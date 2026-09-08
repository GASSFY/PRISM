#!/usr/bin/env python
from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="asdq",
    version="0.1.0",
    description="PRISM Phase-1: pseudo-quant ASD column retention for multimodal models",
    author="PRISM",
    packages=setuptools.find_packages(),
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
