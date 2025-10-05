#!/usr/bin/env python3
"""
Setup script for TaxoCapsNet package.

To install the package:
    pip install -e .

To build distribution:
    python setup.py sdist bdist_wheel
"""

from setuptools import setup, find_packages

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="taxocapsnet",
    version="1.0.0",
    author="Adarsh Kesharwani, Tahami Syed, Shravan Tiwari, Swaleha Deshmukh",
    author_email="akesherwani900@gmail.com",
    description="A taxonomy-aware capsule network for autism prediction from gut microbiome data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/TaxoCapsNet",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800"
        ]
    },
    entry_points={
        "console_scripts": [
            "taxocapsnet=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords="capsule networks, microbiome, autism, taxonomy, deep learning, bioinformatics",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/TaxoCapsNet/issues",
        "Source": "https://github.com/yourusername/TaxoCapsNet",
        "Documentation": "https://github.com/yourusername/TaxoCapsNet/wiki",
    },
)