"""
AI-Assisted Brain Tumor Localization and Classification

A deep learning project for brain tumor classification and segmentation
using MRI images from the BRISC2025 dataset.

Installation:
    pip install -e .

For development:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Core dependencies
INSTALL_REQUIRES = [
    "tensorflow>=2.10.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "matplotlib>=3.4.0",
    "Pillow>=8.0.0",
    "PyYAML>=6.0",
]

# Development dependencies
DEV_REQUIRES = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
]

# Jupyter/notebook dependencies
NOTEBOOK_REQUIRES = [
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
    "ipywidgets>=7.0.0",
]

setup(
    name="brain-tumor-detection",
    version="1.0.0",
    author="Qadeer Haider",
    author_email="",
    description="AI-Assisted Brain Tumor Localization and Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Qadeer-Haider/AI-Assisted-Brain-Tumor-Localization-and-Classification-in-Medical-Imaging",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "notebook": NOTEBOOK_REQUIRES,
        "all": DEV_REQUIRES + NOTEBOOK_REQUIRES,
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords=[
        "brain tumor",
        "medical imaging",
        "deep learning",
        "classification",
        "segmentation",
        "MRI",
        "computer vision",
        "tensorflow",
        "transfer learning",
    ],
    entry_points={
        "console_scripts": [
            "train-classifier=scripts.train_classifier:main",
            "evaluate-classifier=scripts.evaluate_classifier:main",
            "predict-tumor=scripts.predict_single:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
