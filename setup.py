"""Setup configuration for mimic4dataprep package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-r')
        ]

setup(
    name="mimic4dataprep",
    version="1.0.0",
    author="Michael Parkes",  # Update with your name
    author_email="michael.parkes@ualberta.ca",  # Update with your email
    description="Data preprocessing tools for MIMIC-IV",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mdparkes/mimic4dataprep",  # Update with your GitHub URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",  # Update if using different license
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "mimic4dataprep": [
            "resources/*.csv",
            "resources/*.yaml",
            "resources/*.yml",
        ],
    },
    entry_points={
        "console_scripts": [
            "mimic4dataprep-extract-subjects=mimic4dataprep.scripts.extract_subjects:main",
            "mimic4dataprep-validate-events=mimic4dataprep.scripts.validate_events:main",
            "mimic4dataprep-extract-episodes=mimic4dataprep.scripts.extract_episodes_from_subjects:main",
            "mimic4dataprep-split-data=mimic4dataprep.scripts.split_train_and_test:main",
            "mimic4dataprep-hash-tables=mimic4dataprep.tests.hash_tables:main",
        ],
    },
    keywords=[
        "mimic",
        "mimic-iv", 
        "healthcare",
        "medical-data",
        "data-preprocessing",
        "ehr",
        "electronic-health-records",
    ],
)
