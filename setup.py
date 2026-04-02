"""
Setup script for DataPipelineEnv package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
req_file = this_directory / "requirements.txt"
if req_file.exists():
    with open(req_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)

setup(
    name="openenv-project",
    version="1.0.0",
    author="Hackathon Scaler",
    author_email="hackathon@example.com",
    description="DataPipelineEnv - OpenEnv Environment for Data Pipeline Management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hackathon-scaler/openenv-project",
    project_urls={
        "Bug Tracker": "https://github.com/hackathon-scaler/openenv-project/issues",
        "Documentation": "https://github.com/hackathon-scaler/openenv-project#readme",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "datapipeline-env=src.openenv_project.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)