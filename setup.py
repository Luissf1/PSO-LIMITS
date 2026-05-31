"""
Setup configuration for the PSO limits study package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pso-limits-study",
    version="1.0.0",
    author="Luis Adrián Silva Reyes",
    author_email="tu-email@dominio.com",
    description="Study of velocity limits in PSO for PID tuning of quadrotor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/tesis-limites-pso",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov>=4.0", "black>=22.0", "flake8>=5.0"],
        "plot": ["matplotlib>=3.5", "seaborn>=0.12"],
    },
)