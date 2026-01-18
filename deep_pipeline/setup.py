from setuptools import find_packages, setup

setup(
    name="translator-pipeline",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "pytorch-lightning>=1.5.0",
        "polars>=0.18.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "gin-config>=0.4.0",
    ],
)



