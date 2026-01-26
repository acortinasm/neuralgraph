from setuptools import setup, find_packages

setup(
    name="neuralgraph",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyarrow>=14.0.0",
        "pandas",
        "torch",
        "requests",
        # "torch-geometric", # Optional, depends on user environment
    ],
    author="NeuralGraphDB Team",
    description="Python client for NeuralGraphDB via Arrow Flight",
)
