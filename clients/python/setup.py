from setuptools import setup, find_packages

setup(
    name="neuralgraph",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "pyarrow>=14.0.0",
        "pandas",
        "requests",
    ],
    extras_require={
        "torch": ["torch"],
        "torch-geometric": ["torch", "torch-geometric"],
        "langchain": ["langchain>=0.1.0", "langchain-community>=0.0.1"],
        "all": [
            "torch",
            "torch-geometric",
            "langchain>=0.1.0",
            "langchain-community>=0.0.1",
        ],
    },
    author="NeuralGraphDB Team",
    description="Python client for NeuralGraphDB with LangChain integration",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
