from setuptools import setup, find_packages

setup(
    name="orion-v1",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy"
    ],
)
