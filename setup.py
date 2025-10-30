from setuptools import setup, find_packages

setup(
    name="orion-v1",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.4.1",
        "numpy",
        "opencv-python",
        "Pillow",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
