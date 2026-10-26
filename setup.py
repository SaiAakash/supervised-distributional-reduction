from setuptools import setup, find_packages

setup(
    name="distributional-reduction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "tqdm",
        "pillow",
        "scikit-learn",
        "geoopt",
        "pot",
    ],
) 