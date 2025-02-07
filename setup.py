from setuptools import setup, find_packages

setup(
    name="josiah",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.20.0',
        'matplotlib>=3.0.0',
    ],
    author="Arturo Díaz",
    author_email="arturo@entropy.tech",
    description="A package for generating synthetic MMM data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/entropyx/josiah",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)