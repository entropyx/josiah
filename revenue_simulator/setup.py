from setuptools import setup, find_packages

setup(
    name="revenue_simulator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.20.0',
        'matplotlib>=3.0.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for simulating and visualizing revenue data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/revenue_simulator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 