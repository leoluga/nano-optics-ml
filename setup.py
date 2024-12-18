from setuptools import setup, find_packages

setup(
    name="nano-optics-ml",
    version="0.1.0",
    description="Machine learning models for analyzing nano-optics data.",
    author="Leonardo Antonio Lugarini",
    author_email="leonardo.lugarini@example.com",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "tensorflow",
        "torch",
        "pytest",
        "jupyter",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
