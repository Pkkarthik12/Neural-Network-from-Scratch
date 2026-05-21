from setuptools import setup, find_packages

setup(
    name="neural-network-from-scratch",
    version="1.0.0",
    description="A fully-connected neural network built from scratch using NumPy",
    author="Your Name",
    author_email="you@example.com",
    url="https://github.com/yourusername/neural-network-from-scratch",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "examples": ["scikit-learn>=1.3.0", "matplotlib>=3.7.0"],
        "dev": ["pytest>=7.0.0"],
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
