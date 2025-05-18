from setuptools import setup, find_packages

setup(
    name="kernel_methods",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "tqdm>=4.60.0",
        "pillow>=8.0.0",
    ],
    author="Your Names Here",
    author_email="your.email@example.com",
    description="Optimized kernel methods for large-scale image classification",
    keywords="machine learning, kernel methods, scalability, image classification",
    python_requires=">=3.8",
)