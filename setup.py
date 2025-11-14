from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="live-music-classifier",
    version="0.2.2",
    author="Daniel A. G. Manzato",
    author_email="dmanzato@gmail.com",
    description="Live music genre classification using PyTorch and GTZAN dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dmanzato/live-music-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "live-music-train=train:main",
            "live-music-predict=predict:main",
            "live-music-stream=scripts.stream_infer:main",
            "live-music-evaluate=evaluate:main",
        ],
    },
)

