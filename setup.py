import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quoracle",
    version="0.0.1",
    author="Michael Whittaker",
    author_email="mwhittttaker@gmail.com",
    description=("A library for modelling, analyzing, and optimizing quorum " +
                 "systems"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mwhittaker/quorums",
    packages=setuptools.find_packages(),
    install_requires =[
        "matplotlib",
        "numpy",
        "pulp",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
