from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="BlockTrainer",
    version="0.1",
    author="Connia Ren",
    author_email="conniaren@hotmail.com",
    description="A method to effectively train autoencoder networks on large genetic variant datasets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/conniaren/BlockTrainer",
    project_urls={
        "Bug Tracker": (
            "https://github.com/conniaren/BlockTrainer/issues"
        )
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    packages=find_packages(),
    install_requires=[
        "cyvcf2",
        "numpy",
        "torch",
        "zarr",
        "pytorch-lightning"
    ]
)