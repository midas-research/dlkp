from setuptools import find_packages, setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="dlkp",
    version="0.1",
    description="A deep learning library for keyphrase extraction and generation",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Debanjan Mahata",
    author_email="debanjanmahata85@gmail.com",
    url="https://github.com/midas-research/dlkp",
    packages=find_packages(exclude="tests"),  # same as name
    license="Apache License Version 2.0",
    install_requires=required,
    include_package_data=True,
    python_requires=">=3.8.10",
)
