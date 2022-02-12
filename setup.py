from setuptools import find_packages, setup
import os


setup(
    name="dlkp",
    version="0.3",
    author="Amardeep Kumar || Debanjan Mahata",
    author_email="kumaramardipsingh@gmail.com",
    description="A deep learning library for keyphrase extraction and generation",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/midas-research/dlkp",
    download_url="https://github.com/midas-research/dlkp/archive/refs/tags/0.1.tar.gz",
    project_urls={
        "Bug Tracker": "https://github.com/midas-research/dlkp/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    license="Apache License Version 2.0",
    include_package_data=True,
    python_requires=">=3.7",
)
