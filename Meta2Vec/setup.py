from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="Meta2Vec",
    version="0.0.1",
    author="Yuxing Lu",
    author_email="yxlu0613@gmail.com",
    description="Meta2Vec, an embedding for Metabolomics study",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/YuxingLu613/Meta2Vec.git",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)