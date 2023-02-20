from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="HMKG",
    version="0.0.1",
    author="Yuxing Lu",
    author_email="yxlu0613@gmail.com",
    description="HMKG, a Knowledge Graph on Human Metabolome",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/YuxingLu613/HMKG-Human-Metabolome-Knowledge-Graph",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)