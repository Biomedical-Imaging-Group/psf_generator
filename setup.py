import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="propagators",
    version="0.0.1",
    author="",
    author_email="",
    description="PSF Generator.",
    long_description=long_description,
    url=" ",
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},  # Tell setuptools that packages are under 'src'
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "matplotlib", "numpy", "scipy", "torch", "tqdm", "zernikepy",],
    python_requires=">=3.8",
)