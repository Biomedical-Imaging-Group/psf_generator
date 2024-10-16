import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PSF Generator",
    version="0.0.1",
    author="",
    author_email="",
    description="PSF Generator.",
    long_description=long_description,
    url=" ",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # "torch",
        # "abc",
        # "math",
        # "numpy",
        # "scipy",
        # "matplotlib",
        # "zernikepy",
        # "tqdm",
    ],
    include_package_data=True,
)
