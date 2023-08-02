import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    description="Solves eikonal function on instance maps.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.0',
        'triton>=2.0.0'
    ],
)