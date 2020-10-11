import pathlib

import setuptools

here = pathlib.Path(__file__).parent.resolve()


def get_long_description():
    with open(here / "README.rst", encoding="utf-8", mode="r") as file:
        return file.read()


def get_version():
    with open(here / "src" / "pyextremes" / "__init__.py", mode="r") as file:
        version_line = [
            line.strip() for line in file.readlines() if "__version__" in line
        ][0]
        return version_line.split("=")[-1].strip().strip('"')


setuptools.setup(
    name="pyextremes",
    version=get_version(),
    description="Extreme Value Analysis (EVA) in Python",
    long_description=get_long_description(),
    long_description_content_type="text/x-rst",
    author="Georgii Bocharov",
    author_email="bocharovgeorgii@gmail.com",
    url="https://github.com/georgebv/pyextremes",
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    license="MIT",
    keywords=[
        "statistics",
        "extreme",
        "extreme value analysis",
        "eva",
        "coastal",
        "ocean",
        "marine",
        "environmental",
        "engineering",
    ],
    platforms=["linux", "windows", "mac"],
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=True,
    install_requires=["emcee>=3.0", "matplotlib", "numpy", "pandas", "scipy"],
    python_requires=">=3.7",
    project_urls={
        "GitHub": "https://github.com/georgebv/pyextremes",
        "PyPI": "https://pypi.org/project/pyextremes/",
        "conda-forge": "https://anaconda.org/conda-forge/pyextremes",
    },
)
