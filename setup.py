import setuptools

with open("README.rst", encoding="utf-8") as file:
    long_description = file.read()

setuptools.setup(
    name="pyextremes",
    version="1.1.0",
    description="Extreme Value Analysis (EVA) in Python",
    long_description=long_description,
    long_description_content_type="test/x-rst",
    author="Georgii Bocharov",
    author_email="bocharovgeorgii@gmail.com",
    url="https://github.com/georgebv/pyextremes",
    packages=setuptools.find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
    ],
    license="MIT",
    keywords="statistics extreme eva coastal ocean marine engineering",
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["numpy", "scipy", "pandas", "matplotlib", "emcee>=3.0"],
    python_requires=">=3.7",
)
