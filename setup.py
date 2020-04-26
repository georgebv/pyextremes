import setuptools


with open('README.rst', encoding='utf-8') as file:
    long_description = file.read()


setuptools.setup(
    name='pyextremes',
    version='0.1.0',
    author='Georgii Bocharov',
    author_email='bocharovgeorgii@gmail.com',
    description='Extreme value analysis (EVA) in Python',
    long_description=long_description,
    url='https://github.com/georgebv/pyextremes',
    license='GPLv3',
    keywords='statistics extreme eva coastal ocean marine engineering ',
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
    ]
)
