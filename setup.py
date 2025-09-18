from setuptools import setup, find_packages

setup(
    name='meda',
    version='0.1.1',
    author='Noel Kronenberg',
    author_email='noel.kronenberg@charite.de',
    description='MEDA is a Python package for working with EHR data. It aims to provide utilities for the most important medical data science tasks with publication-ready results. We continuously generalize processes and analyses from our team’s publications and integrate them into the package.',
    url='https://github.com/noelkronenberg/meda',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # data handling
        'pandas',
        'numpy',
        # visualization
        'matplotlib',
        'seaborn',
        'plotly',
        # analysis
        'scikit-learn',
        'statsmodels',
        'stepmix'
    ],
)
