from setuptools import setup

setup(
    name='bidsio',
    version='0.0.3',
    packages=['bidsio'],
    url='https://github.com/npnl/bidsio',
    license='Apache 2.0',
    author='Alexandre Hutton',
    install_requires=[
        'numpy',
        'nibabel',
        'bids'
    ],
    author_email='50920802+AlexandreHutton@users.noreply.github.com',
    description='BIDS IO for working with multiple BIDS datasets.'
)
