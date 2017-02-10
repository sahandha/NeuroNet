#! python3

from setuptools import setup

setup(
    name='NeuroNet',
    version='0.1',
    description='An agent-based model of a network of neurons',
    url='http://github.com/sahandha/NeuroNet',
    author='Sahand Hariri',
    author_email='sahandha@gmail.com',
    license='open',
    packages=['NeuroNet'],
    install_requires=['GeneralModel','networkx','numpy'],
    dependency_links=['http://github.com/sahandha/GeneralModel'],
    zip_safe=False
    )
