#!/usr/bin/env python
"""Script for setting up module."""
import io
from os import path
from imp import load_source
from setuptools import setup, find_packages

__version__ = load_source('looking_glass.version', 'looking_glass/version.py').__version__

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with io.open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '')
                    for x in all_reqs if 'git+' not in x]

setup(name='looking-glass',
      author='Development Seed',
      author_email='',
      version=__version__,
      description=('Tool for applying deep learning to building detection.'),
      url='https://github.com/developmentseed/looking-glass',
      license='MIT',
      classifiers=['Intended Audience :: Developers',
                   'License :: Freeware',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6'],
      keywords='',
      entry_points={'console_scripts': ['looking-glass=looking_glass.main:cli']},
      packages=find_packages(exclude=['docs', 'tests*']),
      include_package_data=True,
      install_requires=install_requires,
      dependency_links=dependency_links)
