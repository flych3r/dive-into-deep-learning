import os
import re

from setuptools import find_packages, setup

requirements = [
    'ipython',
    'jupyter',
    'matplotlib',
    'numpy',
    'pandas'
]

# don't import d2l to get __version__ since it has deps
ver_re = re.compile(r"__version__ = \"([\.\d]+).*")
with open(os.path.join('d2l', '__init__.py')) as f:
    lines = f.readlines()
for line in lines:
    m = ver_re.match(line)
    if m:
        version = m.group(1)
        break
setup(
    name='d2l',
    version=version,
    python_requires='>=3.5',
    author='D2L Developers',
    author_email='d2l.devs@gmail.com',
    url='https://d2l.ai',
    description='Dive into Deep Learning',
    license='MIT-0',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
