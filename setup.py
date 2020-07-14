
from setuptools import find_packages, setup

requirements = [
    'matplotlib',
    'numpy',
    'pandas'
]

version = 0.1

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
