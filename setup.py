from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='vgenerator',
    version='0.0.1',
    description='Vacancy generator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/RobotVeraDS/vacancy-generator',
    author='Robot Vera',
    author_email='ischenko.dmitry@gmail.com',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='vacancy nlp rnn lstm generation',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=['torch', 'numpy', 'tqdm'],
)
