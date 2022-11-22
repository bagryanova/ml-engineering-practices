import setuptools


def read_requirements():
    with open('requirements.txt', 'r') as f:
        return list([l for l in f])


setuptools.setup(
    name='sound_classifiers',
    version='0.0.2',
    author='Ekaterina Bagrianova',
    description='Some voice classifiers to Male/Female classes',
    packages=['sound_classifiers'],
    install_requires=read_requirements(),
)
