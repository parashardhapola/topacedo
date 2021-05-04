from setuptools import setup, find_packages
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


if __name__ == "__main__":
    classifiers = [
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        'Operating System :: POSIX :: Linux',
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.7',
    ]
    keywords = ['store']
    version = open('VERSION').readline().rstrip('\n')
    dependency_links = ['https://github.com/fraenkel-lab/pcst_fast/tarball/master#egg=pcst_fast-1.0.7']
    install_requires = []
    setup(
        name='topacedo',
        description='topacedo',
        long_description=read('README.md'),
        author='Parashar Dhapola',
        url='https://github.com/parashardhapola/topacedo',
        author_email='parashar.dhapola@gmail.com',
        classifiers=classifiers,
        keywords=keywords,
        install_requires=install_requires,
        dependency_links=dependency_links,
        version=version,
        packages=find_packages(),
        include_package_data=False
    )

