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
        'Programming Language :: Python :: 3.8',
    ]
    keywords = ['store']
    version = open('VERSION').readline().rstrip('\n')
    install_requires = [x.strip() for x in open("requirements.txt")]
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
        version=version,
        packages=find_packages(),
        include_package_data=False
    )
