import io
import re

from setuptools import find_packages, setup

__version__ = re.search(
    r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    io.open("covjsonkit/version.py", encoding="utf_8_sig").read(),
).group(1)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="covjsonkit",
    version=__version__,
    description="""ECMWF library for encoding and decoding coerageJSON files/objects of meteorlogical features such as
                   vertical profiles and time series.""",
    long_description="",
    url="https://github.com/ecmwf/covjsonkit",
    author="ECMWF",
    author_email="James.Hawkes@ecmwf.int, Adam.Warde@ecmwf.int, Mathilde.Leuridan@ecmwf.int",
    packages=find_packages(),
    zip_safe=False,
    include_package_data=True,
    install_requires=requirements,
    package_data={"covjsonkit": ["data/*.json"]},
)
