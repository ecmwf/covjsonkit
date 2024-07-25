| :warning: This project is BETA and will be experimental for the foreseeable future. Interfaces and functionality are likely to change. DO NOT use this software in any project/software that is operational. |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

<h3 align="center">
<img src="./docs/images/Logo_Destination_Earth_Colours.png" width=60%>
</br>

# covjsonkit

<p align="center">
  <a href="https://github.com/ecmwf/covjsonkit/actions/workflows/ci.yaml">
  <img src="https://github.com/ecmwf/covjsonkit/actions/workflows/ci.yaml/badge.svg" alt="ci">
</a>
  <a href="https://codecov.io/gh/ecmwf/covjsonkit"><img src="https://codecov.io/gh/ecmwf/covjsonkit/branch/develop/graph/badge.svg"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://github.com/ecmwf/covjsonkit/releases"><img src="https://img.shields.io/github/v/release/ecmwf/covjsonkit?color=blue&label=Release&style=flat-square"></a>
</p>
<p align="center">
  <a href="#concept">Concept</a> •
  <a href="#installation">Installation</a> •
  <a href="#example">Example</a> •
  <a href="#testing">Testing</a>
</p>

## Concept

Covjsonkit is an ECMWF library for encoding and decoding coverageJSON files/objects of meteorlogical features such as vertical profiles and time series.

* Encodes and decodes CoverageJSON objects
* Convert CoverageJSON files to and from xarray
* Works in conjunction with ECMWFs Polytope feature extraction library

Current features implemented:

* Time Series
* Vertical Profile
* Bounding Box
* Frame
* Path
* Wkt Polygons
* Shapefiles

## Installation

Install the polytope software with Python 3 (>=3.7) from GitHub directly with the command

    python3 -m pip install git+ssh://git@github.com/ecmwf/covjsonkit.git@develop

or from PyPI with the command

    python3 -m pip install covjsonkit

## Example

The library consists of an encoder and a decoder element. The decoder can be used to decode valid coverageJSON files that can be then be edited and accessed via the api. It can also be used to convert to ther formats such as xarray.

### Decoder

    from covjsonkit.api import Covjsonkit

    decoder = Covjsonkit().decode(coverage.covjson)

    print(decoder.type)
    print(decoder.parameters)
    print(decoder.get_referencing())

    ds = decoder.to_xarray()

### Encoder

The following example encodes data output from the polytope feature extraction library assuming polytope_output is a valid output from polytope.

    from covjsonkit.api import Covjsonkit

    encoder = Covjsonkit().encode("CoverageCollection", "BoundingBox")
    res = encoder.from_polytope(polytope_output)

## Testing

Python unit tests can be run with pytest:

    python -m pytest 

When a pull request is merged into develop or main a github actions CI pipeline is triggered to test formatting and unit tests.
