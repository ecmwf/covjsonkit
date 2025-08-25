import json
from abc import ABC, abstractmethod

from covjsonkit.Coverage import Coverage
from covjsonkit.CoverageCollection import CoverageCollection


class Decoder(ABC):
    def __init__(self, covjson):
        """
        Base class for decoding CovJSON data.

        The Decoder class provides functionality to parse and extract information from CovJSON
        data, supporting both Coverage and CoverageCollection types. It handles parameters,
        referencing systems, Mars metadata, and coverages, and provides abstract methods for
        further processing and conversion of the decoded data.

        Attributes:
            covjson (dict): The CovJSON data being decoded, either as a dictionary or loaded from a file.
            type (str): The type of CovJSON data, either "Coverage" or "CoverageCollection".
            coverage (Coverage or CoverageCollection): An instance representing the decoded coverage data.
            parameters (list): A list of parameter names included in the CovJSON data.
            coordinates (list): A list of referencing coordinates extracted from the CovJSON data.
            mars_metadata (list): A list of Mars metadata extracted from the coverages.
            coverages (list): A list of individual coverages within the CovJSON data.

        Methods:
            get_type(): Returns the type of the CovJSON data ("Coverage" or "CoverageCollection").
            get_parameters(): Returns a list of parameter names included in the CovJSON data.
            get_parameter_metadata(parameter): Returns metadata for a specific parameter.
            get_referencing(): Extracts and returns referencing coordinates from the CovJSON data.
            get_mars_metadata(): Extracts and returns Mars metadata from the coverages.

        Abstract Methods:
            get_ranges(): Abstract method for extracting range data from the CovJSON.
            get_domains(): Abstract method for extracting domain information from the CovJSON.
            get_coordinates(): Abstract method for extracting coordinate data from the CovJSON.
            get_values(): Abstract method for extracting values from the CovJSON.
            to_geopandas(): Abstract method for converting the CovJSON data to a GeoPandas DataFrame.
            to_xarray(): Abstract method for converting the CovJSON data to an xarray Dataset.
        """
        # if python dictionary no need for loading, otherwise load json file
        if isinstance(covjson, dict):
            self.covjson = covjson
        elif isinstance(covjson, str):
            with open(covjson) as json_file:
                self.covjson = json.load(json_file)
        else:
            raise TypeError("Covjson must be dictionary or covjson file")

        self.type = self.get_type()

        if self.type == "Coverage":
            self.coverage = Coverage(self.covjson)
        elif self.type == "CoverageCollection":
            self.coverage = CoverageCollection(self.covjson)
        else:
            raise TypeError("Type must be Coverage or CoverageCollection")

        self.parameters = self.get_parameters()
        self.coordinates = self.get_referencing()
        self.mars_metadata = self.get_mars_metadata()
        self.coverages = self.coverage.coverages

    def get_type(self):
        return self.covjson["type"]

    def get_parameters(self):
        return list(self.covjson["parameters"].keys())

    def get_parameter_metadata(self, parameter):
        return self.covjson["parameters"][parameter]

    def get_referencing(self):
        coordinates = []
        for coords in self.covjson["referencing"]:
            coordinates.append(coords["coordinates"])
        return [coord for sublist in coordinates for coord in sublist]

    def get_mars_metadata(self):
        mars_metadata = []
        for coverage in self.coverage.coverages:
            mars_metadata.append(coverage["mars:metadata"])
        return mars_metadata

    @abstractmethod
    def get_ranges(self):
        pass

    @abstractmethod
    def get_domains(self):
        pass

    @abstractmethod
    def get_coordinates(self):
        pass

    @abstractmethod
    def get_values(self):
        pass

    @abstractmethod
    def to_geopandas(self):
        pass

    @abstractmethod
    def to_xarray(self):
        pass

    @abstractmethod
    def to_geojson(self):
        pass
