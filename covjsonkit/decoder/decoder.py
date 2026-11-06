import functools
import json
from abc import ABC, abstractmethod

import numpy as np

from covjsonkit.Coverage import Coverage
from covjsonkit.CoverageCollection import CoverageCollection
from covjsonkit.utils import coverage_to_coveragecollection


def _is_numeric_or_null(array) -> bool:
    """True if every element of an object array is ``None`` or a real number (not ``bool``/``str``)."""
    return all(
        value is None or (isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)))
        for value in array.ravel()
    )


def nulls_as_nan(method):
    """Decorate a ``to_xarray`` method so that null range values come out as ``NaN``.

    Points without data are encoded with ``null`` range values (``None`` once parsed).
    Building a dataset from nested lists that contain ``None`` gives an ``object`` array;
    this casts those variables to ``float`` so the nulls become ``NaN``. Only genuinely
    numeric variables are converted: a variable holding strings (or any other non-numeric
    value) is left as it is, even if its strings happen to look like numbers.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        result = method(self, *args, **kwargs)
        for dataset in result if isinstance(result, list) else [result]:
            for name, variable in dataset.data_vars.items():
                if variable.dtype == object and _is_numeric_or_null(variable.values):
                    dataset[name] = variable.astype(float)
        return result

    return wrapper


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
            self.covjson = coverage_to_coveragecollection(self.covjson)
            self.type = self.get_type()
            self.coverage = CoverageCollection(self.covjson)

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
        if "mars:metadata" not in self.coverage.coverages[0]:
            return mars_metadata
        else:
            for coverage in self.coverage.coverages:
                mars_metadata.append(coverage["mars:metadata"])
        return mars_metadata

    def _point_groups(self, domain_key, slot_key):
        """Group coverages into one group per requested point, in first-seen order.

        Coverages with the same ``domain_key`` (spatial/time domain) and ``label``
        belong to the same point. When several requested points snapped to the same
        grid point, their coverages repeat for each ``slot_key`` (e.g. number and
        forecast date), so the n-th coverage for a given domain, label and slot
        belongs to the n-th such point.
        """
        groups = {}
        order = []
        seen = {}
        for coverage in self.covjson["coverages"]:
            label = coverage.get("mars:metadata", {}).get("label")
            domain = domain_key(coverage)
            slot = (domain, label, slot_key(coverage))
            occurrence = seen.get(slot, 0)
            seen[slot] = occurrence + 1
            key = (domain, label, occurrence)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(coverage)
        return [groups[key] for key in order]

    def _area_groups(self):
        """Group MultiPoint coverages into one group per requested area (e.g. polygon), in request order.

        Each area has one coverage per (number, forecast date, step, time), so the n-th coverage for a
        given label and slot belongs to the n-th area with that label. Output without labels or
        repeated slots is a single group.
        """

        def slot_key(coverage):
            meta = coverage.get("mars:metadata", {})
            return (
                meta.get("number"),
                meta.get("Forecast date"),
                str(meta.get("step")),
                tuple(coverage["domain"]["axes"]["t"]["values"]),
            )

        return self._point_groups(lambda coverage: None, slot_key)

    def _to_xarray_per_area(self):
        """``to_xarray`` for MultiPoint decoders: one dataset per requested area, or a single dataset as before."""
        groups = self._area_groups()
        if len(groups) <= 1:
            return self._to_xarray_single()
        datasets = []
        for group in groups:
            covjson = dict(self.covjson)
            covjson["coverages"] = group
            datasets.append(type(self)(covjson)._to_xarray_single())
        return datasets

    def _point_dataset_attrs(self, group):
        """Dataset attributes for one point: shared MARS metadata plus the point's own label."""
        attrs = {key: val for key, val in self.mars_metadata[0].items() if key not in ("date", "step", "label")}
        label = group[0].get("mars:metadata", {}).get("label")
        if label is not None:
            attrs["label"] = label
        return attrs

    @staticmethod
    def forecast_reference(coverage):
        """Return a coverage's ``"Forecast date"``, or its first valid time when absent.

        Reanalysis coverages (``class=ce``, ``stream=efcl``) carry no forecast
        date; their valid time (from ``t`` or the composite ``t`` coordinate)
        identifies them instead.
        """
        mars_metadata = coverage.get("mars:metadata", {})
        if "Forecast date" in mars_metadata:
            return mars_metadata["Forecast date"]
        axes = coverage["domain"]["axes"]
        if "t" in axes:
            return axes["t"]["values"][0]
        composite = axes.get("composite", {})
        if "t" in composite.get("coordinates", []):
            return composite["values"][0][composite["coordinates"].index("t")]
        return 0

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

    @abstractmethod
    def to_geotiff(self):
        pass
