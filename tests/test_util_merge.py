import json
import os

import pytest

from covjsonkit import utils


class TestMultipointXarray:
    def setup_method(self):
        current_dir = os.path.dirname(__file__)
        collection1_path = os.path.join(current_dir, "data/test_coverage1.json")
        with open(collection1_path, "r") as f:
            self.collection1 = json.load(f)

        collection2_path = os.path.join(current_dir, "data/test_coverage2.json")
        with open(collection2_path, "r") as f:
            self.collection2 = json.load(f)

    def test_merge(self):
        merged_collection = utils.merge_coverage_collections(self.collection1, self.collection2)
        assert len(merged_collection["coverages"]) == 4

    def test_coverage_collection_types(self):
        self.collection1["type"] = "coverage"
        with pytest.raises(ValueError):
            merged_collection = utils.merge_coverage_collections(self.collection1, self.collection2)  # noqa: F841

    def test_domain_type(self):
        self.collection1["domainType"] = "PointSeries"
        with pytest.raises(ValueError):
            merged_collection = utils.merge_coverage_collections(self.collection1, self.collection2)  # noqa: F841
