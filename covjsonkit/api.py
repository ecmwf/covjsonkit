import logging

from conflator import Conflator

import covjsonkit.decoder.BoundingBox
import covjsonkit.decoder.Frame
import covjsonkit.decoder.Path
import covjsonkit.decoder.Shapefile
import covjsonkit.decoder.TimeSeries
import covjsonkit.decoder.VerticalProfile
import covjsonkit.decoder.Wkt
import covjsonkit.encoder.BoundingBox
import covjsonkit.encoder.Frame
import covjsonkit.encoder.Path
import covjsonkit.encoder.Shapefile
import covjsonkit.encoder.TimeSeries
import covjsonkit.encoder.VerticalProfile
import covjsonkit.encoder.Wkt

from .config import CovjsonKitConfig

features_encoder = {
    "pointseries": covjsonkit.encoder.TimeSeries.TimeSeries,
    "verticalprofile": covjsonkit.encoder.VerticalProfile.VerticalProfile,
    "boundingbox": covjsonkit.encoder.BoundingBox.BoundingBox,
    "shapefile": covjsonkit.encoder.Shapefile.Shapefile,
    "frame": covjsonkit.encoder.Frame.Frame,
    "path": covjsonkit.encoder.Path.Path,
    "polygon": covjsonkit.encoder.Wkt.Wkt,
}
features_decoder = {
    "pointseries": covjsonkit.decoder.TimeSeries.TimeSeries,
    "verticalprofile": covjsonkit.decoder.VerticalProfile.VerticalProfile,
    "boundingbox": covjsonkit.decoder.BoundingBox.BoundingBox,
    "shapefile": covjsonkit.decoder.Shapefile.Shapefile,
    "frame": covjsonkit.decoder.Frame.Frame,
    "path": covjsonkit.decoder.Path.Path,
    "polygon": covjsonkit.decoder.Wkt.Wkt,
}


class Covjsonkit:
    def __init__(self, config=None):
        # If no config check default locations
        if config is None:
            self.conf = Conflator(app_name="covjsonkit", model=CovjsonKitConfig).load()
            logging.debug("Config loaded from file: %s", self.conf)  # noqa: E501
        # else initialise with provided config
        else:
            self.conf = CovjsonKitConfig.model_validate(config)
            logging.debug("Config loaded from dictionary: %s", self.conf)  # noqa: E501

    def encode(self, type, domaintype):
        if domaintype == "timeseries":
            domaintype = "PointSeries"
        elif domaintype == "trajectory":
            domaintype = "path"
        feature = self._feature_factory(domaintype.lower(), "encoder")
        return feature(self.conf, domaintype)

    def decode(self, covjson):
        requesttype = covjson["domainType"]
        if requesttype == "timeseries":
            requesttype = "PointSeries"
        elif requesttype == "MultiPoint":
            requesttype = "boundingbox"
        elif requesttype == "Trajectory":
            requesttype = "path"
        feature = self._feature_factory(requesttype.lower(), "decoder")
        return feature(covjson)

    def _feature_factory(self, feature_type, encoder_decoder):
        if encoder_decoder == "encoder":
            features = features_encoder
        elif encoder_decoder == "decoder":
            features = features_decoder
        return features[feature_type]
