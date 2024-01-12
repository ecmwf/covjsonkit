import json

import eccovjson.decoder.TimeSeries
import eccovjson.decoder.VerticalProfile
import eccovjson.encoder.TimeSeries
import eccovjson.encoder.VerticalProfile

features_encoder = {
    "pointseries": eccovjson.encoder.TimeSeries.TimeSeries,
    "verticalprofile": eccovjson.encoder.VerticalProfile.VerticalProfile,
}
features_decoder = {
    "pointseries": eccovjson.decoder.TimeSeries.TimeSeries,
    "verticalprofile": eccovjson.decoder.VerticalProfile.VerticalProfile,
}


class Eccovjson:
    def __init__(self):
        pass

    def encode(self, type, domaintype):
        if domaintype == "timeseries":
            domaintype = "PointSeries"
        feature = self._feature_factory(domaintype.lower(), "encoder")
        # if requesttype == "timeseries":
        #    encoder_obj = eccovjson.encoder.TimeSeries.TimeSeries(type, domaintype)
        # elif requesttype == "VerticalProfile":
        #    encoder_obj = eccovjson.encoder.VerticalProfile.VerticalProfile(
        #        type, domaintype
        #    )
        return feature(type, domaintype)

    def decode(self, covjson, requesttype):
        if requesttype == "timeseries":
            requesttype = "PointSeries"
        feature = self._feature_factory(requesttype.lower(), "decoder")
        # if requesttype == "timeseries":
        #    decoder_obj = eccovjson.decoder.TimeSeries.TimeSeries(covjson)
        # elif requesttype == "VerticalProfile":
        #    decoder_obj = eccovjson.decoder.VerticalProfile.VerticalProfile(covjson)
        return feature(covjson)

    def _feature_factory(self, feature_type, encoder_decoder):
        if encoder_decoder == "encoder":
            features = features_encoder
        elif encoder_decoder == "decoder":
            features = features_decoder
        return features[feature_type]
