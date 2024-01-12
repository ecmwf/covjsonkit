import json

import eccovjson.decoder.TimeSeries
import eccovjson.decoder.VerticalProfile
import eccovjson.encoder.TimeSeries
import eccovjson.encoder.VerticalProfile


class Eccovjson:
    def __init__(self):
        # Initialise polytope
        pass

    def encode(self, type, domaintype, requesttype):
        if requesttype == "timeseries":
            encoder_obj = eccovjson.encoder.TimeSeries.TimeSeries(type, domaintype)
        elif requesttype == "VerticalProfile":
            encoder_obj = eccovjson.encoder.VerticalProfile.VerticalProfile(
                type, domaintype
            )
        return encoder_obj

    def decode(self, covjson, requesttype):
        if requesttype == "timeseries":
            decoder_obj = eccovjson.decoder.TimeSeries.TimeSeries(covjson)
        elif requesttype == "VerticalProfile":
            decoder_obj = eccovjson.decoder.VerticalProfile.VerticalProfile(covjson)
        return decoder_obj
