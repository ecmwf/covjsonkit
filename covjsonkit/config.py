from typing import Literal

from conflator import ConfigModel


class CovjsonKitConfig(ConfigModel):
    param_db: str = "ecmwf"
    compression: Literal["zstd", "lz4", "binpack", None] = None
