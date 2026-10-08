# Changes on `feat/streaming-encoder`

The legacy encoder API (`Covjsonkit().encode(...)`, `from_polytope*`, decoders, `param_db`) is
unchanged; all existing tests pass.

## New: `covjsonkit.stream.CovjsonStreamEncoder`

A CoverageJSON encoder over the block stream polytope-mars emits (`polytope_mars.blocks`):
`begin(header) -> bytes`, `encode_iter(block) -> Iterator[bytes]`, `encode(block) -> bytes`,
`end() -> bytes`, `content_type = "application/prs.coverage+json"`, `file_extension = "covjson"`.
polytope-mars selects it for `format: covjson` (lazy import, `polytope_mars.encoders`).

- Blocks are read by attribute only; the module imports neither polytope-feature, polytope-mars nor
  covjson-pydantic (tested). Parameter metadata comes resolved in the header (`ParamInfo`), so the encoder does
  no `param_db` lookups.
- **Fragments:** `encode_iter(block)` yields the block one slice of its arrays at a time, each fragment at most
  `max_fragment_bytes` (constructor argument or `encoders.covjson.max_fragment_bytes`, default 8 MiB), so the
  text of a whole block never exists at once: a whole-world O2560 coverage is ~800 MB of coordinates and
  ~470 MB of values, and encoding a 5M-point block costs ~40 MB of peak RSS whatever its size
  (`tests/stream_memory_probe.py`).  The fragments of a block must be consumed completely and in order before
  the next block is encoded.  `encode(block)` returns `b"".join(encode_iter(block))` for buffered callers;
  both give the same bytes.  Structural fragments (a coverage's `mars:metadata` and `t`, the collection's
  `referencing` and `parameters`) are written whole - they are a few hundred bytes.
- **MultiPoint** coverages are written as blocks arrive: coverage opening + `mars:metadata` + `t` on the first
  coordinates block, `[lat, lon, level]` tuples (levels outer, points inner), each range opened on its first
  values block and streamed band by band, closed on `GroupEnd`. Memory per coverage: one fragment, plus the
  coordinates when the coverage has several levels (they are written once per level).
- **PointSeries, VerticalProfile, Trajectory** collections are buffered and written by `end()` (point features
  are small and the legacy layout is point-major across field groups). The PointSeries layout is the single
  function `pointseries_coverages` (to port covjsonkit `develop` #137 there); consecutive groups with equal
  `mars_metadata` and `levels` form one series (one coverage per point and level).
- `referencing` and the composite `coordinates` reproduce what each legacy encoder method wrote
  (`legacy_referencing`: `latitude/longitude/levelist`, `x/y/z` or `t/x/y/z` by feature and time-axis role).
- **Bytes:** identical to `json.dumps(legacy_covjson).encode()` (separators `", "`/`": "`, ASCII escapes,
  key order `type, domainType, coverages, referencing, parameters`). Text with strings in it (metadata,
  parameters, referencing) uses `json.dumps`; numeric arrays use `orjson` with `OPT_SERIALIZE_NUMPY` per
  slice, adjusted to `json.dumps` separators with `bytes.replace`. orjson and CPython agree on the
  shortest round-trip digits; they differ only in how numbers with 0 < |x| < 1e-4 are spelled
  (`1e-5` vs `1e-05`, `0.00002` vs `2e-05`), and arrays containing such values are formatted per value with
  `repr`. `-0.0`, integer-valued floats and `1e+16`-style exponents already match (tested on 5,000 random
  magnitudes from 1e-12 to 1e20).

## Output differences against the legacy encoders

- `NaN`/`inf` values are written as `null` (legacy `json.dumps` wrote the invalid JSON token `NaN`).
- A param absent from a field group (gribjump had no message) has no range in that coverage; a group with no
  param at all produces no coverage, as polytope-mars reports it. The collection's `parameters`
  always lists every parameter of the header.
- Everything else (including the legacy quirks: space-separated datetimes on the `_step` path, the
  `referencing` variants, int `realization`, efcl without `Forecast date`) is byte-identical; see the
  polytope-mars golden corpus and its `CHANGES.md`.

## Tests

`tests/test_stream_encoder.py` (32 tests, synthetic blocks, no polytope): legacy-identical MultiPoint bytes,
band-size invariance (1, 2, 3 and 13 bands, with levels), fragment joins equal to the single-fragment and
legacy bytes at 64 B / 1 KiB / 8 MiB fragment limits, the fragment limit itself, `null` for NaN and omitted
ranges, float and composite formatting against `json.dumps` (fast and per-value paths), referencing quirks,
PointSeries, VerticalProfile and Trajectory layouts, no polytope imports.

`tests/test_stream_memory.py` runs `tests/stream_memory_probe.py` in a subprocess: a 5M-point
`CoordsBlock` + `ValuesBlock` produce 318 MB of CoverageJSON in 54 fragments of at most 8 MiB for ~41 MB of
peak RSS above the blocks (1M points: ~30 MB; 20M points: ~49 MB), asserted below 100 MB and against any
growth with the block size.
