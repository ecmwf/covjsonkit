# Changes on `feat/streaming-encoder`

Per-branch note for the PR description. The legacy encoder API (`Covjsonkit().encode(...)`,
`from_polytope*`, decoders, `param_db`) is unchanged; all existing tests pass.

## New: `covjsonkit.stream.CovjsonStreamEncoder`

A CoverageJSON encoder over the block stream polytope-mars emits (`polytope_mars.blocks`, DESIGN §3):
`begin(header) -> bytes`, `encode(block) -> bytes`, `end() -> bytes`, `content_type =
"application/prs.coverage+json"`, `file_extension = "covjson"`. polytope-mars selects it for `format: covjson`
(lazy import, `polytope_mars.encoders`).

- Blocks are read by attribute only; the module imports neither polytope-feature, polytope-mars nor
  covjson-pydantic (tested). Parameter metadata comes resolved in the header (`ParamInfo`), so the encoder does
  no `param_db` lookups.
- **MultiPoint** coverages are written as blocks arrive: coverage opening + `mars:metadata` + `t` on the first
  coordinates block, `[lat, lon, level]` tuples (levels outer, points inner), each range opened on its first
  values block and streamed band by band, closed on `GroupEnd`. Memory per coverage: one block, plus the
  coordinates when the coverage has several levels (they are written once per level).
- **PointSeries, VerticalProfile, Trajectory** collections are buffered and written by `end()` (point features
  are small and the legacy layout is point-major across field groups). The PointSeries layout is the single
  function `pointseries_coverages` (to port covjsonkit `develop` #137 there); consecutive groups with equal
  `mars_metadata` and `levels` form one series (one coverage per point and level).
- `referencing` and the composite `coordinates` reproduce what each legacy encoder method wrote
  (`legacy_referencing`: `latitude/longitude/levelist`, `x/y/z` or `t/x/y/z` by feature and time-axis role).
- **Bytes:** identical to `json.dumps(legacy_covjson).encode()` (separators `", "`/`": "`, ASCII escapes,
  key order `type, domainType, coverages, referencing, parameters`). Fragments with strings (metadata,
  parameters, referencing) use `json.dumps`; numeric arrays use `orjson` with `OPT_SERIALIZE_NUMPY` over the
  whole array, adjusted to `json.dumps` separators with `bytes.replace`. orjson and CPython agree on the
  shortest round-trip digits; they differ only in how numbers with 0 < |x| < 1e-4 are spelled
  (`1e-5` vs `1e-05`, `0.00002` vs `2e-05`), and arrays containing such values are formatted per value with
  `repr`. `-0.0`, integer-valued floats and `1e+16`-style exponents already match (tested on 5,000 random
  magnitudes from 1e-12 to 1e20).

## Output differences against the legacy encoders

- `NaN`/`inf` values are written as `null` (legacy `json.dumps` wrote the invalid JSON token `NaN`).
- A param absent from a field group (gribjump had no message) has no range in that coverage; a group with no
  param at all produces no coverage (decided by polytope-mars, DESIGN §2.5). The collection's `parameters`
  always lists every parameter of the header.
- Everything else (including the legacy quirks: space-separated datetimes on the `_step` path, the
  `referencing` variants, int `realization`, efcl without `Forecast date`) is byte-identical; see the
  polytope-mars golden corpus and its `CHANGES.md`.

## Tests

`tests/test_stream_encoder.py` (24 tests, synthetic blocks, no polytope): legacy-identical MultiPoint bytes,
band-size invariance (1, 2, 3 and 13 bands, with levels), `null` for NaN and omitted ranges, float and
composite formatting against `json.dumps` (fast and per-value paths), referencing quirks, PointSeries,
VerticalProfile and Trajectory layouts, no polytope imports.
