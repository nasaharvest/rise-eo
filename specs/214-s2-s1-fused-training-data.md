---
status: current
summary: A new model takes Sentinel-2 L2A (all bands) and Sentinel-1 RTC (VV, VH) as one input, as 12 calendar-month median composites over one year, on S2's 10 m B08 grid, with each label polygon on its own one-year window. Add calendar-month windows (`mosaic_days="month"`), per-polygon windows in `create_training_data` (every polygon must give the same T), an explicit opt-in to median both S1 orbit states together, a `CollectionInput` so one call can download, build and infer over several collections, and a fused datacube that builds each collection onto the first collection's grid and joins them along the band axis, at training and at inference. Cube paths become readable (collection, window, bands) with a short hash kept as the identity. Real runs are on Azure ML only. Closes #99.
issue: "#214"
---

# Spec 214 — fused Sentinel-2 + Sentinel-1 datacubes, monthly windows, per-polygon dates

**Status:** DRAFT, waiting for the user's sign-off.

## Problem

A model whose adapter we need to write takes, per pixel:

- every Sentinel-2 L2A spectral band (B01–B12 and B8A; L2A has no B10) and Sentinel-1 RTC `vv` and `vh`;
- one median composite per calendar month (January, February, …) over one year: T = 12;
- all bands on S2's 10 m B08 grid.

Its training labels come from many years, so each label polygon has its own one-year window.

rise-eo cannot produce this today, for five reasons:

1. **Windows are fixed-length days** (`datacube/ops.py` `_fixed_windows`). A one-year window with
   `mosaic_days=30` gives T = ceil(365/30) = 13, and the windows drift off the month boundaries.
2. **One window for every polygon.** `create_training_data` takes one `startdate`/`enddate`, and
   `flatten` refuses cubes whose `timestamps` differ (`datacube/flatten.py`
   `_check_metadata_consistency`), even when their T is the same.
3. **S1 builds raise on mixed orbit states.** `sentinel-1-rtc` declares
   `mosaic_partition=("sat:orbit_state",)` with `partition_policy="raise"` (spec 58 D9), and there is
   no way to say "median both on purpose".
4. **One collection per call.** `download`, `create_training_data` and `run_inference` take one
   `collection=` and one `bands=`.
5. **One cube, one collection** (spec 58 D16). Each build resamples to its own reference image. S1
   RTC is orthorectified to UTM but "not shifted to any specific grid" (MPC collection description),
   so an S1 cube and an S2 cube over the same cell do not share pixels, and cannot be joined.

## Decision

### D1 — Calendar-month windows: `mosaic_days="month"`

Every parameter named `mosaic_days` (`create_training_data`, `run_inference`, `build_datacube`,
`compute_n_timestamps`, the workflow and task CLIs) accepts an `int` (unchanged) or the string
`"month"`.

- Windows are calendar months, `[first of month, first of next month)`, in UTC.
- `startdate` must be the first of a month at 00:00 UTC, or the call raises before any work.
- T = the number of month starts `m` with `startdate <= m < enddate`. So 2018-01-01 → 2019-01-01 and
  2018-01-01 → 2018-12-31 both give T = 12. The last window stays upper-inclusive, as for `int`.
- Each window's label is its month start. An empty month is an all-nodata slice, as an empty
  `int` window is today.
- The cube path's window segment writes `mmonth` where it writes `m<days>` today.

No other strings are accepted. Weeks, dekads and seasons wait for a model that needs them.

### D2 — Per-polygon windows in `create_training_data`

`startdate` and `enddate` may both be `None`. Then `label_polygons` must have `startdate` and
`enddate` columns, read as UTC like the scalar arguments.

- Passing scalar dates when those columns exist raises (which one wins would be a guess). An empty
  or unparsable date in any row raises, naming the ids.
- **Every polygon must give the same T** for the given `mosaic_days`, or the call raises before any
  work, listing the ids for each T found. With an `adapter`, that T must also equal
  `adapter.n_timestamps` (the existing check).
- The download leg runs one `download` per distinct window, each with that window's polygons as ROI.
- The build leg already writes `startdate`/`enddate` per `input.csv` row, and each window has its own
  path segment, so it needs no change beyond being called once per distinct window.
- `flatten` checks that every cube has the same **number** of timestamps and the same `mosaic_days`,
  not the same timestamp values. It writes **`window_starts.npy`**, one UTC window start per sample,
  in the order of `ids.npy`. It writes it for scalar windows too, so the output has one shape.
  `TrainingData` gains `window_starts`.

### D3 — Opt in to median both S1 orbit states

In `_enforce_mosaic_partition`, a partition key that `properties_filter` lists with **more than one
value** is exempt from the single-value check. The caller has named the values to mix:

```python
properties_filter={"sat:orbit_state": ["ascending", "descending"]}
```

- Key absent, or one value: unchanged. Mixed rows still raise with the enumeration.
- `properties_filter` is already part of `params_key`, so a mixed-orbit cube never shares a path
  with a single-orbit cube.
- The docstrings of every verb that takes `properties_filter` say what this costs. Ascending and
  descending passes view a field from different sides and incidence angles, so their median mixes
  two geometries. CropHarvest, sits and WorldCereal each keep one direction. This amends spec 58 D9,
  which stays the default.

### D4 — `CollectionInput`: several collections in one call

A frozen dataclass, exported as `rise_eo.CollectionInput`:

```python
CollectionInput(
    collection: str,
    bands: list[str],
    *,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,       # build: as today; download: None means "latest"
    max_cloudcover: float | None = None, # download only; capability-gated as today
)
```

`download`, `create_training_data` and `run_inference` (ROI mode) gain `inputs: Sequence[CollectionInput] | None`.

- `inputs` excludes `collection`, `bands`, `properties_filter`, `processing` and `max_cloudcover`:
  passing both raises. Pass `bands=None` positionally.
- With `inputs`, the verbs that read a catalog take **`archive_root=`** (the `dst_folderpath` that
  `download` wrote) instead of `catalog_filepath`. Each catalog is
  `{archive_root}/{collection}/catalog.parquet`, the spec 59 layout.
- `download(..., inputs=[...])` runs one download per input, in order, with the shared `roi`, window,
  `source`, `max_tiles` (per collection) and runner settings. It returns
  `{collection: catalog_filepath}`. Every input's preflight runs before any download starts, so an
  S1 input with `max_cloudcover` fails before the S2 download is paid for.
- A collection may appear only once in `inputs`, and band names must be unique across inputs.
  Either violation raises.
- `inputs` of length 1 behaves exactly like the scalar arguments.

### D5 — The fused datacube

When `inputs` has more than one entry, each per-polygon or per-cell task builds a **fused cube**:

1. **Build the first input's cube as today.** Its collection must declare a `reference_band`
   (S2 → B08), or preflight raises. That cube's grid (CRS, transform, height, width) is the
   **reference grid**.
2. **Build every other input onto that grid.** `build_datacube` gains
   `reference_profile=`, which replaces the profile it would compute from its own reference band. The
   S1 cube is resampled onto S2's B08 grid with the builder's existing resampler (nearest neighbour).
   The S1 cube keeps its own declaration, nodata and (absent) mask.
3. **Fix the resample test.** `_get_indices_to_resample` also compares the **transform**, so an image
   with the reference's shape but a shifted origin is resampled, not stacked out of place.
4. **Join** (`datacube.join_cubes`): assert identical `timestamps` labels and height/width, then
   concatenate along the band axis in `inputs` order. The fused cube is `float32`. Each component's
   nodata becomes 0, the fused cube's single nodata (S1 RTC declares −32768).
5. **Paths** follow D7. Component cubes keep their own paths, so a rerun resumes them. A component
   built onto another collection's grid gets `_on-<collection>` in its path and `grid_from` in its
   hash, so it never collides with a standalone cube. The fused cube's metadata and `params.json`
   list each component's collection, path and provenance.

`flatten` and `feature_sequence` work on the fused cube unchanged: it is an ordinary
`datacube.npy` + `metadata.pickle.npy`. With D1 and calendar windows on both inputs, the two
`timestamps` axes are identical by construction (ADR 0010).

### D6 — Inference builds the same fused cube

`run_inference(roi=..., inputs=[...], archive_root=...)` sends the per-cell task down the same
build-and-join function as `create_training_data`, so training and inference features cannot
diverge (ROADMAP F1). Preflight checks that `adapter.required_bands` is a subset of the fused band
list. The adapter contract does not change: `required_bands` names fused bands (`"B04"`, `"vv"`, …).

### D7 — Readable cube paths: name for people, hash for identity

Today a cube lives at `{run}/<start>_<end>_m<days>_<hash8>/{id}/`, and bands, collection,
declaration, `properties_filter` and `processing` hide inside the 8-character hash. Every cube
path, single-collection or fused, becomes:

```
{run}/{collection}/{start}_{end}_m{mosaic}/{bands}_{hash8}/{id}/
```

| part | single collection | fused (D5) |
|---|---|---|
| `{collection}` | `sentinel-2-l2a` | `sentinel-2-l2a__sentinel-1-rtc` |
| window | `20180101_20190101_mmonth` | same |
| `{bands}` | `B01-B02-B03-B04-B05-B06-B07-B08-B8A-B09-B11-B12-SCL` | `B01-…-SCL__vv-vh` |
| S1 component on S2's grid | `sentinel-1-rtc/…/vv-vh_on-sentinel-2-l2a_<hash8>/` | — |

- Bands appear in request order, joined by `-`. Collections are joined by `__`. Only
  `[A-Za-z0-9._-]` may appear in a readable part; a band or collection name with any other character
  raises. These characters need no escaping in a blob URL or on Windows.
- **`{hash8}` stays the identity.** It digests every parameter, readable or not (bands, collection,
  declaration digest, `properties_filter`, `processing`, `grid_from`). The readable parts are for
  people; the hash alone keeps two different requests from sharing a path.
- **`params.json`** sits in each `{bands}_{hash8}/` folder and writes out every parameter in plain
  form, including the ones only the hash carries (`properties_filter`, `processing`, the
  declaration) and, for a fused cube, its components' paths.
- The run's membership never enters the path: adding a polygon or cell adds one `{id}/` folder and
  moves nothing else.
- On Windows, a local cube path longer than 259 characters raises at preflight, naming the path and
  the `LongPathsEnabled` setting. Without that check, the failure would surface later as a confusing
  file-not-found error.
- **This moves every existing cube path once.** Cubes built before it are rebuilt on the next run.
  There is no shim (repo rule).

## Prior art

- **D1:** openEO `aggregate_temporal_period` (openeo-processes 1.0.0, 2020-07-31) has
  `period="month"`, labels `YYYY-01` … `YYYY-12`, and outputs an empty period as no-data. That is
  the window, label and empty-window behaviour D1 adopts. CropHarvest uses fixed 30-day steps
  instead (`DAYS_PER_TIMESTEP = 30`, `config.py`). The user asked for calendar months.
- **D2:** CropHarvest (NASA Harvest, NeurIPS 2021) exports each label over its own window: custom
  labels carry `start_date`/`end_date` columns, and `export_for_labels` asserts they exist
  (`cropharvest/eo/eo.py`). The default is 12 × 30 days ending 1 February of the label's year. The
  "same T or raise" rule is homemade: a fixed-width model input needs it, and nothing we found
  states it as a rule.
- **D3:** prior art is against this, and the spec says so. CropHarvest keeps only the first image's
  pass direction ("we want the coverage to be consistent (so don't want to take both)",
  `cropharvest/eo/sentinel1.py`). sits' S1+S2 example uses `orbit = "descending"`. WorldCereal picks
  one direction (from a summary of the ESSD paper, not checked against its text). For mixing:
  CDSE's Sentinel-1 monthly mosaics (2024, after the cutoff) average "several viewing geometries"
  within a month. The user chose mixing, so D3 makes it an explicit opt-in and keeps the raise as
  the default.
- **D4:** STAC API item search takes `collections` as a list, matched as OR. That is a single request
  over several collections, which `inputs` mirrors at the verb level. The `CollectionInput` type
  itself is homemade: we found no pre-2022 library whose download call bundles per-collection
  bands, filters and cloud cover in one value.
- **D5, D6:** openEO `resample_cube_spatial` ("Resample the spatial dimensions to match a target
  data cube", default `near`), then `merge_cubes` (cubes with different bands merge with no overlap
  resolver). Both are in openeo-processes 1.0.0 (2020). sits does the same: regularize each cube to
  the same grid and period (`sits_regularize(..., period = "P16D", res = 40)`), then
  `sits_merge(cube_s2_reg, cube_s1_reg)`. Resampling *to a real reference image* (B08) rather than an
  abstract grid is the repo's existing rule.
- **D7:** no single standard names cached artifacts. The closest established pattern is the Nix
  store path, whose base name pairs "a 20-byte digest for identification" with "a symbolic name for
  people to read" (`/nix/store/q06x…-firefox-33.1`). D7 copies that split, with the readable part
  first so paths sort by collection and window. Hydra's per-run `.hydra/config.yaml` ("a dump of the
  user specified configuration") is the precedent for `params.json`. Hive-style `key=value` folders
  (`/year=2009/month=11/`, the layout Apache Arrow reads as partitions) were rejected: no query engine
  reads these paths, and the keys would roughly double their length against Windows' 260-character
  `MAX_PATH`.

## How to verify

Tests (synthetic, in `tests/`):

- **AC1:** `compute_n_timestamps` gives 12 for 2018-01-01 → 2019-01-01 and for → 2018-12-31 with
  `"month"`. February's window ends on 1 March. `startdate=2018-01-15` raises. A month with no
  granules is an all-nodata slice.
- **AC2:** two polygons with windows in 2018 and 2019 give one training set with T = 12 each and the
  right `window_starts.npy`. A full-year polygon plus a six-month polygon raises, naming both ids
  under their T. Scalar dates plus date columns raises.
- **AC3:** a catalog with both orbit states raises with no filter and with one value, builds with
  both values listed, and the two selections land at different paths.
- **AC4:** `download(inputs=[s2, s1])` writes two catalogs and returns both paths. An S1 input with
  `max_cloudcover` fails at preflight before the S2 download runs. A repeated collection or band name
  raises.
- **AC5:** an image with the reference's shape and a shifted transform is resampled. The fused
  cube's bands are in `inputs` order, its nodata is 0, and mismatched timestamps raise.
- **AC6:** for one cell geometry, the training path and the inference path produce byte-identical
  fused cubes.

- **AC9 (D7):** the path of a fused cube reads
  `sentinel-2-l2a__sentinel-1-rtc/20180101_20190101_mmonth/B01-…-SCL__vv-vh_<hash8>/<id>/`. Two
  requests that differ only in `properties_filter` share the readable part and get different
  hashes. `params.json` round-trips every parameter. A band name containing `/` raises. On
  Windows, a 300-character local path raises at preflight.

Real runs (gate 4) run **on Azure ML only**, in one P5 run-book. The archive is on blob, and the
downloads and builds run as AML jobs (`runner="aml"`). Only the flattened arrays and one fused cube
(tens of MB) land on the laptop. No S1 scene is ever downloaded locally.

- **AC7:** `s2grid=4772924`, 2018-06-01 → 2018-07-01, `mosaic_days="month"` (T = 1).
  `download(inputs=[s2, s1])` into a blob archive: S2 with all bands plus SCL, S1 with `vv`/`vh` and
  both orbit states. Then `create_training_data(inputs=[s2, s1], runner="aml")` over the
  `AT_2018_TRAIN` fields inside that cell. Asserts: one fused cube's shape is `(1, H, W, 14)` (12 S2
  bands, since SCL is dropped after masking, plus `vv` and `vh`), its dtype is `float32` and its
  nodata is 0, and the count of valid S1 pixels equal to 0.0 is printed. The run-book writes that
  cube's `B08` and `vv` as GeoTIFFs, and in QGIS the user checks that field edges in `vv` line up
  with the edges in `B08`.
- **AC8:** the same run-book, full year: 2018-01-01 → 2019-01-01, T = 12, on the same fields. This
  is the first training set the model can train on. It also runs `run_inference(inputs=[s2, s1])`
  over the cell with a stub adapter and asserts that the inference cube for a training field's
  pixels matches the training cube (D6).

## Out of scope

- **Windowed S1 reads.** S1 still downloads whole ~250 km scenes (see Risks). File as an issue.
- An adapter declaring its own `inputs`, or the bundle recording which inputs it was trained on (a
  #101-style guard). File as an issue.
- Fusion with HLS, speckle filtering, dB conversion (a `feature_sequence` step can do that),
  `sat:relative_orbit` selection, and S1 from CDSE.
- Other calendar periods (week, dekad, season).

## Phases

One PR per phase. P1–P3 do not depend on each other. P5 needs P4.

| Phase | Decisions | Size |
|---|---|---|
| P1 | D3 orbit opt-in | ~1 h |
| P2 | D1 monthly windows, D7 readable cube paths | ~1 day |
| P3 | D2 per-polygon windows | half a day to a day |
| P4 | D4 `CollectionInput`, multi-collection download | ~half a day |
| P5 | D5 + D6 fused cube, training and inference; AC7–AC8 run-book on Azure ML | 1–2 days |

Each implementation PR updates `README.md`, `ARCHITECTURE.md`, `CONTEXT.md`, `ROADMAP.md` and
`AGENTS.md`'s "downloads Sentinel-2 L2A imagery" line, and P1 adds an amendment note to spec 58 D9.

## Risks

- **S1 download size.** Measured 2026-09-12 over one grid cell, April–September 2018, both orbits:
  387.7 GB (~3.7 GB per scene). A year is roughly twice that per location. That is why every real
  run for this spec is on Azure ML (blob storage, in-region transfer), and why no run-book downloads
  S1 to a laptop. Windowed reads would cut it to the bytes the cell needs, but are out of scope.
- **Mixed orbit geometry (D3).** Field-level backscatter differs by look direction, so a model
  trained on mixed medians must also be served mixed medians. D6 makes that automatic; nothing
  checks it if a user changes `properties_filter` between training and inference.
- **A valid S1 pixel of exactly 0.0 becomes nodata** in the fused cube. Linear gamma naught is not
  physically 0 for valid backscatter, so we expect this to be rare. AC7 counts such pixels in the
  real cube.
- **Cube size.** `float32` doubles S2's bytes. A 5 km inference cell is about
  500 × 500 px × 12 T × 14 bands × 4 B ≈ 168 MB, similar to today's largest cubes.

## Sources

- openEO processes 1.2.0 and 1.0.0 JSON, `github.com/Open-EO/openeo-processes`
  (`aggregate_temporal_period.json`, `resample_cube_spatial.json`, `merge_cubes.json`) and its
  releases page: the monthly period, its labels and empty-period no-data (D1); resample to a target
  cube with `near` by default, and merging cubes with different bands without a resolver (D5); the
  1.0.0 date, 2020-07-31.
- `github.com/nasaharvest/cropharvest`, `cropharvest/eo/eo.py`, `cropharvest/eo/sentinel1.py`,
  `cropharvest/config.py`: per-label `start_date`/`end_date` (D2); 30-day steps, not months (D1);
  one orbit direction on purpose (D3); `scale=10` exports (D5).
- sits book, "Merging multi-source EO data cubes" (`e-sensing.github.io/sitsbook/dc_merge.html`):
  regularize S1 RTC and S2 to one grid and period, then `sits_merge` (D5); a descending-only S1
  example (D3).
- Copernicus Data Space, "Announcing Sentinel-1 Monthly Mosaics" (2024): monthly composites that
  average several viewing geometries (D3, the case for mixing).
- WorldCereal, ESSD 15, 5491–5515 (2023): one orbit direction (D3). Taken from a summary of the
  paper, not its text, so it is a lead, not evidence.
- MPC STAC collections `sentinel-2-l2a` and `sentinel-1-rtc`: L2A's 12 spectral bands at
  10/20/60 m with no B10 (Problem); RTC `vv`/`vh` as float32, nodata −32768, 10 m, orthorectified to
  UTM and "not shifted to any specific grid" (Problem 5, D5).
- STAC API item search (`collections` as a list, OR-matched), as described in NASA CMR's STAC OpenAPI
  docs and Element 84's Earth Search docs (D4).
- Spec 58 D9, D11, D16 and its S1 download measurement (Problem 3, 5; Risks).
- Nix manual 2.36, "Store Path" (`nix.dev/manual/nix/2.36/store/store-path`): a base name is a
  digest "for identification" plus a name "for people to read" (D7).
- Hydra docs, "Output/Working directory" (`hydra.cc/docs/tutorials/basic/running_your_app/working_directory/`):
  each run's `.hydra/config.yaml` snapshot (D7's `params.json`).
- Apache Arrow, "Tabular Datasets" (`arrow.apache.org/docs/python/dataset.html`): Hive-style
  `/key=value/` partition folders, the rejected alternative (D7). Taken from a summary of the page.
- Microsoft, "Maximum Path Length Limitation": `MAX_PATH` is 260 characters unless long paths are
  enabled (D7's Windows check). Microsoft, "Naming and Referencing Containers, Blobs, and Metadata":
  blob names up to 1,024 characters, and reserved URL characters must be escaped (D7's character
  set).

Searches run for this spec: openEO resample/merge and monthly aggregation; combining S1 ascending and
descending in monthly composites for crop classification; STAC multi-collection item search;
CropHarvest's export code.
