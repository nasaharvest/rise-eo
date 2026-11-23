---
status: current
summary: A new model takes Sentinel-2 L2A (all bands) and Sentinel-1 RTC (VV, VH) as one input, as 12 calendar-month median composites over one year, on S2's 10 m B08 grid, with each label polygon on its own one-year window. Add calendar-month windows (`mosaic_days="month"`), per-polygon windows in `create_training_data` (every polygon must give the same T), an explicit opt-in to median both S1 orbit states together, a `CollectionInput` so one call can download, build and infer over several collections, and a fused datacube that builds each collection onto the first collection's grid and joins them along the band axis, at training and at inference. Cube paths become short and readable (collection codes, window, ISO 8601 period), with a hash kept as the identity and the full parameters in `params.json`. Every training set and inference output gets a `lineage.json` (its sources and process steps), and a model bundle carries its training lineage so inference refuses inputs the model was not trained on. Real runs are on Azure ML only. Closes #99.
issue: "#214"
---

# Spec 214 — fused Sentinel-2 + Sentinel-1 datacubes, monthly windows, per-polygon dates

**Status:** **SIGNED OFF (user, 2026-11-23).**
**Amendment A1 (2026-11-23, from the gate-3 review on #215), awaiting the user's sign-off.** No decision
changes direction. A1:
- adds ADR 0035 (supersedes ADR 0029 in part) and ADR 0036 (supersedes ADR 0010 in part);
- D1: the period is named as in D7, and `"month"` needs the calendar mosaic scheme;
- D2: states the flatten skip identity per window, and drops the `mosaic_days` comparison;
- D4: corrects the length-1 rule;
- D5: the join also checks CRS and transform, and defines `grid_from` and the fused dtype;
- D7: defines the `SHORT_NAME` fallback;
- D8: states how paths are recorded;
- the ACs now match the phases, and the S2 band count is worded consistently.

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
- (A1) The cube path names the period as D7 does: `P1M` for `"month"`, `P<n>D` for an `int`.
- (A1) `"month"` needs `mosaic_scheme="calendar"`, the default. With `mosaic_scheme="acquisition"`,
  whose windows depend on the data, it raises before any work.

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
- (A1) `flatten` checks that every cube has the same **number** of timestamps and the same bands,
  not the same timestamp values. `mosaic_days` is one argument for the whole call, so it cannot
  differ between cubes and is not compared. `flatten` writes **`window_starts.npy`**, one UTC window
  start per sample, in the order of `ids.npy`. It writes it for scalar windows too, so the output
  has one shape. `TrainingData` gains `window_starts`. ADR 0036 records the change to ADR 0010.
- (A1) **Skip identity.** The flatten skip identity (`_flatten_identity_from_request`) and the
  known-empty manifest are computed once per distinct window, each with its own window segment, and
  joined. One polygon's window change moves only that polygon's cube.
- (A1) P3 changes the `AGENTS.md` line "Cubes with the same start, end and `mosaic_days` share the
  same `timestamps` axis; `flatten` needs that" to say that `flatten` needs the same T.

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
  which stays the default. (A1) ADR 0035 records the change to ADR 0029.

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
- (A1) With one input, `create_training_data` and `run_inference` build and infer exactly as with
  the scalar arguments. `download(inputs=...)` always returns the dict, even with one input.

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
4. **Join** (`datacube.join_cubes`): (A1) assert identical `timestamps` labels, CRS, transform,
   height and width, so a stale component on a shifted grid raises instead of joining. Then
   concatenate along the band axis in `inputs` order. The fused cube is `float32`, in the array and
   in every dtype field of its metadata (including `geotiff_metadata`, which says `uint16` for an S2
   cube today). Each component's nodata becomes 0, the fused cube's single nodata (S1 RTC declares
   −32768).
5. **Paths** follow D7. Component cubes keep their own paths, so a rerun resumes them. A component
   built onto another collection's grid gets `-on-<code>` in its path, and (A1) **`grid_from`**, the
   reference cube's full `params_key`, in its hash. It never collides with a standalone cube, and it
   moves whenever the reference cube's parameters change. The fused cube's metadata and
   `params.json` list each component's collection, path and provenance.
6. (A1) A fused build needs `mosaic_scheme="calendar"` on every input, or preflight raises: only
   calendar windows give two collections the same `timestamps` axis.

`flatten` and `feature_sequence` work on the fused cube unchanged: it is an ordinary
`datacube.npy` + `metadata.pickle.npy`. With D1 and calendar windows on both inputs, the two
`timestamps` axes are identical by construction (ADR 0010).

### D6 — Inference builds the same fused cube

`run_inference(roi=..., inputs=[...], archive_root=...)` sends the per-cell task down the same
build-and-join function as `create_training_data`, so training and inference features cannot
diverge (ROADMAP F1). Preflight checks that `adapter.required_bands` is a subset of the fused band
list. The adapter contract does not change: `required_bands` names fused bands (`"B04"`, `"vv"`, …).

### D7 — Short readable cube paths: name for people, hash for identity

Today a cube lives at `{run}/<start>_<end>_m<days>_<hash8>/{id}/`. The collection,
`properties_filter`, `processing` and declaration hide inside the 8-character hash. Every cube
path, single-collection or fused, becomes one short segment:

```
{run}/{collections}_{start}-{end}_{period}_{hash8}/{id}/
```

| cube | segment (characters) |
|---|---|
| S2, one month, 10-day mosaic | `s2l2a_20180601-20180701_P10D_ab12cd34` (37) |
| fused S2 + S1, one year, monthly | `s2l2a-s1rtc_20180101-20190101_P1M_3f9a1c2e` (42) |
| S1 component built on S2's grid | `s1rtc-on-s2l2a_20180101-20190101_P1M_9e8d7c6b` (45) |

Today's segment is 30 characters, so the new one adds about a dozen.

- **Collection codes.** Each collection module declares a `SHORT_NAME` next to its declaration
  (`s2l2a`, `s1rtc`). A caller's registered collection may pass `short_name=` to `register()`. (A1)
  When it passes none, the code is the collection id with every character outside `[a-z0-9]`
  removed (`sentinel-2-l2a` → `sentinel2l2a`, `hls2-s30` → `hls2s30`). Registering a code that
  another collection already uses raises, naming both collections. Fused cubes join codes with `-`
  in `inputs` order.
  A code must match `[a-z0-9]+`, or registration raises.
- **Dates** use ISO 8601's basic form, `YYYYMMDD`. **The period** is an ISO 8601 duration: `P1M` for
  `mosaic_days="month"`, `P<n>D` for an `int`.
- **Bands are not in the name.** The 13 S2 band names requested (12 spectral plus SCL) alone would
  take 51 characters. Bands are in the
  hash, so two band sets never share a path, and `params.json` lists them.
- **`{hash8}` stays the identity.** It digests every parameter, readable or not (bands, collection,
  declaration digest, `properties_filter`, `processing`, `grid_from`). The readable part is for
  people; the hash alone keeps two different requests from sharing a path.
- **`params.json`** sits in each segment folder and writes out every parameter in plain form: bands,
  `properties_filter`, `processing`, the declaration and, for a fused cube, its components' paths.
- The run's membership never enters the path: adding a polygon or cell adds one `{id}/` folder and
  moves nothing else.
- **This moves every existing cube path once.** Cubes built before it are rebuilt on the next run.
  There is no shim (repo rule).

### D8 — `lineage.json`: what a training set or an inference output went through

Cubes are temporary; the training arrays and the inference COGs are what people keep. Each gets
one readable JSON file saying where it came from and what was done to it:

- `create_training_data` writes `lineage.json` next to `data.npy`;
- `run_inference` writes it next to its STAC catalog.

Its shape follows ISO 19115's lineage model, sources plus ordered process steps:

```json
{
  "statement": "Training data for 900 polygons of AT_2018_TRAIN.geojson: sentinel-2-l2a (12 spectral bands, SCL mask) and sentinel-1-rtc (vv, vh; both orbit states), monthly median, T=12, fused on the sentinel-2-l2a B08 10 m grid.",
  "created": "2026-12-01T10:00:00Z",
  "software": {"rise-eo": "0.4.0", "numpy": "…", "rasterio": "…", "gdal": "…"},
  "runner": "aml",
  "sources": {
    "geometry": {"path": "…", "sha256": "…", "n_features": 900, "id_col": "fid", "label_col": "crop"},
    "collections": [
      {"collection": "sentinel-2-l2a", "bands": ["B01", "…"], "properties_filter": null,
       "processing": "latest", "provider": ["mpc"], "processing_version": ["05.09"],
       "n_granules": 212, "granule_ids": ["S2B_MSIL2A_…", "…"]},
      {"collection": "sentinel-1-rtc", "…": "…"}
    ]
  },
  "process_steps": [
    {"step": "download", "description": "…", "parameters": {"…": "…"}},
    {"step": "mask", "parameters": {"band": "SCL", "classes": ["…"]}},
    {"step": "resample", "parameters": {"method": "nearest", "reference": "sentinel-2-l2a B08"}},
    {"step": "mosaic", "parameters": {"method": "median", "period": "P1M", "windows": "per polygon"}},
    {"step": "fuse", "parameters": {"order": ["sentinel-2-l2a", "sentinel-1-rtc"], "dtype": "float32", "nodata": 0}},
    {"step": "flatten", "parameters": {"aggregate": null}},
    {"step": "features", "parameters": {"feature_sequence": ["module.function", "…"]}}
  ]
}
```

- Every step and parameter is taken from what actually ran: the call's arguments, each
  collection's declaration and the cubes' provenance (spec 59 D9). Nothing is typed in by hand.
- `statement` is one generated sentence for people. `granule_ids` lists every granule used,
  so a training set can be traced to its imagery.
- An inference `lineage.json` has the same shape. It adds the ROI and grid size to `sources`, the
  model (registry ref or bundle path, plus the bundle's digest) and, when the bundle has one, the
  model's **training lineage** (D9), and a `predict` step.
- The inference STAC Collection and Items get `processing:software` and `processing:lineage` (the
  statement), and the Collection links to `lineage.json` as an asset with role `metadata`.
- `lineage.json` only describes. The skip-or-redo decision stays with `_flatten_stamp.json`, which
  records identity, not description.

**Enough to recreate the data later.** A later feature (out of scope) should be able to rebuild a
deleted training set or inference output from `lineage.json` alone, provided the provider still
serves the granules. So the file also holds:

- **`lineage_version`** (starting at `1`), so a future reader knows which fields to expect. Readers
  keep a list of the versions they support (repo rule).
- **`request`**: the verb and every argument it was called with, as JSON, including the
  `CollectionInput`s and the per-polygon date columns. Two exceptions: credentials are never
  written, and `runner_kwargs` is reduced to the runner's name, because it holds private resource
  identifiers. The output location is re-supplied on replay. (A1) Paths (the geometry's
  source, `archive_root`) are recorded as given, blob URL or local path, because a replay on the
  same machine needs them. A local path can name a user's home folder, so treat `lineage.json` like
  any output before sharing it. A replay elsewhere uses the `geometry.geojson` copy and a new
  `archive_root`.
- **The geometry itself**: the label polygons or the ROI are copied next to `lineage.json` as
  `geometry.geojson`, with their sha256 recorded. A path alone can't recover a deleted or edited file.
  A 900-field training set's geometry is a few MB.
- **Code**: rise-eo's version and, for a git install, the commit and whether the tree had
  uncommitted changes. For each feature function, its `module.qualname` *and* a sha256 of its
  source, so an edited body shows. For inference, the bundle digest.
- **Libraries**: versions of Python, numpy, pandas, geopandas, rasterio, GDAL, pyproj, numba and
  fsspec.
- **Determinism caveat**: where MGRS tiles overlap, the pixel kept depends on download order
  (#203), so a rebuild can differ there until #203 is fixed. `lineage.json` lists the overlapping
  tiles used, so a reader knows which pixels that affects.

### D9 — The model bundle carries its training lineage

`save_bundle` gains `training_data=` (a `TrainingData` handle or its `export_folderpath`). It copies
that training set's `lineage.json` into the bundle.

At inference, preflight compares the bundle's training lineage with the inference request: the
collections in order, each one's bands and `properties_filter`, the period and T. A difference
raises, naming the field and both values. For example, a model trained on both orbit states but
served only `descending` fails before any job starts. A bundle saved without `training_data=` skips
the check and says so once.

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
  user specified configuration") is the precedent for `params.json`. The period follows ISO 8601
  durations, as sits writes them (`period = "P16D"`). Hive-style `key=value` folders
  (`/year=2009/month=11/`, the layout Apache Arrow reads as partitions) were rejected: no query engine
  reads these paths, and the keys would make them longer, the opposite of the goal.
- **D8:** ISO 19115's lineage model, "based on sources which are either used or produced in a
  series of process steps". Each step records a description, rationale, date and processor
  (`LI_Lineage` → `statement`, `source`, `processStep`; ESIP wiki, ISO_Lineage, edited 2017). D8
  keeps its three parts (`statement`, `sources`, `process_steps`) in plain JSON, not ISO's XML
  encoding. The STAC processing extension (v1.0.0, 2021-03-08) supplies `processing:software`
  ("name/version … for provenance purposes") and `processing:lineage` ("free text information
  about the how observations were processed") for the inference STAC. W3C PROV (2013) was rejected
  as the format: its entity/activity/agent graph answers questions nobody here asks, at the cost of
  an RDF vocabulary. `granule_ids` per source follows spec 59 D9's provenance, which rise-eo
  already computes.
- **D9:** homemade. The closest established practice is a model card's "training data" section
  (Mitchell et al., 2019), which is prose for people. D9 stores the same fact as data, so inference
  can compare against it.

## How to verify

Tests (synthetic, in `tests/`):

- **AC1:** `compute_n_timestamps` gives 12 for 2018-01-01 → 2019-01-01 and for → 2018-12-31 with
  `"month"`. February's window ends on 1 March. `startdate=2018-01-15` raises. A month with no
  granules is an all-nodata slice.
- **AC2:** two polygons with windows in 2018 and 2019 give one training set with T = 12 each and the
  right `window_starts.npy`. A full-year polygon plus a six-month polygon raises, naming both ids
  under their T. Scalar dates plus date columns raises.
- **AC3:** a catalog with both orbit states raises with no filter and with one value, builds with
  both values listed, and the two selections land at different paths. (A1) This is checked through
  the per-cube task the runners call (`workflows.task.run_task`), not only `build_datacube`, so the
  filter is shown to reach the node.
- **AC4:** `download(inputs=[s2, s1])` writes two catalogs and returns both paths. An S1 input with
  `max_cloudcover` fails at preflight before the S2 download runs. A repeated collection or band name
  raises.
- **AC5:** an image with the reference's shape and a shifted transform is resampled. The fused
  cube's bands are in `inputs` order, its nodata is 0, and it is `float32` in the array and in its
  metadata. (A1) Mismatched timestamps, CRS or transform raise. The fused cube's path reads
  `s2l2a-s1rtc_20180101-20190101_P1M_<hash8>/<id>/`. Changing the S2 input's bands moves the S1
  component's path (`grid_from`). A fused build with `mosaic_scheme="acquisition"` raises.
- **AC6:** for one cell geometry, the training path and the inference path produce byte-identical
  fused cubes.

- **AC9 (D1, D7; A1: single-collection only, since fused cubes come in P5):** an S2 cube's path
  reads `s2l2a_20180101-20190101_P1M_<hash8>/<id>/`, and with `mosaic_days=10` its period is `P10D`.
  Two requests that differ only in bands or `properties_filter` share the readable part and get
  different hashes. `params.json` round-trips every parameter. Registering a collection with
  `short_name="S2/x"` raises. A collection registered without `short_name` gets the fallback code,
  and a second collection with the same code raises. `"month"` with `mosaic_scheme="acquisition"`
  raises.
- **AC10 (D8):** a synthetic fused training run writes `lineage.json`. Its `process_steps` are, in
  order, download, mask, resample, mosaic, fuse, flatten and features. Its `granule_ids` equal the
  catalog rows the cubes used, and its `statement` names both collections and T. An inference run
  writes `lineage.json` with a `predict` step and the model digest. Its STAC Collection carries
  `processing:software["rise-eo"]` and a `metadata` asset pointing to `lineage.json`. Its
  `request`, re-run with only the output folder changed, gives byte-identical arrays on a
  single-MGRS-tile fixture. (A1) It contains no credential and no `runner_kwargs` value. Its paths
  are recorded as given. `geometry.geojson` matches its recorded sha256. Editing a feature function's body changes
  its recorded source hash.
- **AC11 (D9):** a bundle saved with `training_data=` holds the training `lineage.json`. Inference
  with the same inputs passes. Inference with `properties_filter={"sat:orbit_state": "descending"}`
  against a both-orbits training set raises, naming `properties_filter` and both values. A bundle
  without training lineage prints one line and runs.

Real runs (gate 4) run **on Azure ML only**. (A1) P5's run-book holds AC7, and P6 extends the same
run-book with AC8, since AC8 reads D8's lineage. The archive is on blob, and the
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
  pixels matches the training cube (D6). The run-book prints both `lineage.json` files, and the
  user reads them.

## Out of scope

- **Windowed S1 reads.** S1 still downloads whole ~250 km scenes (see Risks). File as an issue.
- **Recreating data from `lineage.json`** (`rise_eo.recreate(lineage_path, output=…)`). D8 records
  everything it needs; the verb itself is a later feature, #216.
- An adapter declaring its own `inputs`. D9's training lineage covers the check this would serve.
- A catalog *across* training sets (a STAC Collection of every training set an archive holds).
  `lineage.json` is per training set. File as an issue if wanted.
- Fusion with HLS, speckle filtering, dB conversion (a `feature_sequence` step can do that),
  `sat:relative_orbit` selection, and S1 from CDSE.
- Other calendar periods (week, dekad, season).

## Phases

One PR per phase. P1–P3 do not depend on each other. P5 needs P4. P6 needs P5, so that the
lineage records the fuse step.

| Phase | Decisions | Size |
|---|---|---|
| P1 | D3 orbit opt-in | ~1 h |
| P2 | D1 monthly windows, D7 readable cube paths | ~1 day |
| P3 | D2 per-polygon windows | half a day to a day |
| P4 | D4 `CollectionInput`, multi-collection download | ~half a day |
| P5 | D5 + D6 fused cube, training and inference; AC7 run-book on Azure ML | 1–2 days |
| P6 | D8 `lineage.json`, D9 training lineage in the bundle; AC8 run-book on Azure ML | ~1 day |

Each implementation PR updates `README.md`, `ARCHITECTURE.md`, `CONTEXT.md`, `ROADMAP.md` and
`AGENTS.md`'s "downloads Sentinel-2 L2A imagery" line, and P1 adds an amendment note to spec 58 D9.

## Risks

- **S1 download size.** Measured 2026-09-12 over one grid cell, April–September 2018, both orbits:
  387.7 GB (~3.7 GB per scene). A year is roughly twice that per location. That is why every real
  run for this spec is on Azure ML (blob storage, in-region transfer), and why no run-book downloads
  S1 to a laptop. Windowed reads would cut it to the bytes the cell needs, but are out of scope.
- **Mixed orbit geometry (D3).** Field-level backscatter differs by look direction, so a model
  trained on mixed medians must also be served mixed medians. D6 builds both the same way, and D9
  refuses an inference request whose `properties_filter` differs from training. A bundle saved
  without `training_data=` is not checked.
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
  enabled, one reason D7 keeps the segment short. Microsoft, "Naming and Referencing Containers,
  Blobs, and Metadata": reserved URL characters must be escaped, which is why D7's codes are
  `[a-z0-9]` joined by `-` and `_`.
- sits book (above): `period = "P16D"`, an ISO 8601 duration, the form of D7's period.
- ESIP wiki, "ISO Lineage" (`wiki.esipfed.org/ISO_Lineage`, edited 2017-09-29): ISO 19115's lineage
  as sources used or produced in process steps, and a process step's fields (D8).
- STAC processing extension README and CHANGELOG (`github.com/stac-extensions/processing`):
  `processing:software` and `processing:lineage` definitions, and v1.0.0 on 2021-03-08 (D8).
- W3C, "PROV-Overview" (2013-04-30): provenance as "entities, activities, and people involved in
  producing a piece of data", the rejected alternative (D8).
- Mitchell et al., "Model Cards for Model Reporting" (FAT* '19; arXiv 1810.03993, 2018-10-05):
  section 4.6 "Training Data", prose for people, the nearest practice to D9.

Searches run for this spec: openEO resample/merge and monthly aggregation; combining S1 ascending and
descending in monthly composites for crop classification; STAC multi-collection item search;
CropHarvest's export code.
