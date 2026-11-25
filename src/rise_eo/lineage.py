"""`lineage.json`: the sources and ordered process steps behind a training set or an inference output.

The shape follows ISO 19115's lineage model. Every value comes from what ran: the call's
arguments, the collection declarations and the cubes' own metadata.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from rise_eo import config
from rise_eo.catalog import processing as _processing
from rise_eo.datacube.files import TRAINING_METADATA_NAME
from rise_eo.storage import fs

LINEAGE_NAME = "lineage.json"
GEOMETRY_NAME = "geometry.geojson"
LINEAGE_VERSION = 1
SUPPORTED_LINEAGE_VERSIONS = (1,)

_LIBRARIES = ("numpy", "pandas", "geopandas", "rasterio", "pyproj", "numba", "fsspec")
_NEVER_RECORDED = ("creds", "runner_kwargs", "storage")
_OVERLAP_CAVEAT = (
    "Where MGRS tiles overlap, the pixel kept depends on download order (#203), so a rebuild "
    "can differ there. overlapping_mgrs_tiles lists the tiles a cube used more than one of."
)


# --- code and libraries ---------------------------------------------------------------------

def _git(*args: str, cwd: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _checkout_state() -> tuple[str, bool] | None:
    """`(commit, dirty)` when this package is imported from a git checkout, else `None`."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.exists(os.path.join(root, ".git")):
        return None
    commit = _git("rev-parse", "HEAD", cwd=root)
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=root)
    return None if not commit or status is None else (commit, bool(status))


def _pip_commit() -> str | None:
    """The commit pip recorded for an install from a git ref (`direct_url.json`), else `None`."""
    try:
        raw = importlib.metadata.distribution("rise-eo").read_text("direct_url.json")
        return (json.loads(raw).get("vcs_info") or {}).get("commit_id") if raw else None
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return None


@functools.lru_cache(maxsize=1)
def code_record() -> dict:
    """The running rise-eo's version and, when known, its git commit and whether the tree was dirty."""
    record: dict = {"rise-eo": importlib.metadata.version("rise-eo")}
    state = _checkout_state()
    if state is not None:
        record.update(commit=state[0], dirty=state[1])
    elif (commit := _pip_commit()) is not None:
        record.update(commit=commit, dirty=False)
    return record


def software() -> dict:
    """The rise-eo version plus the versions of Python and of the libraries that shape the pixels."""
    import rasterio

    versions: dict = {"rise-eo": code_record()["rise-eo"], "python": platform.python_version(),
                      "gdal": rasterio.__gdal_version__}
    for name in _LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


# --- what the cubes say ---------------------------------------------------------------------

class CubeFacts:
    """What a set of cubes' metadata says about where they came from."""

    def __init__(self) -> None:
        self._provenance: list = []
        self._code: dict[str, dict] = {}
        self._missing_code = False
        self._overlap: dict[str, set[str]] = {}

    def add(self, metadata: dict) -> None:
        """Take one cube's metadata (and its components', for a fused cube) into account."""
        from rise_eo.catalog.stac import _parse_mgrs

        provenance = metadata.get("provenance") or {}
        self._provenance.append(provenance)
        for record in [metadata.get("code")] + [c.get("code") for c in metadata.get("components", [])]:
            if record is None:
                self._missing_code = True
            else:
                self._code[json.dumps(record, sort_keys=True)] = record
        for collection, block in provenance.items():
            tiles = {parsed[0] for i in block.get("ids", []) if (parsed := _parse_mgrs(str(i)))}
            if len(tiles) > 1:
                self._overlap.setdefault(collection, set()).update(tiles)

    def result(self) -> dict:
        """`granule_ids` and `provenance` per collection, the distinct `build_code` records
        (`None` when a cube recorded none) and the `overlapping_mgrs_tiles` per collection."""
        merged = _processing.merge_provenance(self._provenance, with_ids=True)
        code = [self._code[k] for k in sorted(self._code)]
        return {
            "granule_ids": {c: b["ids"] for c, b in merged.items()},
            "provenance": {c: {k: v for k, v in b.items() if k != "ids"} for c, b in merged.items()},
            "build_code": code + ([None] if self._missing_code else []),
            "overlapping_mgrs_tiles": {c: sorted(t) for c, t in sorted(self._overlap.items())},
        }


# --- recording a request --------------------------------------------------------------------

def jsonable(value):
    """`value` as plain JSON data: dates as ISO strings, callables and objects as `module.qualname`."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, gpd.GeoDataFrame):
        return f"<GeoDataFrame, see {GEOMETRY_NAME}>"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    owner = value if callable(value) and hasattr(value, "__qualname__") else type(value)
    return f"{owner.__module__}.{owner.__qualname__}"


def request_record(verb: str, arguments: Mapping, *, drop: Sequence[str] = ()) -> dict:
    """The verb and the arguments it was called with. Credentials are never recorded and
    `runner_kwargs` holds private resource identifiers, so neither is."""
    skip = {*_NEVER_RECORDED, *drop}
    return {"verb": verb, "arguments": {k: jsonable(v) for k, v in arguments.items() if k not in skip}}


def function_record(fn, kwargs=None) -> dict:
    """A feature function as `module.qualname` plus a sha256 of its source, so an edited body shows."""
    try:
        digest = hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
    except (OSError, TypeError):
        digest = None
    return {"function": f"{fn.__module__}.{fn.__qualname__}", "source_sha256": digest,
            "kwargs": jsonable(kwargs or {})}


def feature_functions(adapter, feature_sequence) -> list[dict]:
    """The feature functions an adapter (or a raw `feature_sequence`) applies, in order."""
    sequence = getattr(adapter, "feature_sequence", None) if adapter is not None else feature_sequence
    if adapter is not None and sequence is None:
        return [function_record(adapter.features)]
    return [function_record(fn, kwargs) for fn, kwargs in sequence or []]


def write_geometry(gdf: gpd.GeoDataFrame, folderpath: str) -> str:
    """Write `gdf` as `geometry.geojson` in `folderpath`; returns the file's sha256."""
    text = gdf.to_json(default=str)
    fs.write_text(os.path.join(folderpath, GEOMETRY_NAME), text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- the document ---------------------------------------------------------------------------

def _period(mosaic_days) -> str:
    return "P1M" if mosaic_days == "month" else f"P{int(mosaic_days)}D"


def _declaration(collection: str):
    from rise_eo import collections as _collections

    return _collections.get(collection)


def source_collections(inputs: Sequence[dict], facts: dict, provenance: dict) -> list[dict]:
    """One `sources.collections` entry per input, with the granules the cubes used."""
    out = []
    for i in inputs:
        block = provenance.get(i["collection"], {})
        ids = facts["granule_ids"].get(i["collection"], [])
        out.append({
            "collection": i["collection"], "bands": list(i["bands"]),
            "properties_filter": jsonable(i["properties_filter"]), "processing": i["processing"],
            "catalog": i["catalog"], "provider": block.get("source", []),
            "processing_version": block.get("processing_version", []),
            "n_granules": len(ids), "granule_ids": ids,
        })
    return out


def process_steps(inputs: Sequence[dict], *, mosaic_days, startdate, enddate, per_polygon: bool,
                  n_timestamps: int, download: bool, source: str | None) -> list[dict]:
    """The steps every cube went through, in order: download, mask, resample, mosaic and, for
    several inputs, fuse."""
    from rise_eo.datacube import join

    declarations = [_declaration(i["collection"]) for i in inputs]
    first = declarations[0]
    reference = f"{inputs[0]['collection']} {first.reference_band or inputs[0]['bands'][0]}"
    steps: list[dict] = [{
        "step": "download",
        "description": "Imagery read from the archive's catalogs, fetched by rise_eo.download.",
        "parameters": {"downloaded_by_this_call": download, "source": source if download else None,
                       "catalogs": {i["collection"]: i["catalog"] for i in inputs}},
    }]
    masks = [
        {"collection": i["collection"], "band": d.mask_spec.band, "mask_type": d.mask_spec.mask_type,
         "classes": list(d.mask_spec.classes), "band_dropped_after_masking": not d.mask_keep}
        for i, d in zip(inputs, declarations) if d.mask_spec is not None and d.mask_spec.band in i["bands"]
    ]
    if masks:
        steps.append({"step": "mask", "description": "Masked pixels set to nodata.",
                      "parameters": {"masks": masks}})
    steps.append({
        "step": "resample",
        "description": "Every image resampled onto the reference band's grid.",
        "parameters": {"method": "nearest", "reference": reference},
    })
    steps.append({
        "step": "mosaic", "description": "Calendar-interval mosaics of the images in each window.",
        "parameters": {"method": first.mosaic_method, "period": _period(mosaic_days),
                       "startdate": None if per_polygon else jsonable(startdate),
                       "enddate": None if per_polygon else jsonable(enddate),
                       "windows": "per polygon" if per_polygon else "one window",
                       "n_timestamps": n_timestamps},
    })
    if len(inputs) > 1:
        steps.append({
            "step": "fuse", "description": "Collections joined along the band axis.",
            "parameters": {"order": [i["collection"] for i in inputs], "dtype": join.FUSED_DTYPE,
                           "nodata": join.FUSED_NODATA, "grid": reference},
        })
    return steps


def flatten_steps(*, adapter, feature_sequence, aggregate) -> list[dict]:
    """The `flatten` step and, when features were asked for, the `features` step."""
    steps = [{"step": "flatten", "description": "Valid pixels of every cube stacked into one array.",
              "parameters": {"nodata": config.NODATA}}]
    if adapter is not None or feature_sequence is not None or aggregate is not None:
        steps.append({"step": "features", "description": "Optional aggregation, then the feature transform.",
                      "parameters": {"aggregate": jsonable(aggregate),
                                     "feature_sequence": feature_functions(adapter, feature_sequence)}})
    return steps


def statement_for(subject: str, inputs: Sequence[dict], *, mosaic_days, n_timestamps: int) -> str:
    """One generated sentence naming the collections, their bands, the mosaic period and T."""
    parts = " and ".join(f"{i['collection']} ({', '.join(i['bands'])})" for i in inputs)
    fused = f", fused on the {inputs[0]['collection']} reference grid" if len(inputs) > 1 else ""
    return (f"{subject}: {parts}, {_period(mosaic_days)} median mosaics, "
            f"T={n_timestamps}{fused}.")


def document(*, statement: str, runner: str, request: dict, sources: dict, steps: list[dict],
             facts: dict, extra: Mapping | None = None) -> dict:
    """The `lineage.json` content; raises when a cube recorded no build code."""
    if None in facts["build_code"]:
        raise ValueError(
            "a datacube has no `code` record in its metadata, so lineage.json cannot say what "
            "built it. It was built before cubes recorded their code: rebuild the cubes "
            "(create_training_data(overwrite='datacubes') or a fresh run folder)."
        )
    doc = {
        "lineage_version": LINEAGE_VERSION,
        "statement": statement,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "software": software(),
        "runner": runner,
        "code": {"driver": code_record(), "build": facts["build_code"]},
        "sources": sources,
        "process_steps": steps,
        "request": request,
    }
    if "overlapping_mgrs_tiles" in facts:
        doc["determinism"] = {"caveat": _OVERLAP_CAVEAT,
                              "overlapping_mgrs_tiles": facts["overlapping_mgrs_tiles"]}
    doc.update(extra or {})
    return doc


def write_training(
    export_folderpath: str, *, request: dict, runner: str, gdf: gpd.GeoDataFrame,
    geometry_path: str | None, id_col: str, label_col: str | None, inputs: Sequence[dict],
    mosaic_days, startdate, enddate, per_polygon: bool, download: bool, source: str | None,
    adapter, feature_sequence, aggregate,
) -> None:
    """Write `lineage.json` and `geometry.geojson` next to the flattened arrays in `export_folderpath`."""
    metadata = fs.load_npy(os.path.join(export_folderpath, TRAINING_METADATA_NAME),
                           allow_pickle=True)[()]
    keys = ("granule_ids", "build_code", "overlapping_mgrs_tiles")
    if any(k not in metadata for k in keys):
        raise ValueError(
            f"{export_folderpath} was flattened by a rise-eo that did not record its cubes' "
            "granules and build code. Re-run create_training_data(overwrite='flatten').")
    facts = {k: metadata[k] for k in keys}
    n_timestamps = len(metadata["timestamps"])
    name = os.path.basename(geometry_path.rstrip("/")) if geometry_path else "an in-memory GeoDataFrame"
    sources = {
        "geometry": {"path": geometry_path, "file": GEOMETRY_NAME,
                     "sha256": write_geometry(gdf, export_folderpath), "n_features": len(gdf),
                     "id_col": id_col, "label_col": label_col},
        "collections": source_collections(inputs, facts, metadata.get("provenance") or {}),
    }
    steps = process_steps(
        inputs, mosaic_days=mosaic_days, startdate=startdate, enddate=enddate,
        per_polygon=per_polygon, n_timestamps=n_timestamps, download=download, source=source,
    ) + flatten_steps(adapter=adapter, feature_sequence=feature_sequence, aggregate=aggregate)
    doc = document(
        statement=statement_for(f"Training data for {len(gdf)} polygons of {name}", inputs,
                                mosaic_days=mosaic_days, n_timestamps=n_timestamps),
        runner=runner, request=request, sources=sources, steps=steps, facts=facts,
    )
    write(os.path.join(export_folderpath, LINEAGE_NAME), doc)


def _predict_step(spec: dict, *, predict_batch_size, skip_nan) -> dict:
    return {"step": "predict", "description": "The model run over each datacube, one COG per cube.",
            "parameters": {"adapter": spec.get("adapter"), "required_bands": spec.get("required_bands"),
                           "n_timestamps": spec.get("n_timestamps"),
                           "predict_batch_size": predict_batch_size, "skip_nan": skip_nan}}


def cube_facts(metadata_filepaths: Sequence[str]) -> dict:
    """`CubeFacts.result()` over the cubes whose metadata files are `metadata_filepaths`."""
    import concurrent.futures

    facts = CubeFacts()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for metadata in pool.map(lambda p: fs.load_npy(p, allow_pickle=True)[()], metadata_filepaths):
            facts.add(metadata)
    return facts.result()


def write_inference(
    output_folderpath: str, *, request: dict, runner: str, roi: gpd.GeoDataFrame,
    roi_path: str | None, grid_size_km: float, scale_fact: float, n_cells: int,
    inputs: Sequence[dict], facts: dict, mosaic_days, startdate, enddate, n_timestamps: int,
    spec: dict, model: dict, training_lineage: dict | None, predict_batch_size, skip_nan,
) -> dict:
    """Write `lineage.json` and `geometry.geojson` next to the STAC catalog; returns the document."""
    name = os.path.basename(roi_path.rstrip("/")) if roi_path else "an in-memory GeoDataFrame"
    sources = {
        "roi": {"path": roi_path, "file": GEOMETRY_NAME,
                "sha256": write_geometry(roi, output_folderpath), "n_features": len(roi)},
        "grid": {"grid_size_km": grid_size_km, "scale_fact": scale_fact, "n_cells": n_cells},
        "collections": source_collections(inputs, facts, facts["provenance"]),
    }
    steps = process_steps(
        inputs, mosaic_days=mosaic_days, startdate=startdate, enddate=enddate, per_polygon=False,
        n_timestamps=n_timestamps, download=False, source=None,
    ) + [_predict_step(spec, predict_batch_size=predict_batch_size, skip_nan=skip_nan)]
    doc = document(
        statement=statement_for(
            f"Inference with {model['given']} over {n_cells} grid cells of {name}", inputs,
            mosaic_days=mosaic_days, n_timestamps=n_timestamps),
        runner=runner, request=request, sources=sources, steps=steps, facts=facts,
        extra={"model": model, "training_lineage": training_lineage},
    )
    write(os.path.join(output_folderpath, LINEAGE_NAME), doc)
    return doc


def write_prebuilt(
    output_folderpath: str, *, request: dict, runner: str, cubes: Sequence[str], spec: dict,
    model: dict, training_lineage: dict | None, predict_batch_size, skip_nan,
) -> dict:
    """Write the short `lineage.json` of inference on pre-built cubes: their paths, the model and `predict`."""
    doc = document(
        statement=f"Inference with {model['given']} over {len(cubes)} pre-built datacubes.",
        runner=runner, request=request, sources={"inference_datacubes": list(cubes)},
        steps=[_predict_step(spec, predict_batch_size=predict_batch_size, skip_nan=skip_nan)],
        facts={"build_code": []},
        extra={"model": model, "training_lineage": training_lineage},
    )
    write(os.path.join(output_folderpath, LINEAGE_NAME), doc)
    return doc


def write(path: str, doc: dict) -> None:
    """Write `doc` as `path`."""
    fs.write_text(path, json.dumps(doc, indent=2, default=str))


def read(path: str) -> dict:
    """Read a `lineage.json`, refusing a `lineage_version` this rise-eo does not support."""
    with fs.open(path, "r") as f:
        doc = json.load(f)
    if doc.get("lineage_version") not in SUPPORTED_LINEAGE_VERSIONS:
        raise ValueError(
            f"{path} has lineage_version {doc.get('lineage_version')!r}; this rise-eo reads "
            f"{list(SUPPORTED_LINEAGE_VERSIONS)}. Upgrade rise-eo, or rebuild the data."
        )
    return doc


# --- the model's training lineage against an inference request ------------------------------

def _filter_key(properties_filter) -> dict | None:
    if not properties_filter:
        return None
    return {k: sorted([v] if isinstance(v, str) else map(str, v))
            for k, v in sorted(properties_filter.items())}


def training_mismatches(training: dict, inputs: Sequence[dict], *, mosaic_days, n_timestamps) -> list[str]:
    """Where an inference request differs from the lineage the model was trained on: the
    collections in order, each one's bands and `properties_filter`, the period and T."""
    trained = training["sources"]["collections"]
    errs: list[str] = []
    if [c["collection"] for c in trained] != [i["collection"] for i in inputs]:
        return [f"collections: the model was trained on {[c['collection'] for c in trained]}, "
                f"this request has {[i['collection'] for i in inputs]}."]
    for t, i in zip(trained, inputs):
        if list(t["bands"]) != list(i["bands"]):
            errs.append(f"bands of {t['collection']}: the model was trained on {t['bands']}, "
                        f"this request has {list(i['bands'])}.")
        if _filter_key(t["properties_filter"]) != _filter_key(i["properties_filter"]):
            errs.append(f"properties_filter of {t['collection']}: the model was trained on "
                        f"{t['properties_filter']!r}, this request has {i['properties_filter']!r}.")
    mosaic = next(s for s in training["process_steps"] if s["step"] == "mosaic")["parameters"]
    if mosaic["period"] != _period(mosaic_days):
        errs.append(f"period: the model was trained on {mosaic['period']}, this request has "
                    f"{_period(mosaic_days)}.")
    if mosaic["n_timestamps"] != n_timestamps:
        errs.append(f"T: the model was trained on {mosaic['n_timestamps']} timestamps, this "
                    f"request has {n_timestamps}.")
    return errs
