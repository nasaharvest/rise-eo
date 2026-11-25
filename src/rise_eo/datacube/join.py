"""Join datacubes of different collections, built on one grid, along the band axis."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np

FUSED_DTYPE = "float32"
FUSED_NODATA = 0


def _same_grid(first: dict, other: dict) -> str | None:
    """The name of the first grid field on which two `geotiff_metadata` differ, or `None`."""
    for field in ("crs", "transform", "height", "width"):
        if first[field] != other[field]:
            return field
    return None


def join_cubes(
    cubes: Sequence[tuple[np.ndarray, dict]], *, collections: Sequence[str],
    paths: Sequence[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Concatenate `(datacube, metadata)` pairs along the band axis, in the order given.

    Every cube must carry identical `timestamps`, CRS, transform, height and width, or this
    raises. The result is `float32`, in the array and in every dtype field of its metadata;
    each component's nodata (`metadata["nodata"]`) becomes 0, the fused cube's single nodata.
    `metadata["components"]` lists each component's collection, path (from `paths`), bands and
    provenance, and
    `metadata["band_collections"]` maps each band, in band order, to its collection.
    """
    if len(cubes) != len(collections) or len(cubes) < 2:
        raise ValueError("join_cubes needs one collection name per cube, and at least two cubes.")
    first_md = cubes[0][1]
    first_gt = first_md["geotiff_metadata"]
    arrays, band_collections, components, provenance = [], {}, [], {}
    for k, ((cube, md), collection) in enumerate(zip(cubes, collections)):
        if list(md["timestamps"]) != list(first_md["timestamps"]):
            raise ValueError(
                f"join_cubes: {collection!r} has different timestamps from {collections[0]!r}.")
        field = _same_grid(first_gt, md["geotiff_metadata"])
        if field is not None:
            raise ValueError(
                f"join_cubes: {collection!r} is not on {collections[0]!r}'s grid "
                f"({field} differs).")
        nodata = md.get("nodata", md["geotiff_metadata"].get("nodata"))
        out = cube.astype(np.float32)
        if nodata is not None:
            out[cube == nodata] = FUSED_NODATA
        arrays.append(out)
        for band in md["bands"]:
            if band in band_collections:
                raise ValueError(
                    f"join_cubes: band {band!r} is in both {band_collections[band]!r} and "
                    f"{collection!r}.")
            band_collections[band] = collection
        components.append({"collection": collection,
                           "path": None if paths is None else paths[k],
                           "bands": list(md["bands"]),
                           "provenance": md.get("provenance", {})})
        provenance.update(md.get("provenance") or {})

    metadata = copy.deepcopy(first_md)
    metadata["bands"] = list(band_collections)
    metadata["band_collections"] = band_collections
    metadata["components"] = components
    metadata["nodata"] = FUSED_NODATA
    metadata["provenance"] = provenance
    gt = dict(metadata["geotiff_metadata"])
    gt.update(dtype=FUSED_DTYPE, nodata=FUSED_NODATA)
    metadata["geotiff_metadata"] = gt
    return np.concatenate(arrays, axis=-1), metadata
