"""Spec 214 P1 (D3, AC3): listing both Sentinel-1 orbit states opts in to mixing them."""

from __future__ import annotations

import datetime
import json

import geopandas as gpd
import pandas as pd
import pytest
import shapely

from rise_eo import collections as _collections
from rise_eo import config
from rise_eo.datacube import builder
from rise_eo.workflows import create_datacube

BOTH = {"sat:orbit_state": ["ascending", "descending"]}


def _mixed():
    box1 = shapely.geometry.box(0, 0, 10, 10)
    rows = [("a1", "ascending"), ("d1", "descending")]
    gdf = gpd.GeoDataFrame({
        "id": [r[0] for r in rows], "filepath": [""] * 2, "band": ["vv"] * 2,
        "timestamp": [pd.Timestamp("2018-04-01", tz="UTC")] * 2,
        "geometry": [box1] * 2, "area_contribution": [1.0] * 2, "offset": [0] * 2,
        "nodata": [-32768] * 2,
        "properties": [json.dumps({"sat:orbit_state": r[1]}) for r in rows],
    }, crs="EPSG:4326")
    return gdf, gpd.GeoDataFrame({"geometry": [box1]}, crs="EPSG:4326")


@pytest.mark.parametrize("pf", [None, {}, {"sat:orbit_state": "ascending"},
                                {"sat:orbit_state": ["ascending"]}])
def test_mixed_orbits_raise_without_both_values(pf):
    decl = _collections.get("sentinel-1-rtc")
    gdf, shape = _mixed()
    with pytest.raises(ValueError, match="sat:orbit_state"):
        builder._enforce_mosaic_partition(gdf, shape, decl, pf)


def test_mixed_orbits_pass_when_both_values_listed():
    decl = _collections.get("sentinel-1-rtc")
    gdf, shape = _mixed()
    builder._enforce_mosaic_partition(gdf, shape, decl, BOTH)  # must not raise


def test_unrelated_filter_key_does_not_exempt():
    decl = _collections.get("sentinel-1-rtc")
    gdf, shape = _mixed()
    with pytest.raises(ValueError):
        builder._enforce_mosaic_partition(gdf, shape, decl, {"z": ["a", "b"]})


def test_build_datacube_passes_the_filter_to_the_check(tmp_path):
    """Through build_datacube: both values listed get past the partition check (the build
    then fails later on the empty filepaths, not with the partition error)."""
    decl = _collections.get("sentinel-1-rtc")
    gdf, shape = _mixed()
    kw = dict(
        catalog_subset=gdf, shape_gdf=shape, startdate=datetime.datetime(2018, 4, 1),
        enddate=datetime.datetime(2018, 9, 30), bands=["vv"], declaration=decl,
        export_folderpath=str(tmp_path), if_missing_files=None,
    )
    with pytest.raises(ValueError, match="rows span multiple values"):
        builder.build_datacube(**kw)
    try:
        builder.build_datacube(**kw, properties_filter=BOTH)
    except Exception as exc:
        assert "rows span multiple values" not in str(exc)


def test_both_orbits_land_at_a_different_path_than_one():
    decl = _collections.get("sentinel-1-rtc")
    kw = dict(bands=["vv"], mosaic_scheme=config.MOSAIC_SCHEME,
              collection="sentinel-1-rtc", declaration=decl)
    start, end = datetime.datetime(2018, 4, 1), datetime.datetime(2018, 9, 30)
    segs = {
        create_datacube.window_folder_segment(start, end, 30, properties_filter=pf, **kw)
        for pf in (None, {"sat:orbit_state": "ascending"},
                   {"sat:orbit_state": "descending"}, BOTH)
    }
    assert len(segs) == 4
