"""Spec 214 P1 (D3, AC3): listing both Sentinel-1 orbit states opts in to mixing them."""

from __future__ import annotations

import datetime
import json
import pathlib

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


def _s1_tile_catalog(path, tmp_path):
    """A tile-row S1 catalog (the shape `setup` reads) with one ascending and one
    descending acquisition, real 4x4 tiles on disk."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from rise_eo.catalog import declaration as declaration_module
    from rise_eo.storage import fs

    crs = "EPSG:32633"
    tile = gpd.GeoSeries([box(500000, 4999960, 500040, 5000000)], crs=crs).to_crs("EPSG:4326")
    rows = []
    for i, (orbit, ts) in enumerate([("ascending", "2018-06-02"), ("descending", "2018-06-04")]):
        d = tmp_path / f"s1_{i}"
        d.mkdir()
        with rasterio.open(d / "vv.tif", "w", driver="GTiff", height=4, width=4, count=1,
                           dtype="float32", crs=crs,
                           transform=from_origin(500000, 5000000, 10, 10), nodata=0) as dst:
            dst.write(np.full((1, 4, 4), 0.1 + i, dtype="float32"))
        rows.append({
            "id": f"s1_{i}", "acquisition_key": f"s1_{i}", "processing_version": None,
            "processing_datetime": None, "source": "mpc", "satellite": "sentinel-1-rtc",
            "timestamp": pd.Timestamp(ts, tz="UTC"), "s3url": f"s3://x{i}",
            "local_folderpath": str(d), "files": "vv.tif", "cloud_cover": 0.0, "offset": 0,
            "nodata": 0, "area_contribution": 100.0, "geometry": tile.iloc[0],
            "properties": json.dumps({"sat:orbit_state": orbit}),
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    declaration_module.to_attrs(gdf, _collections.get("sentinel-1-rtc"))
    fs.write_parquet(str(path), gdf)
    shape = tmp_path / "shape.geojson"
    gpd.GeoDataFrame({"id": ["c1"], "label": [0], "geometry": [tile.iloc[0]]},
                     crs="EPSG:4326").to_file(str(shape), driver="GeoJSON")
    return shape


def _setup_then_run_task(tmp_path, properties_filter):
    from rise_eo.workflows import task

    cat = tmp_path / "catalog.parquet"
    shape = _s1_tile_catalog(cat, tmp_path)
    csv = tmp_path / "run" / "input.csv"
    start, end = datetime.datetime(2018, 6, 1), datetime.datetime(2018, 6, 21)
    create_datacube.setup(
        catalog_filepath=str(cat), timestamp_col="timestamp", shapefilepath=str(shape),
        id_col="id", run_folderpath=str(tmp_path / "run"), startdate=start, enddate=end,
        bands=["vv"], mosaic_days=20, csv_filepath=str(csv), label_col="label",
        collection="sentinel-1-rtc", properties_filter=properties_filter,
    )
    row = pd.read_csv(csv).iloc[0]
    task.run_task(
        str(row["shapefilepath"]), str(row["catalog_filepath"]), start, end,
        str(row["export_folderpath"]), bands=["vv"], mosaic_days=20,
        declaration_filepath=str(row["declaration_filepath"]), if_missing_files="warn",
    )
    return row


def test_run_task_builds_both_orbits_when_both_listed(tmp_path):
    row = _setup_then_run_task(tmp_path, BOTH)
    assert (pathlib.Path(row["export_folderpath"]) / "datacube.npy").exists()


def test_run_task_with_no_filter_still_raises(tmp_path):
    with pytest.raises(ValueError, match="rows span multiple values"):
        _setup_then_run_task(tmp_path, None)


def test_run_task_with_one_value_builds_that_orbit_only(tmp_path):
    row = _setup_then_run_task(tmp_path, {"sat:orbit_state": "ascending"})
    assert int(row["images_count"]) == 1


def test_path_keeps_the_original_declaration_digest(tmp_path):
    """The control file drops the mixed key, but the path still names the original."""
    row = _setup_then_run_task(tmp_path, BOTH)
    decl = _collections.get("sentinel-1-rtc")
    seg = create_datacube.window_folder_segment(
        datetime.datetime(2018, 6, 1), datetime.datetime(2018, 6, 21), 20, bands=["vv"],
        mosaic_scheme=config.MOSAIC_SCHEME, collection="sentinel-1-rtc", declaration=decl,
        properties_filter=BOTH)
    assert seg in str(row["export_folderpath"])
