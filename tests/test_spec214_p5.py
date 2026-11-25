"""Spec 214 P5: D5 + D6 the fused cube, at training and at inference (AC5, AC6)."""
from __future__ import annotations

import datetime
import json
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import Affine, from_origin
from shapely.geometry import box

import rise_eo
from rise_eo import collections as _collections
from rise_eo.api import PreflightError
from rise_eo.catalog import declaration as declaration_module
from rise_eo.datacube import builder
from rise_eo.datacube.join import join_cubes
from rise_eo.storage import fs
from rise_eo.workflows import create_datacube, infer_task, task
from rise_eo.workflows.create_datacube import Component

CRS = "EPSG:32633"
START, END = datetime.datetime(2018, 6, 1), datetime.datetime(2018, 7, 1)
S2_BANDS = ("B08", "SCL")
S1_BANDS = ("vv",)
S1_FILTER = {"sat:orbit_state": "descending"}
S2_VALUE, S1_VALUE = 300, 0.25


def _to_4326(utm_box):
    return gpd.GeoSeries([utm_box], crs=CRS).to_crs("EPSG:4326").iloc[0]


def _write_tif(path, array, transform, dtype, nodata):
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2],
                       count=1, dtype=dtype, crs=CRS, transform=transform, nodata=nodata) as dst:
        dst.write(array.astype(dtype))


def _catalog(tmp_path, name, *, collection, tile, origin, n_px, bands, values, dtype, nodata, days,
             orbit=None):
    """A catalog of one granule per day in `days`, each a `n_px` square grid at 10 m from
    `origin`, one file per band."""
    transform = from_origin(*origin, 10, 10)
    footprint = _to_4326(tile)
    rows = []
    for i, day in enumerate(days):
        folder = tmp_path / f"{name}{i}"
        folder.mkdir()
        for band, value in zip(bands, values):
            _write_tif(folder / f"{band}.tif", np.full((1, n_px, n_px), value), transform,
                       dtype, nodata)
        rows.append({
            "id": f"{name}_{i}", "acquisition_key": f"{name}_{i}", "processing_version": None,
            "processing_datetime": None, "source": "mpc", "satellite": collection,
            "timestamp": pd.Timestamp(day, tz="UTC"), "s3url": f"s3://x/{name}{i}",
            "local_folderpath": str(folder), "files": ",".join(f"{b}.tif" for b in bands),
            "cloud_cover": 0.0, "offset": 0, "nodata": nodata, "area_contribution": 100.0,
            "geometry": footprint,
            "properties": json.dumps({"sat:orbit_state": orbit} if orbit else {}),
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    declaration_module.to_attrs(gdf, _collections.get(collection))
    path = tmp_path / "archive" / collection / "catalog.parquet"
    path.parent.mkdir(parents=True)
    fs.write_parquet(str(path), gdf)
    return str(path)


class Env:
    """A 200 m S2 granule grid and a smaller S1 grid shifted by half a pixel."""

    def __init__(self, tmp_path, s1_days=("2018-06-03", "2018-06-14", "2018-07-05")):
        self.tmp = tmp_path
        self.s2_catalog = _catalog(
            tmp_path, "s2", collection="sentinel-2-l2a", tile=box(500000, 5000000, 500200, 5000200),
            origin=(500000, 5000200), n_px=20, bands=S2_BANDS, values=(S2_VALUE, 4),
            dtype="uint16", nodata=0, days=("2018-06-02", "2018-06-12", "2018-07-03", "2018-07-13"))
        self.s1_catalog = _catalog(
            tmp_path, "s1", collection="sentinel-1-rtc", tile=box(499995, 5000085, 500115, 5000205),
            origin=(499995, 5000205), n_px=12, bands=S1_BANDS, values=(S1_VALUE,),
            dtype="float32", nodata=-32768, days=s1_days, orbit="descending")
        # c1 is inside both grids; c2 is inside S2's only.
        g1 = _to_4326(box(500032, 5000103, 500088, 5000157))
        g2 = _to_4326(box(500142, 5000022, 500178, 5000058))
        self.shapes = str(tmp_path / "shapes.geojson")
        gpd.GeoDataFrame({"id": ["c1", "c2"], "label": [0, 1], "geometry": [g1, g2]},
                         crs="EPSG:4326").to_file(self.shapes, driver="GeoJSON")
        self.components = [
            Component("sentinel-2-l2a", S2_BANDS, self.s2_catalog),
            Component("sentinel-1-rtc", S1_BANDS, self.s1_catalog, properties_filter=S1_FILTER),
        ]
        self.run = str(tmp_path / "run")
        self.csv = os.path.join(self.run, "input.csv")

    def setup(self, run=None, shapes=None, label_col="label", components=None):
        run = run or self.run
        create_datacube.setup(
            catalog_filepath=self.s2_catalog, timestamp_col="timestamp",
            shapefilepath=shapes or self.shapes, id_col="id", run_folderpath=run,
            startdate=START, enddate=END, bands=list(S2_BANDS), mosaic_days="month",
            csv_filepath=os.path.join(run, "input.csv"), label_col=label_col,
            components=components or self.components,
        )
        with fs.open(os.path.join(run, "input.csv"), "r") as f:
            return create_datacube.read_unit_csv(f)

    def segments(self):
        return (create_datacube.fused_folder_segment(START, END, "month", self.components,
                                                     mosaic_scheme="calendar"),
                create_datacube.component_segments(START, END, "month", self.components,
                                                    mosaic_scheme="calendar"))


def _run_row(row, **kw):
    task.run_task(
        row["shapefilepath"], row["catalog_filepath"], pd.to_datetime(row["startdate"]),
        pd.to_datetime(row["enddate"]), row["export_folderpath"], bands=row["bands"].split(","),
        mosaic_days="month", declaration_filepath=row["declaration_filepath"],
        fusion_filepath=row["fusion_filepath"], **kw)


def _cube(folder):
    return (fs.load_npy(os.path.join(folder, "datacube.npy")),
            fs.load_npy(os.path.join(folder, "metadata.pickle.npy"), allow_pickle=True)[()])


# --- paths (D7) ----------------------------------------------------------------------------


def test_fused_paths_follow_d7(tmp_path):
    env = Env(tmp_path)
    fused, (s2_seg, s1_seg) = env.segments()
    assert re.fullmatch(r"s2l2a-s1rtc_20180601-20180701_P1M_[0-9a-f]{8}", fused)
    assert re.fullmatch(r"s2l2a_20180601-20180701_P1M_[0-9a-f]{8}", s2_seg)
    assert re.fullmatch(r"s1rtc-on-s2l2a_20180601-20180701_P1M_[0-9a-f]{8}", s1_seg)
    # the reference component keeps its standalone path
    assert s2_seg == create_datacube.window_folder_segment(
        START, END, "month", bands=list(S2_BANDS), mosaic_scheme="calendar",
        collection="sentinel-2-l2a", declaration=_collections.get("sentinel-2-l2a"))


def test_changing_the_s2_input_moves_the_s1_component_and_the_fused_path(tmp_path):
    env = Env(tmp_path)
    fused, (_, s1_seg) = env.segments()
    env.components[0] = Component("sentinel-2-l2a", ("B08",), env.s2_catalog)
    fused2, (_, s1_seg2) = env.segments()
    assert s1_seg != s1_seg2 and fused != fused2
    # and a standalone S1 cube never shares the component's path
    standalone = create_datacube.window_folder_segment(
        START, END, "month", bands=["vv"], mosaic_scheme="calendar",
        collection="sentinel-1-rtc", declaration=_collections.get("sentinel-1-rtc"),
        properties_filter=S1_FILTER)
    assert standalone != s1_seg


# --- setup and the fused unit ----------------------------------------------------------------


def test_setup_writes_one_fused_row_with_a_fusion_file_and_params(tmp_path, capsys):
    env = Env(tmp_path)
    df = env.setup()
    fused, (s2_seg, s1_seg) = env.segments()
    assert list(df["id"]) == ["c1"]                       # c2 has no S1 imagery
    row = df.iloc[0]
    assert row["export_folderpath"] == os.path.join(env.run, fused, "c1")
    assert row["collection"] == "sentinel-2-l2a+sentinel-1-rtc"
    fusion = json.load(open(row["fusion_filepath"]))["components"]
    assert [c["collection"] for c in fusion] == ["sentinel-2-l2a", "sentinel-1-rtc"]
    assert fusion[1]["export_folderpath"] == os.path.join(env.run, s1_seg, "c1")
    assert os.path.exists(os.path.join(env.run, s1_seg, "declaration.json"))
    params = json.load(open(os.path.join(env.run, fused, "params.json")))
    assert [c["segment"] for c in params["components"]] == [s2_seg, s1_seg]
    out = capsys.readouterr().out
    assert "skip id=c2: no sentinel-1-rtc imagery in the window" in out


def test_a_unit_with_no_s1_rows_is_known_empty_and_the_skip_line_names_s1(tmp_path, capsys):
    env = Env(tmp_path)
    fused, _ = env.segments()
    n_present, n_missing, n_known_empty = create_datacube.build_shortfall_only(
        catalog_filepath=env.s2_catalog, timestamp_col="timestamp", shapefilepath=env.shapes,
        id_col="id", run_folderpath=env.run, startdate=START, enddate=END,
        bands=list(S2_BANDS), mosaic_days="month", csv_filepath=env.csv, label_col="label",
        components=env.components)
    assert (n_present, n_missing) == (0, 2)
    assert create_datacube._read_known_empty(env.run, fused) == {"c2"}
    assert "no sentinel-1-rtc imagery" in capsys.readouterr().out
    # a second pass finds c1 present and c2 known-empty: nothing to set up
    assert create_datacube.build_shortfall_only(
        catalog_filepath=env.s2_catalog, timestamp_col="timestamp", shapefilepath=env.shapes,
        id_col="id", run_folderpath=env.run, startdate=START, enddate=END,
        bands=list(S2_BANDS), mosaic_days="month", csv_filepath=env.csv, label_col="label",
        components=env.components) == (1, 0, 1)


def test_no_unit_with_every_collection_raises(tmp_path):
    env = Env(tmp_path)
    only_c2 = str(tmp_path / "c2.geojson")
    gpd.read_file(env.shapes).query("id == 'c2'").to_file(only_c2, driver="GeoJSON")
    with pytest.raises(create_datacube.NoWorkUnitsError):
        env.setup(shapes=only_c2)


# --- the build (AC5) -------------------------------------------------------------------------


def test_the_fused_cube_joins_the_components_on_the_reference_grid(tmp_path):
    env = Env(tmp_path)
    row = env.setup().iloc[0]
    _run_row(row)

    cube, md = _cube(row["export_folderpath"])
    assert cube.dtype == np.float32 and cube.shape[0] == 1 and cube.shape[-1] == 2
    assert md["bands"] == ["B08", "vv"]                    # inputs order; SCL dropped
    assert md["band_collections"] == {"B08": "sentinel-2-l2a", "vv": "sentinel-1-rtc"}
    assert list(md["band_collections"]) == md["bands"]
    assert md["nodata"] == 0 and md["geotiff_metadata"]["nodata"] == 0
    assert md["geotiff_metadata"]["dtype"] == "float32"
    assert [c["collection"] for c in md["components"]] == ["sentinel-2-l2a", "sentinel-1-rtc"]
    assert (cube[..., 0] == S2_VALUE).sum() == (cube[..., 1] == S1_VALUE).sum() > 0

    fusion = json.load(open(row["fusion_filepath"]))["components"]
    s2_cube, s2_md = _cube(fusion[0]["export_folderpath"])
    s1_cube, s1_md = _cube(fusion[1]["export_folderpath"])
    assert s2_cube.dtype == np.uint16 and s1_cube.dtype == np.float32
    for key in ("crs", "transform", "height", "width"):
        assert s1_md["geotiff_metadata"][key] == s2_md["geotiff_metadata"][key]


def test_a_component_half_a_pixel_off_covers_every_edge_pixel_and_nothing_outside(tmp_path):
    """AC5 (A2): S1's grid is offset by half a pixel from S2's."""
    env = Env(tmp_path)
    row = env.setup().iloc[0]
    _run_row(row)
    fusion = json.load(open(row["fusion_filepath"]))["components"]
    s2_cube, _ = _cube(fusion[0]["export_folderpath"])
    s1_cube, s1_md = _cube(fusion[1]["export_folderpath"])
    assert s1_md["geotiff_metadata"]["transform"].c % 10 == 0     # on the S2 grid, not S1's
    reference = s2_cube[0, :, :, 0] != 0
    component = s1_cube[0, :, :, 0] != -32768
    assert reference.sum() > 20
    assert (component == reference).all()                        # valid at every edge pixel; nodata outside
    assert (s1_cube[0, :, :, 0][reference] == np.float32(S1_VALUE)).all()


def test_a_finished_component_is_reused_when_the_fused_cube_is_rebuilt(tmp_path, monkeypatch):
    env = Env(tmp_path)
    row = env.setup().iloc[0]
    _run_row(row)
    fused_cube = os.path.join(row["export_folderpath"], "datacube.npy")
    before = fs.load_npy(fused_cube)
    os.remove(fused_cube)

    built = []
    real = builder.build_datacube
    monkeypatch.setattr(builder, "build_datacube", lambda **kw: built.append(kw) or real(**kw))
    _run_row(row)
    assert built == []                                        # both components were on disk
    assert np.array_equal(fs.load_npy(fused_cube), before)
    _run_row(row)                                             # the fused cube exists: no-op
    assert built == []


def test_the_second_component_is_built_onto_the_first_ones_grid(tmp_path, monkeypatch):
    env = Env(tmp_path)
    row = env.setup().iloc[0]
    profiles = []
    real = builder.build_datacube
    monkeypatch.setattr(builder, "build_datacube",
                        lambda **kw: profiles.append(kw.get("reference_profile")) or real(**kw))
    _run_row(row)
    assert profiles[0] is None and profiles[1] is not None


def test_join_cubes_refuses_a_mismatch(tmp_path):
    env = Env(tmp_path)
    row = env.setup().iloc[0]
    _run_row(row)
    fusion = json.load(open(row["fusion_filepath"]))["components"]
    names = [c["collection"] for c in fusion]
    s2, s1 = (_cube(c["export_folderpath"]) for c in fusion)
    join_cubes([s2, s1], collections=names)

    def changed(**edits):
        md = {**s1[1], **edits}
        return (s1[0], md)

    gt = s1[1]["geotiff_metadata"]
    with pytest.raises(ValueError, match="timestamps"):
        join_cubes([s2, changed(timestamps=[pd.Timestamp("2018-07-01", tz="UTC")])],
                   collections=names)
    with pytest.raises(ValueError, match="transform"):
        join_cubes([s2, changed(geotiff_metadata={**gt, "transform": gt["transform"] * Affine.translation(0.5, 0)})],
                   collections=names)
    with pytest.raises(ValueError, match="crs"):
        join_cubes([s2, changed(geotiff_metadata={**gt, "crs": rasterio.crs.CRS.from_epsg(32632)})],
                   collections=names)
    with pytest.raises(ValueError, match="both"):
        join_cubes([s2, (s1[0], {**s1[1], "bands": ["B08"]})], collections=names)


def test_an_image_with_the_reference_shape_and_a_shifted_transform_is_resampled():
    ref = {"crs": "c", "height": 4, "width": 4, "transform": from_origin(0, 40, 10, 10)}
    same = {"crs": "c", "height": 4, "width": 4, "transform": from_origin(0, 40, 10, 10)}
    shifted = {**same, "transform": from_origin(5, 40, 10, 10)}
    data = [(None, same), (None, shifted)]
    assert builder._get_indices_to_resample([0, 1], data, ref) == [1]


# --- AC6: training and inference build the same cube ----------------------------------------


def test_ac6_training_and_inference_paths_build_byte_identical_fused_cubes(tmp_path, monkeypatch):
    env = Env(tmp_path)
    train = env.setup(run=str(tmp_path / "train")).iloc[0]
    _run_row(train)

    infer_run = str(tmp_path / "infer")
    infer = env.setup(run=infer_run, label_col=None).iloc[0]
    from rise_eo.model import engine

    monkeypatch.setattr(engine, "_adapter_from_bundle_cached", lambda p: object())
    monkeypatch.setattr(engine, "infer_datacube_to_cog", lambda a, cube, out, **kw: out)
    infer_task.run_infer_task(
        infer["shapefilepath"], infer["catalog_filepath"], pd.to_datetime(infer["startdate"]),
        pd.to_datetime(infer["enddate"]), infer["export_folderpath"],
        bands=infer["bands"].split(","), mosaic_days="month",
        declaration_filepath=infer["declaration_filepath"], bundle_path="b",
        output_filepath=os.path.join(infer["export_folderpath"], "output.tif"),
        fusion_filepath=infer["fusion_filepath"])

    a = open(os.path.join(train["export_folderpath"], "datacube.npy"), "rb").read()
    b = open(os.path.join(infer["export_folderpath"], "datacube.npy"), "rb").read()
    assert a == b


def test_run_infer_group_reads_the_fusion_column_and_ignores_it_when_empty(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(infer_task, "run_infer_task",
                        lambda *a, **kw: seen.append(kw["fusion_filepath"]) or "o")
    base = {"shapefilepath": "s", "catalog_filepath": "c", "startdate": "2018-06-01",
            "enddate": "2018-07-01", "export_folderpath": "e", "bands": "B08",
            "mosaic_days": "month", "declaration_filepath": "d", "mosaic_scheme": "calendar"}
    csv = str(tmp_path / "rows.csv")
    pd.DataFrame([{**base, "fusion_filepath": "f.json"}, {**base}]).to_csv(csv, index=False)
    infer_task.run_infer_group(csv, (0, 2), "bundle")
    assert seen == ["f.json", None]


# --- the Snakefile hop ---------------------------------------------------------------------


def test_the_snakefile_hands_a_fused_row_to_the_task_cli(tmp_path):
    from rise_eo.workflows import runners

    env = Env(tmp_path)
    df = env.setup()
    csv = str(tmp_path / "rows.csv")
    df.to_csv(csv, index=False)
    result = runners.run_local(csv, cores=1)
    assert result.returncode == 0, result.stderr
    assert os.path.exists(os.path.join(df.iloc[0]["export_folderpath"], "datacube.npy"))


# --- API: preflight, identity, download ---------------------------------------------------


S2_IN = rise_eo.CollectionInput("sentinel-2-l2a", list(S2_BANDS))
S1_IN = rise_eo.CollectionInput("sentinel-1-rtc", list(S1_BANDS), properties_filter=S1_FILTER)


def _polygons(env):
    gdf = gpd.read_file(env.shapes).rename(columns={"id": "fid", "label": "crop"})
    return gdf[gdf.fid == "c1"]


def _train(env, tmp_path, **kw):
    args = dict(
        label_polygons=_polygons(env), catalog_filepath=None, startdate=START, enddate=END,
        mosaic_days="month", bands=None, id_col="fid", label_col="crop",
        export_folderpath=str(tmp_path / "export"), inputs=[S2_IN, S1_IN],
        archive_root=str(tmp_path / "archive"))
    args.update(kw)
    return rise_eo.create_training_data(**args)


def _fake_runners(monkeypatch, rows_seen=None):
    from rise_eo.workflows import runners

    def run_local(csv, **kw):
        for _, row in pd.read_csv(csv).iterrows():
            _run_row(row)
            if rows_seen is not None:
                rows_seen.append(row)

    monkeypatch.setattr(runners, "run_local", run_local)


def test_create_training_data_fuses_two_inputs_and_short_circuits_on_rerun(tmp_path, monkeypatch):
    env = Env(tmp_path)
    dispatched = []
    _fake_runners(monkeypatch, dispatched)
    td = _train(env, tmp_path)
    assert td.bands == ["B08", "vv"] and td.n_pixels > 0
    out = td.load()
    assert out["data"].dtype == np.float32 and out["data"].shape[1:] == (1, 2)
    assert list(out["ids"]) == ["c1"] * len(out["ids"])
    n = len(dispatched)
    assert n == 1

    td2 = _train(env, tmp_path)                           # satisfied: nothing is dispatched
    assert len(dispatched) == n and td2.n_pixels == td.n_pixels


def test_a_change_to_either_input_moves_the_flatten_identity(tmp_path):
    env = Env(tmp_path)
    kw = dict(id_col="fid", run_folderpath=str(tmp_path / "run"), startdate=START, enddate=END,
              mosaic_days="month", bands=["B08", "vv"], collection="sentinel-2-l2a",
              mosaic_scheme="calendar", adapter=None, feature_sequence=None, aggregate=None)
    from rise_eo import api

    def identity(components):
        return api._flatten_identity_from_request(_polygons(env), components=components, **kw)

    base = identity(env.components)
    assert base == identity(list(env.components))
    s1_changed = [env.components[0], Component(
        "sentinel-1-rtc", S1_BANDS, env.s1_catalog,
        properties_filter={"sat:orbit_state": ["ascending", "descending"]})]
    s2_changed = [Component("sentinel-2-l2a", ("B08",), env.s2_catalog), env.components[1]]
    assert identity(s1_changed) != base and identity(s2_changed) != base


def test_the_adapter_check_uses_the_fused_band_list(tmp_path):
    env = Env(tmp_path)

    class Needs:
        required_bands = ["B08", "vv", "vh"]
        n_timestamps = 1

    with pytest.raises(PreflightError, match=r"required_bands not in requested bands: \['vh'\]"):
        _train(env, tmp_path, adapter=Needs())

    class NeedsMask:
        required_bands = ["SCL"]           # dropped after masking, so not in the fused cube
        n_timestamps = 1

    with pytest.raises(PreflightError, match=r"\['SCL'\]"):
        _train(env, tmp_path, adapter=NeedsMask())


def test_the_first_input_must_declare_a_reference_band(tmp_path):
    env = Env(tmp_path)
    with pytest.raises(PreflightError, match="reference_band"):
        _train(env, tmp_path, inputs=[S1_IN, S2_IN])


def test_download_true_downloads_every_input_after_every_preflight(tmp_path, monkeypatch):
    from rise_eo import api

    env = Env(tmp_path)
    calls = []
    monkeypatch.setattr(api, "_download_verb", lambda **kw: calls.append(kw))
    _fake_runners(monkeypatch)
    s1_cloudy = rise_eo.CollectionInput("sentinel-1-rtc", ["vv"], max_cloudcover=10.0,
                                        properties_filter=S1_FILTER)
    with pytest.raises(PreflightError, match="cloud"):
        _train(env, tmp_path, inputs=[S2_IN, s1_cloudy], download=True, max_tiles=5)
    assert calls == []                                     # S1's preflight ran before any download

    _train(env, tmp_path, download=True, max_tiles=5)
    assert len(calls) == 1
    assert calls[0]["inputs"] == [S2_IN, S1_IN]
    assert calls[0]["dst_folderpath"] == str(tmp_path / "archive")


def test_inference_preflight_checks_required_bands_against_the_fused_list(tmp_path, monkeypatch):
    from rise_eo import api

    env = Env(tmp_path)
    monkeypatch.setattr(api, "_resolve_model_ref", lambda model, *a, **kw: model)
    monkeypatch.setattr(api, "_model_spec", lambda model: {"required_bands": ["B08", "vh"],
                                                           "n_timestamps": 1})
    with pytest.raises(PreflightError, match=r"miss model-required \['vh'\]"):
        rise_eo.run_inference(
            "bundle", output_folderpath=str(tmp_path / "out"), roi=_polygons(env),
            inputs=[S2_IN, S1_IN], archive_root=str(tmp_path / "archive"), startdate=START,
            enddate=END, mosaic_days="month")


# --- review round 1 ----------------------------------------------------------------------


def test_a_forced_rebuild_rebuilds_every_component(tmp_path, monkeypatch):
    env = Env(tmp_path)
    row = env.setup().iloc[0]
    _run_row(row)
    create_datacube._force_rebuild(env.csv)
    built = []
    real = builder.build_datacube
    monkeypatch.setattr(builder, "build_datacube", lambda **kw: built.append(kw) or real(**kw))
    _run_row(row)
    assert len(built) == 2                                   # both components, from the slices on disk
    assert os.path.exists(os.path.join(row["export_folderpath"], "datacube.npy"))


def test_processing_is_checked_for_every_input_before_any_file_is_written(tmp_path, monkeypatch):
    from rise_eo import api

    env = Env(tmp_path)
    s1 = rise_eo.CollectionInput("sentinel-1-rtc", ["vv"], properties_filter=S1_FILTER,
                                 processing=">=05.00")
    with pytest.raises(PreflightError, match="sentinel-1-rtc"):
        _train(env, tmp_path, inputs=[S2_IN, s1])
    assert not (tmp_path / "export").exists()

    monkeypatch.setattr(api, "_resolve_model_ref", lambda model, *a, **kw: model)
    monkeypatch.setattr(api, "_model_spec", lambda model: {"required_bands": ["B08"],
                                                           "n_timestamps": 1})
    with pytest.raises(PreflightError, match="sentinel-1-rtc"):
        rise_eo.run_inference(
            "bundle", output_folderpath=str(tmp_path / "out"), roi=_polygons(env),
            inputs=[S2_IN, s1], archive_root=str(tmp_path / "archive"), startdate=START,
            enddate=END, mosaic_days="month")
    assert not (tmp_path / "out").exists()


def test_a_component_keeps_its_own_dtype_and_nodata_in_its_metadata(tmp_path):
    env = Env(tmp_path)
    row = env.setup().iloc[0]
    _run_row(row)
    fusion = json.load(open(row["fusion_filepath"]))["components"]
    _, s1_md = _cube(fusion[1]["export_folderpath"])
    _, s2_md = _cube(fusion[0]["export_folderpath"])
    assert (s1_md["geotiff_metadata"]["dtype"], s1_md["geotiff_metadata"]["nodata"]) == (
        "float32", -32768)
    assert (s2_md["geotiff_metadata"]["dtype"], s2_md["geotiff_metadata"]["nodata"]) == ("uint16", 0)
    assert s1_md["geotiff_metadata"]["transform"] == s2_md["geotiff_metadata"]["transform"]


def test_per_polygon_windows_fuse_end_to_end(tmp_path, monkeypatch):
    env = Env(tmp_path)
    dispatched = []
    _fake_runners(monkeypatch, dispatched)
    g = _polygons(env).geometry.iloc[0]
    polys = gpd.GeoDataFrame({
        "fid": ["june", "july"], "crop": [0, 1],
        "startdate": ["2018-06-01", "2018-07-01"], "enddate": ["2018-07-01", "2018-08-01"],
        "geometry": [g, g]}, crs="EPSG:4326")
    td = _train(env, tmp_path, label_polygons=polys, startdate=None, enddate=None)
    out = td.load()
    assert td.bands == ["B08", "vv"] and td.n_timestamps == 1
    assert set(out["ids"]) == {"june", "july"}
    assert set(out["window_starts"]) == {np.datetime64("2018-06-01", "ns"),
                                         np.datetime64("2018-07-01", "ns")}
    assert len(dispatched) == 2

    _train(env, tmp_path, label_polygons=polys, startdate=None, enddate=None)   # satisfied
    assert len(dispatched) == 2
