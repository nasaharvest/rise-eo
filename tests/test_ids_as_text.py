"""#226: polygon and cell ids stay text across every `input.csv` / shard CSV hop."""
from __future__ import annotations

import datetime
import json
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import test_spec214_p3 as p3
from shapely.geometry import box
from test_api_roi import ROI, _Tiny
from test_infer_aml import _FakeMLClient as _InferClient
from test_infer_aml import _write_bundle
from test_scale_runner import _FakeMLClient as _ScaleClient

import rise_eo.grid as grid_mod
from rise_eo import api
from rise_eo.storage import fs
from rise_eo.workflows import create_datacube, flatten, infer_shard, runners, shard
from rise_eo.workflows.unit_csv import read_unit_csv

IDS = ["001", "1e5", "007"]


def test_read_unit_csv_keeps_the_id_column_and_any_named_id_columns_as_text(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"id": IDS, "fid": IDS, "num": ["001", "002", "003"], "n": range(3)}).to_csv(path, index=False)
    with open(path) as f:
        df = read_unit_csv(f, "fid")
    assert list(df["id"]) == IDS and list(df["fid"]) == IDS
    assert df["n"].dtype.kind == "i"


def test_flatten_task_reads_a_custom_id_column_as_text(tmp_path, monkeypatch):
    csv = tmp_path / "in.csv"
    pd.DataFrame({"fid": ["001", "007"], "datacube_filepath": ["a", "b"]}).to_csv(csv, index=False)
    seen = {}
    monkeypatch.setattr(flatten._flatten, "flatten", lambda **kw: seen.update(kw))
    flatten.run(
        input_csv=str(csv), export_folderpath=str(tmp_path / "out"),
        status_url=str(tmp_path / "s.json"), id_col="fid",
    )
    assert list(seen["filepaths_df"]["fid"]) == ["001", "007"]


def _scalar_polys(ids):
    gdf = p3._polys([(i, "x", "y") for i in ids]).drop(columns=["startdate", "enddate"])
    return gdf


def _call_scalar(tmp_path, ids, cat, **kw):
    return p3._call(tmp_path, _scalar_polys(ids), catalog_filepath=cat,
                    startdate="2018-01-01", enddate="2019-01-01", **kw)


def _known_empty(run_folder):
    manifest = run_folder / "_manifest.json"
    if not manifest.exists():
        return set()
    entries = json.loads(manifest.read_text()).values()
    return {i for e in entries if isinstance(e, dict) for i in e.get("ids", [])}


@pytest.mark.parametrize("runner", ["local", "aml"])
def test_zero_padded_ids_survive_setup_task_and_flatten(tmp_path, monkeypatch, runner):
    cat = p3._archive(tmp_path, monkeypatch)
    dispatches = []

    def fake_run_local(csv, **kw):
        dispatches.append(csv)
        p3._run_rows(csv, [])
        return SimpleNamespace(returncode=0, stderr="")

    def fake_run_aml(csv, **kw):
        # The AML hop through the real node-side reader: shard CSV -> `shard.run_shard`.
        shard_csv = str(tmp_path / "runs" / "r" / "shards" / "0.csv")
        fs.makedirs(str(tmp_path / "runs" / "r" / "shards"))
        with fs.open(csv, "r") as f:
            read_unit_csv(f).to_csv(shard_csv, index=False)
        monkeypatch.setattr(runners, "run_local", fake_run_local)
        status = shard.run_shard(shard_csv, cores=1)
        assert status["status"] == "ok", status

    def fake_run_aml_flatten(input_csv, export, *, id_col, label_col, filepath_col, nodata, **kw):
        runners._aml_flatten_preflight(
            _ScaleClient(["Completed"]), cluster="c", environment="e", root=str(tmp_path / "pf"),
            input_csv=input_csv, id_col=id_col, label_col=label_col)
        flatten.run(input_csv=input_csv, export_folderpath=export,
                    status_url=str(tmp_path / "flatten_status.json"), id_col=id_col,
                    label_col=label_col, filepath_col=filepath_col, nodata=nodata)

    monkeypatch.setattr(runners, "run_local", fake_run_local)
    monkeypatch.setattr(runners, "run_aml", fake_run_aml)
    monkeypatch.setattr(runners, "run_aml_flatten", fake_run_aml_flatten)
    kw = {}
    if runner == "aml":
        kw = dict(runner="aml", runner_kwargs=dict(
            root=str(tmp_path / "blob"), cluster="c", environment="e", identity_client_id="i"))

    run_folder = tmp_path / "blob" / "runs" / "train" if runner == "aml" else tmp_path / "export" / "run"
    _call_scalar(tmp_path, ["001", "007"], cat, **kw)

    ids = fs.load_npy(str(tmp_path / "export" / "ids.npy"))
    assert sorted(set(map(str, ids))) == ["001", "007"]
    cube_dirs = {p.parent.name for p in tmp_path.rglob("datacube.npy")}
    assert {"001", "007"} <= cube_dirs
    assert not {"1", "01"} & cube_dirs

    # A second call adds a polygon: the shortfall path reads `input.csv` back.
    _call_scalar(tmp_path, ["001", "007", "002"], cat, **kw)
    ids = fs.load_npy(str(tmp_path / "export" / "ids.npy"))
    assert sorted(set(map(str, ids))) == ["001", "002", "007"]
    assert _known_empty(run_folder) == set()


def test_build_shortfall_only_does_not_record_a_built_zero_padded_cube_as_empty(tmp_path, monkeypatch):
    cat = p3._archive(tmp_path, monkeypatch)
    monkeypatch.setattr(runners, "run_local", lambda csv, **kw: p3._run_rows(csv, []))
    _call_scalar(tmp_path, ["001"], cat)
    _call_scalar(tmp_path, ["001", "002"], cat)
    assert _known_empty(tmp_path / "export" / "run") == set()
    assert np.isin(fs.load_npy(str(tmp_path / "export" / "ids.npy")), ["001", "002"]).all()


def test_the_build_readers_in_create_datacube_keep_ids_as_text(tmp_path):
    cube = tmp_path / "win" / "001"
    cube.mkdir(parents=True)
    (cube / "datacube.npy").write_bytes(b"x")
    (cube / "metadata.pickle.npy").write_bytes(b"x")
    csv = tmp_path / "input.csv"
    pd.DataFrame({"id": ["001", "007"],
                  "datacube_filepath": [str(cube / "datacube.npy"),
                                        str(tmp_path / "win" / "007" / "datacube.npy")]}
                 ).to_csv(csv, index=False)
    _, existing_ids = create_datacube._current_rows(str(csv), keep=lambda r: True)
    assert existing_ids == {"001", "007"}
    shortfall, n_total, n_missing = create_datacube._build_shortfall(str(csv), force=False)
    assert (n_total, n_missing) == (2, 1) and shortfall.endswith(".shortfall.csv")
    with open(shortfall) as f:
        assert list(pd.read_csv(f, dtype=str)["id"]) == ["007"]


# --- inference and runner readers ----------------------------------------------------


@pytest.fixture
def fake_aml_command(monkeypatch):
    monkeypatch.setattr(runners, "_import_aml_command", lambda: (lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setattr(runners, "_import_command_job_limits",
                        lambda: (lambda timeout: SimpleNamespace(timeout=timeout)))


def _units(prefix="memory://ids"):
    return pd.DataFrame({
        "id": ["001", "007"], "export_folderpath": [f"{prefix}/001", f"{prefix}/007"],
        "shapefilepath": ["a.geojson", "b.geojson"], "catalog_filepath": ["c", "c"],
        "startdate": ["2018-06-01"] * 2, "enddate": ["2018-06-11"] * 2,
        "mosaic_days": [20, 20], "mosaic_scheme": ["calendar"] * 2,
        "declaration_filepath": ["d"] * 2, "bands": ["B04,B08"] * 2,
    })


def _shard_ids(root, run_id):
    out = []
    for k in range(2):
        url = f"{root}/runs/{run_id}/shards/{k}.csv"
        if fs.exists(url):
            with fs.open(url, "r") as f:
                out += list(pd.read_csv(f, dtype=str)["id"])
    return sorted(out)


def test_run_aml_writes_zero_padded_ids_into_the_shard_csvs(fake_aml_command):
    with fs.open("memory://ids_aml/input.csv", "w") as f:
        _units().to_csv(f, index=False)
    runners.run_aml(
        "memory://ids_aml/input.csv", cluster="c", environment="e:1", root="memory://ids_aml/root",
        identity_client_id="x", n_shards=2, ml_client=_ScaleClient(["Completed", "Completed"]),
        run_id="r",
    )
    assert _shard_ids("memory://ids_aml/root", "r") == ["001", "007"]


def test_run_aml_inference_writes_zero_padded_ids_into_the_shard_csvs(tmp_path, fake_aml_command):
    bundle_dir = _write_bundle(tmp_path)
    with fs.open("memory://ids_inf/input.csv", "w") as f:
        _units().to_csv(f, index=False)
    runners.run_aml_inference(
        "memory://ids_inf/input.csv", bundle_dir, cluster="c", environment="e:1",
        root="memory://ids_inf/root", identity_client_id="x", n_shards=2, skip_smoke=True,
        ml_client=_InferClient({"0": "Completed", "1": "Completed"}), run_id="r",
    )
    assert _shard_ids("memory://ids_inf/root", "r") == ["001", "007"]


def test_shard_hands_the_local_runner_zero_padded_ids(monkeypatch):
    url = "memory://ids_shard/runs/r/shards/0.csv"
    with fs.open(url, "w") as f:
        _units().to_csv(f, index=False)
    seen = {}

    def fake_run_local(csv, **kw):
        seen["ids"] = list(pd.read_csv(csv, dtype=str)["id"])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(runners, "run_local", fake_run_local)
    shard.run_shard(url, cores=1)
    assert seen["ids"] == ["001", "007"]


def test_infer_shard_hands_the_local_runner_zero_padded_ids(tmp_path, monkeypatch):
    staged = "memory://ids_ishard/_bundle"
    runners._stage_bundle(_write_bundle(tmp_path), staged)
    url = "memory://ids_ishard/runs/r/shards/0.csv"
    with fs.open(url, "w") as f:
        _units().to_csv(f, index=False)
    seen = {}

    def fake_run_local_inference(csv, **kw):
        seen["ids"] = list(pd.read_csv(csv, dtype=str)["id"])
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(runners, "run_local_inference", fake_run_local_inference)
    infer_shard.run_infer_shard(url, staged, cores=1, cubes_per_task=1)
    assert seen["ids"] == ["001", "007"]


def test_resolve_inference_pairs_names_outputs_by_the_zero_padded_id(tmp_path):
    csv = tmp_path / "input.csv"
    pd.DataFrame({"id": ["001", "007"], "datacube_filepath": ["a/datacube.npy", "b/datacube.npy"],
                  "shapefilepath": ["a.geojson", "b.geojson"]}).to_csv(csv, index=False)
    pairs, geometries = api._resolve_inference_pairs(str(csv), str(tmp_path / "out"))
    assert [p[1] for p in pairs] == [str(tmp_path / "out" / i / "output.tif") for i in ("001", "007")]
    assert set(geometries) == {p[1] for p in pairs}


def test_check_resume_identity_sees_no_mismatch_for_zero_padded_ids(tmp_path):
    csv = tmp_path / "input.csv"
    pd.DataFrame({"id": ["001", "007"]}).to_csv(csv, index=False)
    grids = gpd.GeoDataFrame({"id": ["001", "007"]}, geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
                             crs="EPSG:4326")
    api._check_resume_identity(str(csv), grids, str(tmp_path))      # no PreflightError


def test_run_inference_roi_matches_each_cached_row_to_its_zero_padded_cell(tmp_path, monkeypatch):
    fresh = gpd.GeoDataFrame({"id": ["001", "007"]}, geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
                             crs="EPSG:4326")
    monkeypatch.setattr(grid_mod, "roi_to_s2_grids", lambda *a, **kw: fresh)
    monkeypatch.setattr(api, "_ensure_bundle", lambda *a, **kw: "bundle_path")
    monkeypatch.setattr(api._create_datacube, "setup", lambda *a, **kw: None)
    monkeypatch.setattr(runners, "run_local_inference",
                        lambda *a, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(api, "_existing_outputs", lambda paths, **kw: list(paths))
    captured = {}
    monkeypatch.setattr(api, "_finalize_outputs", lambda *a, **kw: captured.update(kw) or "DONE")

    cells = tmp_path / "cells"
    cells.mkdir()
    pd.DataFrame({"id": ["001", "007"], "export_folderpath": ["a", "b"],
                  "shapefilepath": ["a.geojson", "b.geojson"]}).to_csv(cells / "input.csv", index=False)

    api.run_inference(
        _Tiny(), output_folderpath=str(tmp_path), roi=ROI, catalog_filepath="sentinel-2-l2a/c.parquet",
        startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),
        mosaic_days=20, bands=["B04", "B08"],
    )
    assert set(captured["geometries"]) == {"a/output.tif", "b/output.tif"}
