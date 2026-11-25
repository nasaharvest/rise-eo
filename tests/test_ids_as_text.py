"""#226: polygon and cell ids stay text across every `input.csv` / shard CSV hop."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_spec214_p3 as p3

from rise_eo.storage import fs
from rise_eo.workflows import create_datacube, flatten, runners, shard
from rise_eo.workflows.unit_csv import read_unit_csv

IDS = ["001", "x19", "1e5", "007"]


def test_read_unit_csv_keeps_the_id_column_and_any_named_id_columns_as_text(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"id": IDS, "fid": IDS, "num": ["001", "002", "003", "004"], "n": range(4)}).to_csv(path, index=False)
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


def _known_empty(tmp_path):
    manifest = tmp_path / "export" / "run" / "_manifest.json"
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
        with fs.open(input_csv, "r") as f:
            df = read_unit_csv(f, id_col)
        p3.flatten_lib.flatten(df, filepath_col, id_col, export, label_col)

    monkeypatch.setattr(runners, "run_local", fake_run_local)
    monkeypatch.setattr(runners, "run_aml", fake_run_aml)
    monkeypatch.setattr(runners, "run_aml_flatten", fake_run_aml_flatten)
    kw = {}
    if runner == "aml":
        kw = dict(runner="aml", runner_kwargs=dict(
            root=str(tmp_path / "blob"), cluster="c", environment="e", identity_client_id="i"))

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
    assert _known_empty(tmp_path) == set()


def test_build_shortfall_only_does_not_record_a_built_zero_padded_cube_as_empty(tmp_path, monkeypatch):
    cat = p3._archive(tmp_path, monkeypatch)
    monkeypatch.setattr(runners, "run_local", lambda csv, **kw: p3._run_rows(csv, []))
    _call_scalar(tmp_path, ["001"], cat)
    _call_scalar(tmp_path, ["001", "002"], cat)
    assert _known_empty(tmp_path) == set()
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
