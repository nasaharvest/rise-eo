"""Spec 214 P6: D8 `lineage.json` and D9 the bundle's training lineage (AC10, AC11)."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest
from test_spec214_p5 import (
    END,
    S1_IN,
    S2_IN,
    START,
    Env,
    _fake_runners,
    _train,
)

import rise_eo
from rise_eo import lineage
from rise_eo.datacube import builder


def _lineage_of(td):
    return lineage.read(os.path.join(td.export_folderpath, lineage.LINEAGE_NAME))


def _feature_double(bands, band_indices):
    return bands * 2, band_indices


@pytest.fixture
def trained(tmp_path, monkeypatch):
    env = Env(tmp_path)
    dispatched: list = []
    _fake_runners(monkeypatch, dispatched)
    return env, _train(env, tmp_path), dispatched


# --- AC10: a training set's lineage ----------------------------------------------------------


def test_a_fused_training_run_writes_lineage_with_steps_in_order(trained):
    env, td, _ = trained
    doc = _lineage_of(td)
    assert doc["lineage_version"] == 1
    assert [s["step"] for s in doc["process_steps"]] == [
        "download", "mask", "resample", "mosaic", "fuse", "flatten"]
    assert doc["sources"]["collections"][0]["collection"] == "sentinel-2-l2a"
    assert doc["sources"]["collections"][1]["collection"] == "sentinel-1-rtc"
    for name in ("sentinel-2-l2a", "sentinel-1-rtc"):
        assert name in doc["statement"]
    assert "T=1" in doc["statement"]
    assert doc["runner"] == "local"


def test_the_features_step_follows_flatten_when_features_are_asked_for(tmp_path, monkeypatch):
    env = Env(tmp_path)
    _fake_runners(monkeypatch)
    td = _train(env, tmp_path, feature_sequence=[(_feature_double, {})])
    steps = _lineage_of(td)["process_steps"]
    assert [s["step"] for s in steps][-2:] == ["flatten", "features"]
    (fn,) = steps[-1]["parameters"]["feature_sequence"]
    assert fn["function"].endswith("_feature_double")
    assert fn["source_sha256"] == lineage.function_record(_feature_double)["source_sha256"]


def test_granule_ids_equal_the_catalog_rows_the_cubes_used(trained):
    env, td, _ = trained
    collections = {c["collection"]: c for c in _lineage_of(td)["sources"]["collections"]}
    s2 = collections["sentinel-2-l2a"]
    # c1 sits in June 2018; the S2 catalog has granules on 06-02 and 06-12 in the window
    assert s2["granule_ids"] == ["s2_0", "s2_1"] and s2["n_granules"] == 2
    assert s2["provider"] == ["mpc"]
    assert collections["sentinel-1-rtc"]["granule_ids"] == ["s1_0", "s1_1"]
    assert collections["sentinel-1-rtc"]["properties_filter"] == {"sat:orbit_state": "descending"}


def test_geometry_is_copied_next_to_the_lineage_with_its_sha256(trained):
    env, td, _ = trained
    doc = _lineage_of(td)
    path = os.path.join(td.export_folderpath, lineage.GEOMETRY_NAME)
    with open(path, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == doc["sources"]["geometry"]["sha256"]
    assert doc["sources"]["geometry"]["n_features"] == 1
    assert doc["sources"]["geometry"]["id_col"] == "fid"


def test_the_request_holds_no_credential_and_no_runner_kwargs_and_paths_as_given(tmp_path, monkeypatch):
    from rise_eo.sources.cdse import CdseCredentials

    env = Env(tmp_path)
    _fake_runners(monkeypatch)
    td = _train(env, tmp_path, runner_kwargs={"cluster": "SECRET-CLUSTER"},
                creds=CdseCredentials(sh_client_id="id-secret", sh_client_secret="pw-secret"))
    with open(os.path.join(td.export_folderpath, lineage.LINEAGE_NAME)) as f:
        text = f.read()
    assert "SECRET-CLUSTER" not in text and "secret" not in text
    request = json.loads(text)["request"]
    assert request["verb"] == "create_training_data"
    assert request["arguments"]["archive_root"] == str(tmp_path / "archive")
    assert "runner_kwargs" not in request["arguments"] and "creds" not in request["arguments"]
    assert request["arguments"]["inputs"][1]["properties_filter"] == {"sat:orbit_state": "descending"}


def test_the_request_run_again_with_only_the_output_folder_changed_gives_the_same_arrays(trained, tmp_path):
    env, td, _ = trained
    request = _lineage_of(td)["request"]["arguments"]
    inputs = [rise_eo.CollectionInput(i["collection"], i["bands"],
                                      properties_filter=i["properties_filter"],
                                      processing=i["processing"]) for i in request["inputs"]]
    again = _train(env, tmp_path, inputs=inputs, export_folderpath=str(tmp_path / "export2"),
                   run_folderpath=str(tmp_path / "run2"))
    a, b = td.load(), again.load()
    for key in ("data", "ids", "coords", "labels"):
        assert np.array_equal(a[key], b[key])


def test_an_edited_feature_body_changes_the_recorded_source_hash():
    def f(bands, band_indices):
        return bands, band_indices

    def g(bands, band_indices):
        return bands + 1, band_indices

    g.__qualname__ = f.__qualname__ = "same_name"
    assert lineage.function_record(f)["source_sha256"] != lineage.function_record(g)["source_sha256"]


# --- A3: code.driver and code.build ------------------------------------------------------------


def test_each_cube_records_its_builder_code(tmp_path, monkeypatch):
    from test_spec214_p5 import _cube, _run_row

    env = Env(tmp_path)
    row = env.setup().iloc[0]
    record = {"rise-eo": "9.9", "commit": "aaa", "dirty": False}
    monkeypatch.setattr(lineage, "code_record", lambda: record)
    _run_row(row)
    _, md = _cube(row["export_folderpath"])
    assert md["code"] == record
    assert [c["code"] for c in md["components"]] == [record, record]


def test_lineage_lists_two_build_commits_and_flags_a_cube_with_none():
    facts = lineage.CubeFacts()
    one = {"rise-eo": "1", "commit": "aaa", "dirty": False}
    two = {"rise-eo": "1", "commit": "bbb", "dirty": False}
    facts.add({"code": one, "components": []})
    facts.add({"code": two, "components": [{"code": one}]})
    assert facts.result()["build_code"] == [one, two]
    facts.add({"components": []})                       # a cube that recorded none
    assert None in facts.result()["build_code"]
    with pytest.raises(ValueError, match="rebuild the cubes"):
        lineage.document(statement="s", runner="local", request={}, sources={}, steps=[],
                         facts=facts.result())


def test_code_record_reads_a_git_checkout_and_a_pip_recorded_commit(monkeypatch):
    lineage.code_record.cache_clear()
    monkeypatch.setattr(lineage, "_checkout_state", lambda: ("abc123", True))
    assert lineage.code_record()["commit"] == "abc123" and lineage.code_record()["dirty"] is True
    lineage.code_record.cache_clear()
    monkeypatch.setattr(lineage, "_checkout_state", lambda: None)
    monkeypatch.setattr(lineage, "_pip_commit", lambda: "def456")
    assert lineage.code_record()["commit"] == "def456"
    lineage.code_record.cache_clear()
    monkeypatch.setattr(lineage, "_pip_commit", lambda: None)
    assert "commit" not in lineage.code_record()
    lineage.code_record.cache_clear()


def test_a_training_set_whose_lineage_was_deleted_reflattens_without_rebuilding_a_cube(trained, tmp_path, monkeypatch):
    env, td, dispatched = trained
    built = []
    real = builder.build_datacube
    monkeypatch.setattr(builder, "build_datacube", lambda **kw: built.append(kw) or real(**kw))
    flattened = []
    from rise_eo.datacube import flatten as flatten_mod
    real_flatten = flatten_mod.flatten
    monkeypatch.setattr(flatten_mod, "flatten", lambda **kw: flattened.append(1) or real_flatten(**kw))

    _train(env, tmp_path)
    assert flattened == []                              # nothing to do

    os.remove(os.path.join(td.export_folderpath, lineage.LINEAGE_NAME))
    _train(env, tmp_path)
    assert flattened == [1] and built == []
    assert os.path.exists(os.path.join(td.export_folderpath, lineage.LINEAGE_NAME))


def test_the_driver_code_is_recorded_beside_the_build_code(trained):
    doc = _lineage_of(trained[1])
    assert doc["code"]["driver"]["rise-eo"] == lineage.code_record()["rise-eo"]
    assert doc["code"]["build"] == [lineage.code_record()]
    assert doc["determinism"]["overlapping_mgrs_tiles"] == {}


# keep START/END/S1_IN/S2_IN referenced for readers of this module
_ = (START, END, S1_IN, S2_IN)
