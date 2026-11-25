"""Spec 214 P6: D8 `lineage.json` and D9 the bundle's training lineage (AC10, AC11)."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from test_spec214_p5 import (
    END,
    S1_IN,
    S2_IN,
    START,
    Env,
    _fake_runners,
    _polygons,
    _run_row,
    _train,
)

import rise_eo
from rise_eo import api, lineage
from rise_eo.datacube import builder
from rise_eo.model import BaseModelAdapter
from rise_eo.model import bundle as bundle_mod
from rise_eo.storage import fs  # noqa: F401


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
    from test_spec214_p5 import _cube

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



# --- D9 + inference lineage -----------------------------------------------------------------


class _Stub(BaseModelAdapter):
    required_bands = ["B08", "vv"]
    n_timestamps = 1
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["c"]
    feature_sequence = []

    def load(self):
        pass

    def predict(self, X):
        return X[:, 0].astype("uint8")


def _save_bundle(tmp_path, name, training_data=None):
    return bundle_mod.save(_Stub(), {}, str(tmp_path / name), code=False, training_data=training_data,
                           verbose=False)


def _write_output(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint8",
                       crs="EPSG:4326", transform=from_origin(15, 48, 0.01, 0.01), nodata=255) as dst:
        dst.write(np.ones((1, 2, 2), dtype="uint8"))


@pytest.fixture
def infer_env(tmp_path, monkeypatch):
    """A trained fused set, plus the ROI-mode plumbing faked below the per-cell task."""
    import types

    import rise_eo.grid as grid_mod
    from rise_eo.workflows import runners

    env = Env(tmp_path)
    _fake_runners(monkeypatch)
    td = _train(env, tmp_path)
    cell = _polygons(env).rename(columns={"fid": "id"})[["id", "geometry"]]
    monkeypatch.setattr(grid_mod, "roi_to_s2_grids", lambda *a, **kw: cell)
    ran = []

    def run_local_inference(csv, **kw):
        for _, row in pd.read_csv(csv).iterrows():
            _run_row(row)
            _write_output(os.path.join(row["export_folderpath"], "output.tif"))
            ran.append(row["id"])
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runners, "run_local_inference", run_local_inference)
    return env, td, ran


def _infer(env, tmp_path, bundle, out="out", **kw):
    args = dict(roi=_polygons(env), inputs=[S2_IN, S1_IN], archive_root=str(tmp_path / "archive"),
                startdate=START, enddate=END, mosaic_days="month",
                output_folderpath=str(tmp_path / out))
    args.update(kw)
    return rise_eo.run_inference(bundle, **args)


def test_a_bundle_saved_with_training_data_holds_the_training_lineage(infer_env, tmp_path):
    env, td, _ = infer_env
    b = _save_bundle(tmp_path, "b", training_data=td)
    manifest = bundle_mod.read_spec(b)
    assert manifest["training_lineage"] == "lineage.json"
    assert lineage.read(os.path.join(b, "lineage.json")) == _lineage_of(td)
    again = _save_bundle(tmp_path, "b2", training_data=td.export_folderpath)   # a path works too
    assert bundle_mod.read_training_lineage(again)["statement"] == _lineage_of(td)["statement"]
    with pytest.raises(FileNotFoundError, match="create_training_data"):
        _save_bundle(tmp_path, "b3", training_data=str(tmp_path / "nowhere"))


def test_the_registry_digest_covers_the_training_lineage(infer_env, tmp_path):
    from rise_eo.model import registry

    env, td, _ = infer_env
    plain = registry.content_digest(_save_bundle(tmp_path, "plain"))
    with_lineage = registry.content_digest(_save_bundle(tmp_path, "with", training_data=td))
    assert plain != with_lineage


def test_inference_with_the_same_inputs_passes_and_writes_lineage_and_stac(infer_env, tmp_path):
    env, td, ran = infer_env
    bundle = _save_bundle(tmp_path, "b", training_data=td)
    result = _infer(env, tmp_path, bundle)
    assert ran == ["c1"]
    out = tmp_path / "out"
    doc = lineage.read(str(out / "lineage.json"))
    assert [s["step"] for s in doc["process_steps"]] == [
        "download", "mask", "resample", "mosaic", "fuse", "predict"]
    assert doc["model"]["digest"].startswith("sha256:") and doc["model"]["given"] == bundle
    assert doc["training_lineage"] == _lineage_of(td)
    assert doc["sources"]["grid"]["n_cells"] == 1 and doc["sources"]["roi"]["n_features"] == 1
    assert [c["granule_ids"] for c in doc["sources"]["collections"]] == [["s2_0", "s2_1"], ["s1_0", "s1_1"]]
    assert doc["request"]["verb"] == "run_inference"
    assert "output_folderpath" not in doc["request"]["arguments"]
    with open(out / "geometry.geojson", "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == doc["sources"]["roi"]["sha256"]

    collection_path = next((out / "stac").rglob("collection.json"))
    collection = json.loads(collection_path.read_text())
    assert collection["processing:software"]["rise-eo"] == doc["software"]["rise-eo"]
    assert collection["processing:lineage"] == doc["statement"]
    asset = collection["assets"]["lineage"]
    assert asset["roles"] == ["metadata"]
    assert (collection_path.parent / asset["href"]).resolve() == (out / "lineage.json").resolve()
    items = [json.loads(p.read_text()) for p in (out / "stac").rglob("*.json")]
    items = [i for i in items if i.get("type") == "Feature"]
    assert items and all(i["properties"]["processing:lineage"] == doc["statement"] for i in items)
    assert result.stac_catalog_filepath.endswith("catalog.json")


def test_a_different_properties_filter_is_refused_before_any_job_starts(infer_env, tmp_path):
    env, td, ran = infer_env
    bundle = _save_bundle(tmp_path, "b", training_data=td)
    both = rise_eo.CollectionInput("sentinel-1-rtc", ["vv"],
                                   properties_filter={"sat:orbit_state": ["ascending", "descending"]})
    with pytest.raises(rise_eo.PreflightError) as exc:
        _infer(env, tmp_path, bundle, inputs=[S2_IN, both])
    message = str(exc.value)
    assert "properties_filter" in message and "'descending'" in message and "ascending" in message
    assert ran == []


def test_other_differences_name_the_field_and_both_values(infer_env, tmp_path):
    env, td, ran = infer_env
    bundle = _save_bundle(tmp_path, "b", training_data=td)
    vh = rise_eo.CollectionInput("sentinel-1-rtc", ["vh"], properties_filter={"sat:orbit_state": "descending"})
    with pytest.raises(rise_eo.PreflightError, match=r"bands of sentinel-1-rtc.*\['vv'\].*\['vh'\]"):
        _infer(env, tmp_path, bundle, inputs=[S2_IN, vh])
    with pytest.raises(rise_eo.PreflightError, match="collections"):
        _infer(env, tmp_path, bundle, inputs=[S2_IN])
    with pytest.raises(rise_eo.PreflightError, match="period"):
        _infer(env, tmp_path, bundle, mosaic_days=10)
    assert ran == []


def test_a_bundle_without_training_lineage_prints_one_line_and_runs(infer_env, tmp_path, capsys):
    env, td, ran = infer_env
    _infer(env, tmp_path, _save_bundle(tmp_path, "plain"))
    assert ran == ["c1"]
    assert capsys.readouterr().out.count("saved without training lineage") == 1
    assert lineage.read(str(tmp_path / "out" / "lineage.json"))["training_lineage"] is None


def test_inference_on_prebuilt_cubes_prints_one_line_and_writes_a_short_lineage(infer_env, tmp_path,
                                                                            monkeypatch, capsys):
    env, td, _ = infer_env
    row = env.setup(run=str(tmp_path / "prebuilt"), label_col=None).iloc[0]
    _run_row(row)
    cube = os.path.join(row["export_folderpath"], "datacube.npy")
    monkeypatch.setattr(api._engine, "run_local", lambda model, pairs, **kw: [
        _write_output(out) or out for _, out in pairs])
    out = tmp_path / "prebuilt_out"
    api.run_inference(_save_bundle(tmp_path, "b", training_data=td), inference_datacubes=[cube],
                      output_folderpath=str(out))
    assert capsys.readouterr().out.count("pre-built cubes: there is no request") == 1
    doc = lineage.read(str(out / "lineage.json"))
    assert doc["sources"] == {"inference_datacubes": [cube]}
    assert [s["step"] for s in doc["process_steps"]] == ["predict"]
    assert doc["model"]["digest"].startswith("sha256:")
