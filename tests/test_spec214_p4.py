"""Spec 214 P4: D4 `CollectionInput` and multi-collection `download` (AC4)."""
from __future__ import annotations

import dataclasses
import datetime
import os

import pytest

import rise_eo
from rise_eo import api
from rise_eo.api import PreflightError

# --- CollectionInput ---------------------------------------------------------------------


def test_collection_input_is_exported_frozen_and_keyword_only_after_bands():
    ci = rise_eo.CollectionInput("sentinel-1-rtc", ["vv"], properties_filter={"a": "b"})
    assert (ci.collection, ci.bands, ci.processing, ci.max_cloudcover) == (
        "sentinel-1-rtc", ["vv"], None, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ci.collection = "x"
    with pytest.raises(TypeError):
        rise_eo.CollectionInput("sentinel-1-rtc", ["vv"], {"a": "b"})
    assert "CollectionInput" in api.__all__


# --- BANDS per collection ----------------------------------------------------------------


def test_builtin_collections_declare_pairwise_disjoint_bands():
    from rise_eo.collections import BUILTIN_MODULES, REGISTRY

    assert {m.COLLECTION_ID for m in BUILTIN_MODULES} == set(REGISTRY)
    owner: dict[str, str] = {}
    for module in BUILTIN_MODULES:
        assert module.BANDS, f"{module.COLLECTION_ID} declares no BANDS"
        for band in module.BANDS:
            assert owner.setdefault(band, module.COLLECTION_ID) == module.COLLECTION_ID, (
                f"band {band!r} is declared by both {owner[band]!r} and {module.COLLECTION_ID!r}")


# --- download(inputs=...) ----------------------------------------------------------------

S2 = rise_eo.CollectionInput("sentinel-2-l2a", ["B04", "B08"], processing=">=05.00",
                             max_cloudcover=30.0)
S1 = rise_eo.CollectionInput("sentinel-1-rtc", ["vv", "vh"],
                             properties_filter={"sat:orbit_state": "descending"})
WINDOW = ("2018-06-01", "2018-06-11")


class _Fake:
    """Records every per-input download of both runners."""

    def __init__(self, monkeypatch):
        from rise_eo.workflows import runners

        self.calls = []
        monkeypatch.setattr(api, "_configure_storage", lambda *a, **kw: None)
        monkeypatch.setattr(api, "_mpc_download", lambda **kw: self.calls.append(("local", kw)))
        monkeypatch.setattr(runners, "run_aml_download",
                            lambda **kw: self.calls.append(("aml", kw)))


def _download(tmp_path, inputs, **kw):
    kw.setdefault("max_tiles", 5)
    return rise_eo.download("roi.geojson", *WINDOW, None, str(tmp_path / "arch"),
                            inputs=inputs, **kw)


AML = {"cluster": "c", "environment": "e:1", "root": "memory://r", "identity_client_id": "id"}


@pytest.mark.parametrize("runner", ["local", "aml"])
def test_ac4_download_two_inputs_forwards_each_inputs_arguments(monkeypatch, tmp_path, runner):
    fake = _Fake(monkeypatch)
    kw = {"runner": runner, "runner_kwargs": AML} if runner == "aml" else {}
    out = _download(tmp_path, [S2, S1], **kw)

    arch = str(tmp_path / "arch")
    assert out == {
        "sentinel-2-l2a": api.archive_catalog_filepath(arch, "sentinel-2-l2a"),
        "sentinel-1-rtc": api.archive_catalog_filepath(arch, "sentinel-1-rtc"),
    }
    assert list(out) == ["sentinel-2-l2a", "sentinel-1-rtc"]
    assert [c[0] for c in fake.calls] == [runner, runner]
    s2, s1 = (c[1] for c in fake.calls)
    assert (s2["collection"], s2["bands"], s2["processing"], s2["max_cloudcover"],
            s2["properties_filter"]) == ("sentinel-2-l2a", ["B04", "B08"], ">=05.00", 30.0, None)
    assert (s1["collection"], s1["bands"], s1["processing"], s1["max_cloudcover"],
            s1["properties_filter"]) == (
        "sentinel-1-rtc", ["vv", "vh"], "latest", None, {"sat:orbit_state": "descending"})
    for call in (s2, s1):
        assert (call["roi"], call["max_tiles"]) == ("roi.geojson", 5)
        assert str(call["startdate"].date()) == WINDOW[0]


def test_download_with_one_input_returns_a_dict(monkeypatch, tmp_path):
    _Fake(monkeypatch)
    out = _download(tmp_path, [S1])
    assert out == {"sentinel-1-rtc": api.archive_catalog_filepath(
        str(tmp_path / "arch"), "sentinel-1-rtc")}


def test_ac4_s1_max_cloudcover_fails_before_the_s2_download(monkeypatch, tmp_path):
    fake = _Fake(monkeypatch)
    bad_s1 = rise_eo.CollectionInput("sentinel-1-rtc", ["vv"], max_cloudcover=10.0)
    with pytest.raises(api.PreflightError, match="sentinel-1-rtc: .*supports_cloud_cover"):
        _download(tmp_path, [S2, bad_s1])
    assert fake.calls == []


def test_second_inputs_other_preflight_errors_also_precede_every_download(monkeypatch, tmp_path):
    fake = _Fake(monkeypatch)
    bad_s1 = rise_eo.CollectionInput("sentinel-1-rtc", ["vv"], processing=">=05.00")
    with pytest.raises(api.PreflightError, match="sentinel-1-rtc"):
        _download(tmp_path, [S2, bad_s1])
    assert fake.calls == []


def test_ac4_repeated_collection_and_band_raise(monkeypatch, tmp_path):
    fake = _Fake(monkeypatch)
    with pytest.raises(api.PreflightError, match="'sentinel-2-l2a' appears more than once"):
        _download(tmp_path, [S2, rise_eo.CollectionInput("sentinel-2-l2a", ["B02"])])
    clash = rise_eo.CollectionInput("sentinel-1-rtc", ["B04"])
    with pytest.raises(api.PreflightError) as exc:
        _download(tmp_path, [S2, clash])
    msg = str(exc.value)
    assert "'B04'" in msg and "'sentinel-2-l2a'" in msg and "'sentinel-1-rtc'" in msg
    assert fake.calls == []


@pytest.mark.parametrize("extra", [
    {"collection": "sentinel-2-l2a"}, {"properties_filter": {"a": "b"}},
    {"processing": "latest"}, {"max_cloudcover": 5.0},
])
def test_inputs_with_a_scalar_argument_raises(monkeypatch, tmp_path, extra):
    fake = _Fake(monkeypatch)
    with pytest.raises(api.PreflightError, match=f"inputs= replaces .*{list(extra)[0]}"):
        _download(tmp_path, [S1], **extra)
    assert fake.calls == []


def test_inputs_with_bands_raises(monkeypatch, tmp_path):
    _Fake(monkeypatch)
    with pytest.raises(api.PreflightError, match="inputs= replaces .*bands"):
        rise_eo.download("roi.geojson", *WINDOW, ["vv"], str(tmp_path), inputs=[S1], max_tiles=5)


@pytest.mark.parametrize("bad", [[], "s2", [("sentinel-2-l2a", ["B04"])]])
def test_inputs_must_be_a_nonempty_list_of_collection_inputs(monkeypatch, tmp_path, bad):
    _Fake(monkeypatch)
    with pytest.raises(api.PreflightError, match="inputs must"):
        _download(tmp_path, bad)


def test_aml_run_ids_are_distinct_per_collection(monkeypatch, tmp_path):
    fake = _Fake(monkeypatch)
    _download(tmp_path, [S2, S1], runner="aml", runner_kwargs={**AML, "run_id": "r1"})
    assert [c[1]["run_id"] for c in fake.calls] == ["r1-s2l2a", "r1-s1rtc"]


def test_scalar_download_is_unchanged(monkeypatch, tmp_path):
    fake = _Fake(monkeypatch)
    out = rise_eo.download("roi.geojson", *WINDOW, ["B04"], str(tmp_path / "a"), max_tiles=5)
    assert out == api.archive_catalog_filepath(str(tmp_path / "a"), "sentinel-2-l2a")
    kw = fake.calls[0][1]
    assert (kw["collection"], kw["processing"], kw["bands"]) == ("sentinel-2-l2a", "latest", ["B04"])
    with pytest.raises(api.PreflightError, match="processing=None"):
        rise_eo.download("roi.geojson", *WINDOW, ["B04"], str(tmp_path / "b"), max_tiles=5,
                         processing=None)


# --- create_training_data(inputs=...) ----------------------------------------------------

Y18 = ("2018-01-01", "2019-01-01")


def _train(tmp_path, export, **kw):
    import test_spec214_p3 as p3

    args = dict(
        label_polygons=p3._polys([("p18", *Y18)]), startdate=None, enddate=None,
        mosaic_days="month", id_col="fid", label_col="crop",
        export_folderpath=str(tmp_path / export),
    )
    args.update(kw)
    return rise_eo.create_training_data(**args)


def test_one_input_trains_exactly_as_the_scalar_arguments(tmp_path, monkeypatch):
    import numpy as np
    import test_spec214_p3 as p3

    from rise_eo.workflows import runners

    cat = p3._archive(tmp_path, monkeypatch)
    monkeypatch.setattr(runners, "run_local",
                        lambda csv, **kw: p3._run_rows(csv, []))
    bands = ["B04", "B08", "SCL"]
    scalar = _train(tmp_path, "scalar", catalog_filepath=cat, bands=bands)
    fused = _train(tmp_path, "inputs", catalog_filepath=None, bands=None,
                   archive_root=str(tmp_path / "a"),
                   inputs=[rise_eo.CollectionInput("sentinel-2-l2a", bands)])

    a, b = scalar.load(), fused.load()
    assert set(a) == set(b)
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    assert (scalar.bands, scalar.n_pixels) == (fused.bands, fused.n_pixels)


def test_download_leg_gets_the_inputs_arguments(tmp_path, monkeypatch):
    class Stop(Exception):
        pass

    seen = []

    def fake_download(**kw):
        seen.append(kw)
        raise Stop

    monkeypatch.setattr(api, "_download_verb", fake_download)
    ci = rise_eo.CollectionInput(
        "sentinel-1-rtc", ["vv", "vh"], properties_filter={"sat:orbit_state": "descending"},
        processing=None)
    for runner in ("local", "aml"):
        seen.clear()
        kw = {"runner": runner, "runner_kwargs": {**AML, "root": str(tmp_path / "blob")}}
        with pytest.raises(Stop):
            _train(tmp_path, f"dl-{runner}", catalog_filepath=None, bands=None,
                   archive_root=str(tmp_path / "arch"), inputs=[ci], download=True,
                   max_tiles=5, **kw)
        (call,) = seen
        assert (call["collection"], call["bands"], call["properties_filter"]) == (
            "sentinel-1-rtc", ["vv", "vh"], {"sat:orbit_state": "descending"})
        assert call["dst_folderpath"] == str(tmp_path / "arch")
        assert (call["runner"], call["processing"]) == (runner, "latest")


def test_training_with_two_inputs_is_not_supported_yet(tmp_path):
    with pytest.raises(PreflightError, match="not supported yet"):
        _train(tmp_path, "x", catalog_filepath=None, bands=None, archive_root=str(tmp_path),
               inputs=[S2, S1])


@pytest.mark.parametrize("extra, match", [
    ({"collection": "sentinel-2-l2a"}, r"inputs= replaces .*collection"),
    ({"properties_filter": {"a": "b"}}, r"inputs= replaces .*properties_filter"),
    ({"processing": "latest"}, r"inputs= replaces .*processing"),
    ({"max_cloudcover": 5.0}, r"inputs= replaces .*max_cloudcover"),
    ({"bands": ["B04"]}, r"inputs= replaces .*bands"),
    ({"catalog_filepath": "x/sentinel-2-l2a/c.parquet"}, r"inputs= replaces .*catalog_filepath"),
    ({"archive_root": None}, "needs archive_root"),
])
def test_training_inputs_exclusions(tmp_path, extra, match):
    args = dict(catalog_filepath=None, bands=None, archive_root=str(tmp_path), inputs=[S1])
    args.update(extra)
    with pytest.raises(PreflightError, match=match):
        _train(tmp_path, "x", **args)


def test_training_archive_root_without_inputs_raises(tmp_path):
    with pytest.raises(PreflightError, match="archive_root= goes with inputs="):
        _train(tmp_path, "x", catalog_filepath="a/sentinel-2-l2a/c.parquet", bands=["B04"],
               archive_root=str(tmp_path))


def test_training_scalar_path_still_needs_a_catalog_filepath(tmp_path):
    with pytest.raises(PreflightError, match="catalog_filepath is required"):
        _train(tmp_path, "x", catalog_filepath=None, bands=["B04"])


# --- run_inference(inputs=...) -----------------------------------------------------------


def _infer(tmp_path, monkeypatch, **kw):
    """ROI-mode inference with the tiler, runner and finalize faked; returns what the per-cell
    `setup` was given."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    import rise_eo.grid as _grid_mod
    from rise_eo.workflows import runners as _runners

    fresh = gpd.GeoDataFrame({"id": ["cell_a"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    given = {}

    def fake_setup(*a, **setup_kw):
        given.update(setup_kw)
        os.makedirs(os.path.dirname(setup_kw["csv_filepath"]), exist_ok=True)
        pd.DataFrame({"id": ["cell_a"], "export_folderpath": ["a"]}).to_csv(
            setup_kw["csv_filepath"], index=False)

    monkeypatch.setattr(_grid_mod, "roi_to_s2_grids", lambda *a, **k: fresh)
    monkeypatch.setattr(api, "_ensure_bundle", lambda *a, **k: "bundle_path")
    monkeypatch.setattr(api, "_configure_storage", lambda *a, **k: None)
    monkeypatch.setattr(api._create_datacube, "setup", fake_setup)

    class _Result:
        returncode = 0

    monkeypatch.setattr(_runners, "run_local_inference", lambda *a, **k: _Result())
    monkeypatch.setattr(api, "_existing_outputs", lambda paths, **k: list(paths))
    monkeypatch.setattr(api, "_finalize_outputs", lambda *a, **k: "DONE")
    from test_api_roi import ROI, _Tiny

    args = dict(
        output_folderpath=str(tmp_path / "out"), roi=ROI,
        startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),
        mosaic_days=20,
    )
    args.update(kw)
    assert rise_eo.run_inference(_Tiny(), **args) == "DONE"
    return given


def test_one_input_infers_exactly_as_the_scalar_arguments(tmp_path, monkeypatch):
    scalar = _infer(tmp_path / "s", monkeypatch, collection="sentinel-2-l2a",
                    catalog_filepath=api.archive_catalog_filepath(str(tmp_path / "arch"),
                                                                  "sentinel-2-l2a"),
                    bands=["B04", "B08"])
    via_inputs = _infer(tmp_path / "i", monkeypatch, archive_root=str(tmp_path / "arch"),
                        inputs=[rise_eo.CollectionInput("sentinel-2-l2a", ["B04", "B08"])])
    keys = ("catalog_filepath", "collection", "bands", "properties_filter", "processing",
            "startdate", "enddate", "mosaic_days")
    assert {k: scalar[k] for k in keys} == {k: via_inputs[k] for k in keys}
    assert via_inputs["catalog_filepath"].endswith("sentinel-2-l2a/catalog.parquet")


def test_inference_inputs_carry_the_inputs_filter_and_processing(tmp_path, monkeypatch):
    got = _infer(tmp_path, monkeypatch, archive_root=str(tmp_path / "arch"),
                 inputs=[rise_eo.CollectionInput(
                     "sentinel-2-l2a", ["B04", "B08"], processing=">=05.00")])
    assert got["processing"] == ">=05.00"


def test_inference_with_two_inputs_is_not_supported_yet(tmp_path, monkeypatch):
    with pytest.raises(PreflightError, match="not supported yet"):
        _infer(tmp_path, monkeypatch, archive_root=str(tmp_path), inputs=[S2, S1])


@pytest.mark.parametrize("extra, match", [
    ({"collection": "sentinel-2-l2a"}, r"inputs= replaces .*collection"),
    ({"bands": ["B04"]}, r"inputs= replaces .*bands"),
    ({"catalog_filepath": "x/sentinel-2-l2a/c.parquet"}, r"inputs= replaces .*catalog_filepath"),
    ({"archive_root": None}, "needs archive_root"),
    ({"inputs": [rise_eo.CollectionInput("sentinel-2-l2a", ["B04"], max_cloudcover=5.0)]},
     "max_cloudcover is a download option"),
])
def test_inference_inputs_exclusions(tmp_path, monkeypatch, extra, match):
    args = dict(archive_root=str(tmp_path), inputs=[rise_eo.CollectionInput(
        "sentinel-2-l2a", ["B04", "B08"])])
    args.update(extra)
    with pytest.raises(PreflightError, match=match):
        _infer(tmp_path, monkeypatch, **args)


def test_inference_archive_root_without_inputs_raises(tmp_path, monkeypatch):
    with pytest.raises(PreflightError, match="archive_root= goes with inputs="):
        _infer(tmp_path, monkeypatch, archive_root=str(tmp_path))


def test_inference_inputs_need_roi_mode(tmp_path):
    from test_api_roi import _Tiny

    with pytest.raises(PreflightError, match="inputs= is for ROI mode"):
        rise_eo.run_inference(_Tiny(), inference_datacubes=["x"], output_folderpath=str(tmp_path),
                              archive_root=str(tmp_path), inputs=[S1])
