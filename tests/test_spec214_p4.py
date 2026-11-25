"""Spec 214 P4: D4 `CollectionInput` and multi-collection `download` (AC4)."""
from __future__ import annotations

import dataclasses

import pytest

import rise_eo
from rise_eo import api

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
    from rise_eo.collections import s1_rtc, s2_l2a

    assert set(s2_l2a.BANDS) == set(rise_eo.config.S2L2A_ALL_BANDS)
    assert tuple(s1_rtc.BANDS) == ("vv", "vh")
    assert not set(s2_l2a.BANDS) & set(s1_rtc.BANDS)


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
