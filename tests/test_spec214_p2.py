"""Spec 214 P2: D10 (remove mosaic_scheme="acquisition"), D1 (mosaic_days="month"), D7 (cube paths)."""
from __future__ import annotations

import datetime
import json
import re

import numpy as np
import pandas as pd
import pytest
from test_workflows import _make_catalog, _two_shapes

from rise_eo import api
from rise_eo import collections as _collections
from rise_eo.catalog import declaration as declaration_module
from rise_eo.catalog.declaration import S2_L2A_DECLARATION
from rise_eo.datacube import builder, ops
from rise_eo.workflows import create_datacube, infer_task, task


def test_d10_task_cli_and_runners_refuse_acquisition(tmp_path):
    args = task._parse_args([
        "s.geojson", "c.parquet", "2018-01-01", "2018-02-01", str(tmp_path / "out"),
        "--bands", "B04", "--mosaic-scheme", "acquisition", "--declaration-filepath", "d.json",
    ])
    with pytest.raises(ValueError, match='"calendar"'):
        task.run_task("s", "c", None, None, str(tmp_path / "out"), bands=["B04"], mosaic_days=10,
                      declaration_filepath="d", mosaic_scheme=args.mosaic_scheme)
    with pytest.raises(ValueError, match="removed"):
        infer_task.run_infer_task("s", "c", None, None, str(tmp_path / "e"), bands=["B04"],
                                  mosaic_days=10, declaration_filepath="d", bundle_path="b",
                                  output_filepath=str(tmp_path / "o.tif"),
                                  mosaic_scheme="acquisition")


def test_d10_build_and_setup_paths_refuse_acquisition(tmp_path):
    with pytest.raises(ValueError, match='"calendar"'):
        builder.build_datacube(None, None, None, None, ["B04"], export_folderpath=str(tmp_path),
                               mosaic_scheme="acquisition")
    with pytest.raises(ValueError, match='"calendar"'):
        create_datacube.setup(
            "c", "timestamp", "s", "id", str(tmp_path), None, None, ["B04"], 10, "x.csv", None,
            mosaic_scheme="acquisition")
    with pytest.raises(ValueError, match='"calendar"'):
        create_datacube.run_create_datacube(
            "c", "timestamp", "s", "id", str(tmp_path), None, None, ["B04"], 10, "x.csv", None, 1,
            mosaic_scheme="acquisition")


# --- D1: mosaic_days="month" ------------------------------------------------------------


def _d(s):
    return datetime.datetime.fromisoformat(s)


def test_ac1_monthly_t_is_12_for_both_year_ends():
    assert api.compute_n_timestamps(_d("2018-01-01"), _d("2019-01-01"), "month") == 12
    assert api.compute_n_timestamps(_d("2018-01-01"), _d("2018-12-31"), "month") == 12
    assert api.compute_n_timestamps(_d("2018-01-01"), _d("2018-12-31"), 20) == 19


def test_ac1_february_window_ends_on_1_march():
    windows = ops.mosaic_windows(pd.Timestamp("2018-01-01", tz="UTC"),
                                 pd.Timestamp("2018-04-01", tz="UTC"), "month")
    assert windows[1] == (pd.Timestamp("2018-02-01", tz="UTC"), pd.Timestamp("2018-03-01", tz="UTC"))


@pytest.mark.parametrize("start", ["2018-01-15", "2018-01-01T06:00:00"])
def test_ac1_month_start_must_be_first_at_midnight(start):
    with pytest.raises(ValueError, match="1st of a month at 00:00 UTC"):
        api.compute_n_timestamps(_d(start), _d("2018-06-01"), "month")
    errs = api._check_window(_d(start), _d("2018-06-01"), "month", ["B04"])
    assert any("1st of a month" in e for e in errs)


@pytest.mark.parametrize("bad", ["week", "Month", 1.5, True, None])
def test_mosaic_days_accepts_only_int_or_month(bad):
    with pytest.raises(ValueError, match='int or "month"'):
        ops.check_mosaic_days(bad)


def test_setup_and_task_refuse_a_bad_month_start_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="1st of a month"):
        create_datacube.setup("missing.parquet", "timestamp", "s", "id", str(tmp_path),
                              _d("2018-01-15"), _d("2018-06-01"), ["B04"], "month", "x.csv", None)
    with pytest.raises(ValueError, match="1st of a month"):
        task.run_task("s", "c", _d("2018-01-15"), _d("2018-06-01"), str(tmp_path / "o"),
                      bands=["B04"], mosaic_days="month", declaration_filepath="d")


def test_a_month_with_no_granules_is_an_all_nodata_slice():
    ts = [pd.Timestamp("2018-01-10", tz="UTC"), pd.Timestamp("2018-03-05", tz="UTC")]
    dc = np.array([10, 30], dtype=np.int32).reshape(2, 1, 1, 1)
    out, md = ops.median_mosaic(
        dc, {"bands": ["B04"], "timestamps": ts}, startdate=_d("2018-01-01"),
        enddate=_d("2018-04-01"), mosaic_days="month", mask_value=0)
    assert out.shape[0] == 3 and out[0, 0, 0, 0] == 10 and out[1, 0, 0, 0] == 0
    assert out[2, 0, 0, 0] == 30
    assert md["timestamps"] == [pd.Timestamp(f"2018-0{m}-01", tz="UTC") for m in (1, 2, 3)]


@pytest.mark.parametrize("value", ["month", 20])
def test_mosaic_days_survives_csv_cli_and_arrives_at_run_task_unchanged(tmp_path, monkeypatch, value):
    rows = [{"id": "a", "export_folderpath": str(tmp_path / "a")}]
    csv = str(tmp_path / "input.csv")
    create_datacube._append_input_rows(
        rows, csv, mosaic_days=value, mosaic_scheme="calendar", collection="c", bands=["B04"],
        properties_filter=None, processing=None)
    shard = str(tmp_path / "shard.csv")           # the AML hop: read, rewrite, read again
    pd.read_csv(csv).to_csv(shard, index=False)
    cell = pd.read_csv(shard).iloc[0]["mosaic_days"]

    # Snakefile hop: the cell goes through the shell as text, then the task CLI.
    got = []
    monkeypatch.setattr(task, "run_task", lambda *a, **kw: got.append(kw["mosaic_days"]))
    task.main(["s", "c", "2018-01-01", "2019-01-01", str(tmp_path / "o"), "--bands", "B04",
               "--mosaic-days", str(cell), "--declaration-filepath", "d"])
    assert got == [value] and type(got[0]) is type(value)

    # infer group hop: the row is read straight from the CSV.
    seen = []
    monkeypatch.setattr(infer_task, "run_infer_task",
                        lambda *a, **kw: seen.append(kw["mosaic_days"]) or "o")
    pd.read_csv(shard).assign(
        shapefilepath="s", catalog_filepath="c", startdate="2018-01-01", enddate="2019-01-01",
        declaration_filepath="d").to_csv(shard, index=False)
    infer_task.run_infer_group(shard, (0, 1), "bundle")
    assert seen == [value] and type(seen[0]) is type(value)


# --- D7: short readable cube paths -------------------------------------------------------




def _segment(mosaic_days="month", bands=("B04",), **kw):
    return create_datacube.window_folder_segment(
        _d("2018-01-01"), _d("2019-01-01"), mosaic_days, bands=list(bands),
        mosaic_scheme="calendar", collection="sentinel-2-l2a", declaration=S2_L2A_DECLARATION, **kw)


def test_ac9_s2_segment_names_collection_dates_period_and_hash():
    assert re.fullmatch(r"s2l2a_20180101-20190101_P1M_[0-9a-f]{8}", _segment("month"))
    assert re.fullmatch(r"s2l2a_20180101-20190101_P10D_[0-9a-f]{8}", _segment(10))
    assert re.fullmatch(r"s1rtc_20180101-20190101_P1M_[0-9a-f]{8}", create_datacube.window_folder_segment(
        _d("2018-01-01"), _d("2019-01-01"), "month", bands=["vv"], mosaic_scheme="calendar",
        collection="sentinel-1-rtc", declaration=_collections.get("sentinel-1-rtc")))


def test_ac9_bands_or_filter_change_the_hash_not_the_readable_part():
    a, b = _segment(bands=("B04",)), _segment(bands=("B04", "B08"))
    c = _segment(properties_filter={"s2:foo": "bar"})
    assert a.rsplit("_", 1)[0] == b.rsplit("_", 1)[0] == c.rsplit("_", 1)[0]
    assert len({a, b, c}) == 3


def test_ac9_params_json_writes_every_parameter(tmp_path):
    cat, shapes = tmp_path / "catalog.parquet", tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _two_shapes(shapes)
    create_datacube.setup(
        catalog_filepath=str(cat), timestamp_col="timestamp", shapefilepath=str(shapes),
        id_col="id", run_folderpath=str(tmp_path / "run"), startdate=_d("2018-01-01"),
        enddate=_d("2019-01-01"), bands=["B04", "SCL"], mosaic_days="month",
        csv_filepath=str(tmp_path / "run" / "input.csv"), label_col=None,
    )
    seg = pd.read_csv(tmp_path / "run" / "input.csv").iloc[0]["export_folderpath"]
    params = json.loads((tmp_path / "run" / seg.split("/")[-2] / "params.json").read_text())
    assert params["collection"] == "sentinel-2-l2a" and params["bands"] == ["B04", "SCL"]
    assert params["mosaic_days"] == "month" and params["mosaic_scheme"] == "calendar"
    assert params["startdate"].startswith("2018-01-01") and params["enddate"].startswith("2019-01-01")
    assert params["properties_filter"] == {} and params["processing"] is None
    assert params["declaration"] == declaration_module.to_json(S2_L2A_DECLARATION)


@pytest.fixture
def _clean_registry(monkeypatch):
    monkeypatch.setattr(_collections, "REGISTRY", dict(_collections.REGISTRY))
    monkeypatch.setattr(_collections, "SHORT_NAMES", dict(_collections.SHORT_NAMES))


@pytest.mark.parametrize("bad", ["S2/x", "s2-x", "", "S2"])
def test_ac9_a_bad_short_name_raises(_clean_registry, bad):
    with pytest.raises(ValueError, match=r"\[a-z0-9\]\+"):
        _collections.register("my-coll", S2_L2A_DECLARATION, short_name=bad)


def test_ac9_fallback_code_and_collision(_clean_registry):
    _collections.register("hls2-s30", S2_L2A_DECLARATION)
    assert _collections.short_name_of("hls2-s30") == "hls2s30"
    with pytest.raises(ValueError, match="already used by 'hls2-s30'"):
        _collections.register("hls2.s30", S2_L2A_DECLARATION)
    with pytest.raises(ValueError, match="already used by 'sentinel-2-l2a'"):
        _collections.register("other", S2_L2A_DECLARATION, short_name="s2l2a")
    assert "other" not in _collections.REGISTRY


def test_ac9_force_reregistering_the_same_id_is_not_a_collision(_clean_registry):
    _collections.register("sentinel-2-l2a", S2_L2A_DECLARATION, force=True, short_name="s2l2a")
    assert _collections.short_name_of("sentinel-2-l2a") == "s2l2a"


def test_segment_differs_when_only_the_start_time_or_period_differs():
    def seg(start, days=10):
        return create_datacube.window_folder_segment(
            _d(start), _d("2018-03-01"), days, bands=["B04"], mosaic_scheme="calendar",
            collection="sentinel-2-l2a", declaration=S2_L2A_DECLARATION)
    a, b = seg("2018-01-01T00:00:00"), seg("2018-01-01T12:00:00")
    assert a.rsplit("_", 1)[0] == b.rsplit("_", 1)[0] and a != b
    assert seg("2018-01-01") != seg("2018-01-01", days=11)


@pytest.mark.parametrize("cell, want", [(20.0, 20), ("20.0", 20), ("20", 20), (" month ", "month")])
def test_parse_mosaic_days_reads_back_what_a_csv_cell_holds(cell, want):
    got = ops.parse_mosaic_days(cell)
    assert got == want and type(got) is type(want)


@pytest.mark.parametrize("bad", [20.5, "20.5", "week"])
def test_parse_mosaic_days_still_rejects_non_whole_numbers(bad):
    with pytest.raises(ValueError):
        ops.parse_mosaic_days(bad)


def test_float_days_from_a_retyped_input_csv_still_arrive_as_int(tmp_path, monkeypatch):
    csv = tmp_path / "input.csv"
    csv.write_text("id,export_folderpath,mosaic_days,bands\na,x,20.0,B04\n")        # what a header-only input.csv hop leaves
    df = pd.read_csv(csv)
    assert df["mosaic_days"].dtype == float
    df.assign(shapefilepath="s", catalog_filepath="c", startdate="2018-01-01",
              enddate="2019-01-01", declaration_filepath="d").to_csv(csv, index=False)
    seen, got = [], []
    monkeypatch.setattr(infer_task, "run_infer_task",
                        lambda *a, **kw: seen.append(kw["mosaic_days"]) or "o")
    infer_task.run_infer_group(str(csv), (0, 1), "bundle")
    monkeypatch.setattr(task, "run_task", lambda *a, **kw: got.append(kw["mosaic_days"]))
    task.main(["s", "c", "2018-01-01", "2019-01-01", "o", "--bands", "B04",
               "--mosaic-days", str(df.iloc[0]["mosaic_days"]), "--declaration-filepath", "d"])
    assert seen == [20] and got == [20] and type(seen[0]) is int and type(got[0]) is int
