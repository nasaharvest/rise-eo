"""Spec 214 P2: D10 (remove mosaic_scheme="acquisition"), D1 (mosaic_days="month"), D7 (cube paths)."""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from rise_eo import api
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
