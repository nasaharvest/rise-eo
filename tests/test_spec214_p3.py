"""Spec 214 P3: D2 per-polygon windows in `create_training_data` (AC2)."""
from __future__ import annotations

import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import test_workflows as tw

import rise_eo
from rise_eo import api
from rise_eo.api import PreflightError
from rise_eo.datacube import flatten as flatten_lib
from rise_eo.storage import fs
from rise_eo.workflows import runners, task

BANDS = ["B04", "B08", "SCL"]
Y18 = ("2018-01-01", "2019-01-01")
Y19 = ("2019-01-01", "2020-01-01")
H1_18 = ("2018-01-01", "2018-07-01")


def _polys(rows):
    """`rows` = [(fid, startdate, enddate)] -> polygons inside the synthetic tile."""
    shapes = [tw.TILE_4326.buffer(0).centroid.buffer(1e-5) for _ in rows]
    return gpd.GeoDataFrame(
        {"fid": [r[0] for r in rows], "crop": list(range(len(rows))),
         "startdate": [r[1] for r in rows], "enddate": [r[2] for r in rows],
         "geometry": shapes}, crs="EPSG:4326")


def _call(tmp_path, polygons, **kw):
    args = dict(
        label_polygons=polygons, catalog_filepath=str(tmp_path / "a" / "sentinel-2-l2a" / "c.parquet"),
        startdate=None, enddate=None, mosaic_days="month", bands=BANDS, id_col="fid",
        label_col="crop", export_folderpath=str(tmp_path / "export"),
    )
    args.update(kw)
    return rise_eo.create_training_data(**args)


# --- preflight: inputs and T -----------------------------------------------------------


def test_ac2_t_mismatch_names_ids_under_each_t_before_download_and_catalog(tmp_path, monkeypatch):
    def boom(**kw):
        raise AssertionError("download must not run")

    monkeypatch.setattr(api, "_download_verb", boom)
    polys = _polys([("full", *Y18), ("half", *H1_18), ("full2", *Y19)])
    with pytest.raises(PreflightError) as exc:
        _call(tmp_path, polys, download=True, max_tiles=5)   # the catalog does not exist
    msg = str(exc.value)
    assert "T=12: [full, full2]" in msg and "T=6: [half]" in msg
    assert "does not exist" not in msg                       # no catalog was looked at


def test_ac2_t_mismatch_with_an_empty_catalog_raises_the_same(tmp_path):
    cat = tmp_path / "a" / "sentinel-2-l2a" / "c.parquet"
    cat.parent.mkdir(parents=True)
    cat.write_bytes(b"")
    with pytest.raises(PreflightError, match=r"T=6: \[half\]"):
        _call(tmp_path, _polys([("full", *Y18), ("half", *H1_18)]), catalog_filepath=str(cat))


def test_ac2_scalar_dates_with_date_columns_raise(tmp_path):
    with pytest.raises(PreflightError, match="startdate/enddate were also given"):
        _call(tmp_path, _polys([("a", *Y18)]), startdate=datetime.datetime(2018, 1, 1),
              enddate=datetime.datetime(2019, 1, 1))


def test_one_date_none_and_the_other_set_raises(tmp_path):
    with pytest.raises(PreflightError, match="both be dates, or both None"):
        _call(tmp_path, _polys([("a", *Y18)]), startdate=datetime.datetime(2018, 1, 1))


def test_none_dates_without_the_columns_raise(tmp_path):
    with pytest.raises(PreflightError, match="missing"):
        _call(tmp_path, _polys([("a", *Y18)]).drop(columns=["enddate"]))


@pytest.mark.parametrize("bad", [None, "", "not a date", 2018, float("nan")])
def test_an_empty_or_unparsable_date_raises_naming_the_ids(tmp_path, bad):
    polys = _polys([("good", *Y18), ("bad1", *Y18), ("bad2", *Y18)])
    polys["enddate"] = polys["enddate"].astype(object)
    polys.loc[polys.fid.isin(["bad1", "bad2"]), "enddate"] = bad
    with pytest.raises(PreflightError, match=r"unparsable startdate/enddate for ids: \[bad1, bad2\]"):
        _call(tmp_path, polys)


def test_month_windows_need_a_month_start_in_every_row(tmp_path):
    polys = _polys([("a", *Y18), ("b", "2018-01-15", "2019-01-15")])
    with pytest.raises(PreflightError, match=r"ids \[b\].*1st of a month"):
        _call(tmp_path, polys)


def test_duplicate_ids_across_windows_raise(tmp_path):
    with pytest.raises(PreflightError, match=r"duplicate 'fid' values: \[a\]"):
        _call(tmp_path, _polys([("a", *Y18), ("a", *Y19)]))


def test_parse_window_date():
    assert api._parse_window_date("2018-01-01") == pd.Timestamp("2018-01-01", tz="UTC")
    assert api._parse_window_date(np.datetime64("2018-01-01")) == pd.Timestamp("2018-01-01", tz="UTC")
    assert api._parse_window_date(pd.NaT) is None


# --- end to end: real setup, task, flatten, with per-row dates crossing each hop ----------


def _archive(tmp_path, monkeypatch):
    """One tile, two acquisitions in June 2018 (value 100) and two in June 2019 (value 200)."""
    stamps = [pd.Timestamp(d, tz="UTC") for d in
              ("2018-06-01", "2018-06-11", "2019-06-01", "2019-06-11")]
    monkeypatch.setattr(tw, "TS", stamps)
    for i in range(4):
        d = tmp_path / f"prod{i}"
        d.mkdir()
        for band, val in [("B04", 100 if i < 2 else 200), ("B08", 300), ("SCL", 4)]:
            tw._write_tile(d / f"{band}.tif", val)
    cat = tmp_path / "a" / "sentinel-2-l2a" / "catalog.parquet"
    cat.parent.mkdir(parents=True)
    tw._make_catalog(cat, tmp_path, files="B04.tif,B08.tif,SCL.tif")
    return str(cat)


def _run_rows(csv, seen):
    """Each row crosses the cell -> shell -> CLI hop the Snakefile makes."""
    for _, row in pd.read_csv(csv).iterrows():
        seen.append((row["id"], row["startdate"], row["enddate"], row["mosaic_days"]))
        task.main([
            row["shapefilepath"], row["catalog_filepath"], str(row["startdate"]),
            str(row["enddate"]), row["export_folderpath"], "--bands", row["bands"],
            "--mosaic-days", str(row["mosaic_days"]), "--mosaic-scheme", row["mosaic_scheme"],
            "--declaration-filepath", row["declaration_filepath"],
        ])


def _check_arrays(td):
    out = td.load()
    ids, starts, data = out["ids"], out["window_starts"], out["data"]
    assert data.shape[1] == 12
    assert set(map(str, ids)) == {"p18", "p19"}
    for fid, start, value in (("p18", "2018-01-01", 100), ("p19", "2019-01-01", 200)):
        sel = ids.astype(str) == fid
        assert (starts[sel] == np.datetime64(start, "ns")).all()
        assert (data[sel][:, 5, 0] == value).all()       # June
        assert (data[sel][:, 0, 0] == 0).all()           # an empty month is nodata
    assert td.n_timestamps == 12


@pytest.mark.parametrize("runner", ["local", "aml"])
def test_ac2_two_windows_give_one_training_set_with_window_starts(tmp_path, monkeypatch, runner):
    cat = _archive(tmp_path, monkeypatch)
    dispatches, seen = [], []

    def fake_run_local(csv, **kw):
        dispatches.append(csv)
        _run_rows(csv, seen)

    def fake_run_aml(csv, **kw):
        dispatches.append(csv)
        shard = str(tmp_path / "shard.csv")              # the AML hop: read, rewrite, read
        pd.DataFrame(pd.read_csv(csv).to_dict("records")).to_csv(shard, index=False)
        _run_rows(shard, seen)

    def fake_run_aml_flatten(input_csv, export, *, id_col, label_col, filepath_col, nodata, **kw):
        flatten_lib.flatten(pd.read_csv(input_csv), filepath_col, id_col, export, label_col)

    monkeypatch.setattr(runners, "run_local", fake_run_local)
    monkeypatch.setattr(runners, "run_aml", fake_run_aml)
    monkeypatch.setattr(runners, "run_aml_flatten", fake_run_aml_flatten)
    kw = {}
    if runner == "aml":
        kw = dict(runner="aml", runner_kwargs=dict(
            root=str(tmp_path / "blob"), cluster="c", environment="e", identity_client_id="i"))
    polys = _polys([("p18", *Y18), ("p19", *Y19)])

    td = _call(tmp_path, polys, catalog_filepath=cat, **kw)

    assert len(dispatches) == 1                           # one dispatch for both windows
    assert sorted(seen) == [("p18", "2018-01-01 00:00:00+00:00", "2019-01-01 00:00:00+00:00", "month"),
                            ("p19", "2019-01-01 00:00:00+00:00", "2020-01-01 00:00:00+00:00", "month")]
    _check_arrays(td)

    # a second call is the top-level short-circuit: nothing is dispatched again
    dispatches.clear()
    td2 = _call(tmp_path, polys, catalog_filepath=cat, **kw)
    assert dispatches == [] and td2.n_pixels == td.n_pixels


def test_a_polygon_that_changes_window_moves_only_its_own_cube(tmp_path, monkeypatch):
    cat = _archive(tmp_path, monkeypatch)
    dispatched, built = [], []
    monkeypatch.setattr(runners, "run_local",
                        lambda csv, **kw: (dispatched.append(csv), _run_rows(csv, built)))
    _call(tmp_path, _polys([("p18", *Y18), ("p19", *Y19)]), catalog_filepath=cat)
    ids_before = fs.load_npy(str(tmp_path / "export" / "ids.npy"))

    dispatched.clear()
    built.clear()
    td = _call(tmp_path, _polys([("p18", *Y19), ("p19", *Y19)]), catalog_filepath=cat)
    ids_after = fs.load_npy(str(tmp_path / "export" / "ids.npy"))
    assert len(dispatched) == 1
    assert [row[0] for row in built] == ["p18"]           # p19's cube was not rebuilt
    assert set(ids_before) == set(ids_after) == {"p18", "p19"}
    out = td.load()
    assert (out["window_starts"] == np.datetime64("2019-01-01", "ns")).all()
    merged = pd.read_csv(tmp_path / "export" / "run" / "input.csv")
    assert sorted(merged["id"]) == ["p18", "p19"]


@pytest.mark.parametrize("path", ["coverage_hint", "download", "overwrite_datacubes"])
def test_a_window_with_no_imagery_does_not_stop_the_other(tmp_path, monkeypatch, path):
    cat = _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(runners, "run_local", lambda csv, **kw: _run_rows(csv, []))
    monkeypatch.setattr(api, "_download_verb", lambda **kw: None)
    kw = {"download": True, "max_tiles": 5} if path == "download" else {}
    if path == "overwrite_datacubes":
        kw["overwrite"] = "datacubes"
    polys = _polys([("p18", *Y18), ("none", "2015-01-01", "2016-01-01")])

    td = _call(tmp_path, polys, catalog_filepath=cat, **kw)

    assert set(td.load()["ids"]) == {"p18"}


def test_a_repeat_call_with_an_all_known_empty_window_short_circuits(tmp_path, monkeypatch, capsys):
    cat = _archive(tmp_path, monkeypatch)
    dispatched = []
    monkeypatch.setattr(runners, "run_local",
                        lambda csv, **kw: (dispatched.append(csv), _run_rows(csv, [])))
    monkeypatch.setattr(api, "_download_verb", lambda **kw: None)
    polys = _polys([("p18", *Y18), ("none", "2015-01-01", "2016-01-01")])
    kw = {"download": True, "max_tiles": 5}
    _call(tmp_path, polys, catalog_filepath=cat, **kw)
    dispatched.clear()
    capsys.readouterr()

    _call(tmp_path, polys, catalog_filepath=cat, **kw)

    assert "CURRENT" in capsys.readouterr().out and dispatched == []


def test_load_of_a_set_without_window_starts_says_how_to_fix_it(tmp_path):
    export = tmp_path / "export"
    export.mkdir()
    for name in ("data", "ids", "coords"):
        fs.save_npy(str(export / f"{name}.npy"), np.zeros(1))
    fs.save_npy(str(export / "metadata.pickle.npy"), {"bands": ["B04"]}, allow_pickle=True)
    td = api.TrainingData(str(export), str(tmp_path), 1, 1, ["B04"])
    with pytest.raises(FileNotFoundError, match="Re-run create_training_data"):
        td.load()


def test_a_request_with_no_imagery_in_any_window_raises(tmp_path, monkeypatch):
    cat = _archive(tmp_path, monkeypatch)
    polys = _polys([("a", "2015-01-01", "2016-01-01"), ("b", "2016-01-01", "2017-01-01")])
    with pytest.raises(PreflightError, match="no catalog tiles intersect"):
        _call(tmp_path, polys, catalog_filepath=cat)


def test_zero_padded_ids_are_not_dropped_by_the_per_window_csv_read(tmp_path, monkeypatch):
    cat = _archive(tmp_path, monkeypatch)
    monkeypatch.setattr(runners, "run_local", lambda csv, **kw: _run_rows(csv, []))
    td = _call(tmp_path, _polys([("001", *Y18), ("x19", *Y19)]), catalog_filepath=cat)
    merged = pd.read_csv(tmp_path / "export" / "run" / "input.csv", dtype={"id": str})
    assert sorted(merged["id"]) == ["001", "x19"]
    assert sorted(set(map(str, td.load()["ids"]))) == ["001", "x19"]
    assert sorted(set(fs.load_npy(str(tmp_path / "export" / "ids.npy")).astype(str))) == ["001", "x19"]
