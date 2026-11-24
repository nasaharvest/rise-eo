"""Spec 214 P2: D10 (remove mosaic_scheme="acquisition"), D1 (mosaic_days="month"), D7 (cube paths)."""
from __future__ import annotations

import pytest

from rise_eo.datacube import builder
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
