"""Fixtures shared by every test module."""

from __future__ import annotations

import os

import fsspec
import pytest


@pytest.fixture(autouse=True)
def clean_fsspec_conf():
    """Snapshot and restore `fsspec.config.conf` and `FSSPEC_ABFSS_ANON` around every test."""
    before = {k: dict(v) for k, v in fsspec.config.conf.items()}
    before_env = os.environ.get("FSSPEC_ABFSS_ANON")
    yield
    fsspec.config.conf.clear()
    fsspec.config.conf.update(before)
    if before_env is None:
        os.environ.pop("FSSPEC_ABFSS_ANON", None)
    else:
        os.environ["FSSPEC_ABFSS_ANON"] = before_env


@pytest.fixture
def stub_lineage(monkeypatch):
    """For tests that fake the flatten or the inference runner: no cube holds the facts
    `lineage.json` reads and the model is no real bundle, so stub the writers."""
    from rise_eo import api

    monkeypatch.setattr(
        api._lineage, "write_training",
        lambda export_folderpath, **kw: fs_write(
            os.path.join(export_folderpath, api._lineage.LINEAGE_NAME), "{}"))
    stub_doc = {"software": {"rise-eo": "0"}, "statement": "stub"}
    monkeypatch.setattr(api._lineage, "write_inference", lambda *a, **kw: stub_doc)
    monkeypatch.setattr(api._lineage, "write_prebuilt", lambda *a, **kw: stub_doc)
    monkeypatch.setattr(api._lineage, "cube_facts", lambda paths: {})
    monkeypatch.setattr(api, "_model_record", lambda *a, **kw: {})


def fs_write(path, text):
    from rise_eo.storage import fs

    fs.write_text(path, text)
