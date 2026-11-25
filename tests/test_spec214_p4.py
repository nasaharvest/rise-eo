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
