"""Placeholder for model forward-pass smoke tests.

Once the model code is migrated from mymodel/ and esm_rosetta/, add a
MockESM-based forward test (see esm_rosetta/model_ablation.py __main__ for the
pattern) exercising each ablation mode A-H.
"""
import pytest

@pytest.mark.skip(reason="Enable after migrating model code into spectra.models")
def test_forward_smoke():
    ...
