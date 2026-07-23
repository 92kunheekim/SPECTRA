"""Smoke test: the package and its submodules import cleanly."""
import importlib
import pytest

MODULES = [
    "spectra", "spectra.config",
    "spectra.utils.seed", "spectra.utils.logging",
]

@pytest.mark.parametrize("mod", MODULES)
def test_import(mod):
    importlib.import_module(mod)


def test_seed_returns_value():
    from spectra.utils.seed import seed_everything
    assert seed_everything(123) == 123
