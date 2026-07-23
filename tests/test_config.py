"""Config path resolution respects environment variables."""
import os
from importlib import reload


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SPECTRA_OUT_DIR", str(tmp_path / "out"))
    import spectra.config as cfg
    reload(cfg)
    assert str(tmp_path / "out") in str(cfg.Paths().out_dir)
