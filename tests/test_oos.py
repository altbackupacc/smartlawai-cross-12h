"""Tests for OOS Platt calibration (PLAN.md M5.6)."""

from __future__ import annotations

import tempfile

from smartlawai.core.oos import PlattScaler, fit_platt_scaling


def test_platt_scaler_sigmoid_monotonic():
    scaler = PlattScaler(a=1.0, b=0.0)
    # Higher logit -> higher in-scope probability
    assert scaler.predict_proba(5.0) > scaler.predict_proba(0.0)
    assert scaler.predict_proba(0.0) > scaler.predict_proba(-5.0)
    assert 0.0 <= scaler.predict_proba(-10.0) <= 1.0


def test_platt_scaler_is_oos():
    scaler = PlattScaler(a=1.0, b=0.0, threshold=0.5)
    assert scaler.is_oos(-2.0) is True
    assert scaler.is_oos(2.0) is False


def test_fit_platt_scaling_separates_classes():
    logits = [3.0, 4.0, 5.0, -3.0, -4.0, -5.0]
    labels = [1, 1, 1, 0, 0, 0]

    scaler = fit_platt_scaling(logits, labels, learning_rate=0.2, max_iter=100)
    assert scaler.predict_proba(4.0) > 0.8
    assert scaler.predict_proba(-4.0) < 0.2


def test_platt_scaler_save_load_roundtrip():
    scaler = PlattScaler(a=2.5, b=-1.2, threshold=0.65)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    scaler.save(path)
    loaded = PlattScaler.load(path)
    assert loaded.a == 2.5
    assert loaded.b == -1.2
    assert loaded.threshold == 0.65
