"""Unit tests for ModelMonitor (no Spark required)."""
import pytest


def test_import():
    import dashml
    assert hasattr(dashml, "__version__")


def test_launch_importable():
    from dashml import launch
    assert callable(launch)


def test_main_class_importable():
    from dashml import ModelMonitor
    assert ModelMonitor is not None
