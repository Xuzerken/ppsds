"""Shared fixtures and suite-wide helpers.

Marker registration (``slow`` / ``packaging``) lives in ``pyproject.toml`` under
``[tool.pytest.ini_options]``, so it applies however pytest is invoked.

Layout note: this directory has no ``__init__.py``, so pytest's default ``prepend`` import mode
puts it on ``sys.path`` and the helpers (``_systems``, ``_hexjson``, ``_golden``) import as plain
top-level modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _systems import linear_ambiguity_set, nonlinear_ambiguity_set

#: Repository root, resolved from this file rather than from the working directory, so the suite
#: behaves the same no matter where it is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def repo_root():
    """Repository root. Session-scoped because it is a constant, and because the packaging tests
    build the artifacts once for the whole module."""
    return REPO_ROOT


@pytest.fixture
def linear_set():
    """A fresh linear ambiguity set (never shared: the sets keep references to their arrays)."""
    return linear_ambiguity_set()


@pytest.fixture
def nonlinear_set():
    """A fresh nonlinear ambiguity set (a damped pendulum)."""
    return nonlinear_ambiguity_set()
