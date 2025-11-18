"""Lightweight shim: re-export top-level `arbo` module under `graph.arbo`.
This keeps compatibility while avoiding duplication.
"""
from warnings import warn
warn("graph.arbo is a shim re-exporting top-level 'arbo' module", DeprecationWarning)

from arbo import *
