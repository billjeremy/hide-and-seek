"""Lightweight shim: re-export top-level `file_distribution_refactored` module under `graph`.
"""
from warnings import warn
warn("graph.file_distribution_refactored is a shim re-exporting top-level 'file_distribution_refactored'", DeprecationWarning)

from file_distribution_refactored import *
