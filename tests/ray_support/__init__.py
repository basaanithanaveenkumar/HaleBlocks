"""Importable helpers for the Ray tests.

Ray workers are separate processes that cannot import pytest test modules or ``conftest``; this
package is shipped to them with ``runtime_env={"py_modules": [<this dir>]}``.
"""
