"""Exception hierarchy shared by HaleBlocks and every Hale* package.

.. code-block:: text

    HaleCoreError
    ├── ConfigError              (also ValueError)
    ├── DataError                (also ValueError)
    ├── ExportError              (also RuntimeError)
    ├── RegistryError            (also KeyError, ValueError)
    ├── MissingDependencyError   (also ImportError)
    └── DeployError              (hale_core.deploy; Stage/Pipeline/Command/Conversion/Parity errors)
"""

from hale_core.exceptions.config_error import ConfigError
from hale_core.exceptions.data_error import DataError
from hale_core.exceptions.export_error import ExportError
from hale_core.exceptions.hale_core_error import HaleCoreError
from hale_core.exceptions.missing_dependency_error import MissingDependencyError
from hale_core.exceptions.registry_error import RegistryError

__all__ = [
    "ConfigError",
    "DataError",
    "ExportError",
    "HaleCoreError",
    "MissingDependencyError",
    "RegistryError",
]
