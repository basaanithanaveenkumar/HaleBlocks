"""Exception hierarchy of :mod:`hale_core.deploy`.

.. code-block:: text

    HaleCoreError                       (hale_core.exceptions)
    ├── ConfigError                     shared: config loading is not deploy-only
    ├── RegistryError                   shared: the registry is not deploy-only
    ├── MissingDependencyError          shared
    └── DeployError
        ├── PipelineError
        ├── StageError
        ├── CommandError
        ├── ConversionError
        └── ParityError

Catch :class:`hale_core.exceptions.HaleCoreError` to handle every expected failure of a
deployment run (the CLI does).
"""

from hale_core.deploy.exceptions.command_error import CommandError
from hale_core.deploy.exceptions.config_error import ConfigError
from hale_core.deploy.exceptions.conversion_error import ConversionError
from hale_core.deploy.exceptions.deploy_error import DeployError
from hale_core.deploy.exceptions.missing_dependency_error import MissingDependencyError
from hale_core.deploy.exceptions.parity_error import ParityError
from hale_core.deploy.exceptions.pipeline_error import PipelineError
from hale_core.deploy.exceptions.registry_error import RegistryError
from hale_core.deploy.exceptions.stage_error import StageError

__all__ = [
    "CommandError",
    "ConfigError",
    "ConversionError",
    "DeployError",
    "MissingDependencyError",
    "ParityError",
    "PipelineError",
    "RegistryError",
    "StageError",
]
