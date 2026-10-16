"""Config files with ``_base_`` inheritance, ``_delete_`` replacement and CLI overrides.

The generic machinery moved to :mod:`hale_core.config` (shared by every Hale* package); it is
re-exported here so existing ``hale_core.deploy.config`` imports keep working.
"""

from hale_core.config.config import Config
from hale_core.config.config_loader import ConfigLoader
from hale_core.config.dict_merger import DictMerger
from hale_core.config.override_parser import OverrideParser
from hale_core.deploy.config.deploy_config import DeployConfig

__all__ = ["Config", "ConfigLoader", "DeployConfig", "DictMerger", "OverrideParser"]
