"""Built-in stages. Importing this package registers all of them in ``STAGES``.

Vendor SDKs are imported inside ``run`` only, so registration works on any machine; a stage
whose SDK is missing reports it during pipeline execution (or is skipped with
``skip_if_unavailable: true``).
"""

from hale_core.deploy.stages import amd, apple, common, nvidia, qualcomm  # noqa: F401

__all__ = ["amd", "apple", "common", "nvidia", "qualcomm"]
