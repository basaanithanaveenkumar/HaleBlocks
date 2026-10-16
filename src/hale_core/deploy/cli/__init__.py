"""Command line interface (``hale-deploy`` / ``python -m hale_core.deploy``)."""

from hale_core.deploy.cli.deploy_cli import DeployCLI, main

__all__ = ["DeployCLI", "main"]
