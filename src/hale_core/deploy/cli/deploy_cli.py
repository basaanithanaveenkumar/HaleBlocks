"""``hale-deploy`` command line: run pipelines, inspect configs, list components, probe backends."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

from loguru import logger

from hale_core.deploy.config.config import Config
from hale_core.deploy.core.deploy_runner import DeployRunner
from hale_core.exceptions import HaleCoreError


class DeployCLI:
    """Subcommands:

    * ``run CONFIG [--cfg-options k=v ...] [--work-dir DIR]`` - execute a pipeline
    * ``show CONFIG [--cfg-options ...]`` - print the merged config (after ``_base_``)
    * ``list [--imports MOD ...]`` - registered stages, adapters, runtimes, calibration sources
    * ``env`` - which vendor backends this machine can run

    Exit codes: 0 success, 1 pipeline/stage failure, 2 configuration error.
    """

    def __init__(self, prog: str = "hale-deploy") -> None:
        self.parser = self._build_parser(prog)

    @staticmethod
    def _build_parser(prog: str) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog=prog, description="Export Hale models to TensorRT, MLX, AMD and Qualcomm targets"
        )
        parser.add_argument(
            "--log-level", default="INFO", choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]
        )
        sub = parser.add_subparsers(dest="command", required=True)
        for name in ("run", "show"):
            cmd = sub.add_parser(name)
            cmd.add_argument("config")
            cmd.add_argument("--cfg-options", nargs="*", default=[], metavar="KEY=VALUE")
            if name == "run":
                cmd.add_argument("--work-dir", default=None)
        listing = sub.add_parser("list")
        listing.add_argument("--imports", nargs="*", default=[])
        sub.add_parser("env")
        return parser

    def main(self, argv: Sequence[str] | None = None) -> int:
        args = self.parser.parse_args(argv)
        logger.remove()
        logger.add(sys.stderr, level=args.log_level)
        try:
            return getattr(self, f"cmd_{args.command}")(args)
        except HaleCoreError as exc:
            logger.error("{}", exc)
            from hale_core.deploy.exceptions import ConfigError, RegistryError

            return 2 if isinstance(exc, (ConfigError, RegistryError)) else 1

    @staticmethod
    def cmd_run(args) -> int:
        report = DeployRunner.from_file(args.config, args.cfg_options, args.work_dir).run()
        return 0 if report.ok else 1

    @staticmethod
    def cmd_show(args) -> int:
        print(Config.fromfile(args.config, args.cfg_options).pretty())
        return 0

    @staticmethod
    def cmd_list(args) -> int:
        from hale_core.deploy.registry import ADAPTERS, CALIBRATION, RUNTIMES, STAGES

        for module in (
            "hale_core.deploy.stages",
            "hale_core.deploy.runtimes",
            "hale_core.deploy.calibration",
            "hale_core.deploy.adapters",
            *args.imports,
        ):
            importlib.import_module(module)
        for registry in (STAGES, ADAPTERS, RUNTIMES, CALIBRATION):
            print(f"{registry.name}s:")
            for key in registry.qualified_keys():  # includes downstream scopes (hale_llm.X ...)
                print(f"  {key}")
        return 0

    @staticmethod
    def cmd_env(args) -> int:
        from hale_core.deploy.environment import EnvironmentChecker

        checker = EnvironmentChecker()
        for key, value in checker.accelerators().items():
            print(f"{key:>10}: {value}")
        print()
        for status in checker.check():
            mark = "✓" if status.available else "·"
            print(
                f"  {mark} {status.vendor:<9} {status.backend:<16} "
                f"{status.version or '':<12} {status.detail}"
            )
        return 0


def main(argv: Sequence[str] | None = None) -> int:
    return DeployCLI().main(argv)
