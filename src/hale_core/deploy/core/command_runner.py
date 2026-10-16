"""Run vendor command-line tools with streamed, logged output."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path

from loguru import logger

from hale_core.deploy.exceptions import CommandError, MissingDependencyError


class CommandRunner:
    """Execute a command, stream its output to loguru and keep a bounded tail for errors.

    Several vendor workflows are most stable through their CLIs (``python -m
    modelopt.onnx.quantization``, ``migraphx-driver``, ``trtexec``), whose Python APIs change
    between releases. The tail buffer is a fixed-size ``deque`` so a chatty tool cannot grow
    memory without bound.
    """

    def __init__(self, tail_lines: int = 40, timeout_s: float | None = None) -> None:
        self.tail_lines = tail_lines
        self.timeout_s = timeout_s

    @staticmethod
    def which(executable: str, hint: str = "") -> str:
        path = shutil.which(executable)
        if path is None:
            raise MissingDependencyError(executable, hint or f"{executable} must be on PATH")
        return path

    def run(
        self, command: Sequence[str], cwd: Path | None = None, env: Mapping[str, str] | None = None
    ) -> str:
        argv = [str(c) for c in command]
        logger.info("$ {}", " ".join(argv))
        tail: deque[str] = deque(maxlen=self.tail_lines)
        merged_env = {**os.environ, **(env or {})}
        with subprocess.Popen(
            argv,
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                tail.append(line)
                logger.debug("  | {}", line)
            try:
                code = proc.wait(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise CommandError(argv, -9, "timed out\n" + "\n".join(tail)) from None
        if code != 0:
            raise CommandError(argv, code, "\n".join(tail))
        return "\n".join(tail)
