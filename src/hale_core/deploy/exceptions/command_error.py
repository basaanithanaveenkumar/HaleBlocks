"""An external tool (trtexec, migraphx-driver, python -m modelopt ...) exited non-zero."""

from __future__ import annotations

from hale_core.deploy.exceptions.deploy_error import DeployError


class CommandError(DeployError):
    """Attributes:
    command: the argv that failed.
    returncode: its exit status.
    output_tail: the last lines of its combined stdout/stderr.
    """

    def __init__(self, command: list[str], returncode: int, output_tail: str) -> None:
        self.command = command
        self.returncode = returncode
        self.output_tail = output_tail
        super().__init__(f"command {' '.join(command)!r} exited with {returncode}:\n{output_tail}")
