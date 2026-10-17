"""Contract of a pipeline stage (Command pattern + Template Method)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.optional_dependency import OptionalDependency

if TYPE_CHECKING:
    from hale_core.deploy.core.export_context import ExportContext

#: Virtual artifact present from the start: the PyTorch model behind the adapter.
TORCH_MODULE = "torch_module"


class Stage(ABC):
    """One step of a deployment pipeline (export, optimize, quantize, compile, validate ...).

    Class attributes declare the data flow so a pipeline can be validated *before* anything
    runs:

    * ``requires`` - artifact kinds that must exist (produced by an earlier stage),
    * ``provides`` - artifact kinds this stage produces,
    * ``dependencies`` - optional packages it imports.

    Args:
        name: unique name in the pipeline (default: class name); also the output sub-directory.
        enabled: ``False`` keeps the stage in the config but skips it.
        skip_if_unavailable: skip instead of failing when a dependency is missing (useful for
            configs shared between a Mac laptop and a CUDA build machine).
    """

    requires: ClassVar[tuple[str, ...]] = (TORCH_MODULE,)
    provides: ClassVar[tuple[str, ...]] = ()
    dependencies: ClassVar[tuple[OptionalDependency, ...]] = ()

    def __init__(
        self, name: str | None = None, enabled: bool = True, skip_if_unavailable: bool = False
    ) -> None:
        self.name = name or type(self).__name__
        self.enabled = enabled
        self.skip_if_unavailable = skip_if_unavailable

    def missing_dependencies(self) -> list[OptionalDependency]:
        """Optional dependencies of this stage that are not installed."""
        return [dep for dep in self.dependencies if not dep.available]

    @abstractmethod
    def run(self, ctx: ExportContext) -> list[Artifact]:
        """Do the work and return the produced artifacts (the pipeline stores them)."""

    def artifact(self, ctx: ExportContext, kind: str, path, **metadata) -> Artifact:
        """Helper to build an artifact attributed to this stage."""
        from pathlib import Path

        return Artifact(kind=kind, path=Path(path), producer=self.name, metadata=metadata)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, requires={self.requires}, "
            f"provides={self.provides})"
        )
