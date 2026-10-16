"""``NamedRegistry``: the flat name → object table used by the training-side plugin registries."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from loguru import logger

from hale_core.exceptions import RegistryError
from hale_core.registry.registry import Registry


class NamedRegistry(Registry):
    """A :class:`Registry` with the original ``add``/``items`` interface.

    The plugin registries (``MODELS``, ``LOSSES``, …) and many downstream packages register
    with ``add(name, obj)`` and read ``registry.items``. Subclassing the mmdet-style
    :class:`Registry` gives them ``register_module``, ``build(cfg)``, parent scopes and
    :class:`RegistryError` without changing a single call site.

    Differences from :class:`Registry` kept for compatibility:
        * iteration follows **registration order** (the base class sorts);
        * :attr:`items` is the live mapping, so ``COLLATES = REG.items`` sees later additions.

    Args:
        kind: what is registered (``"model"``, ``"loss"`` …), used in messages.
        parent: optional parent registry (lookups fall back to it).
        scope: package scope of this registry.
    """

    def __init__(self, kind: str, parent: Registry | None = None, scope: str = "hale_core") -> None:
        super().__init__(kind, parent=parent, scope=scope)
        self.kind = kind

    def add(self, name: str, obj: Any) -> Any:
        """Register ``obj`` under ``name`` and return it unchanged.

        Raises:
            RegistryError: if ``name`` is already registered (also a ``ValueError``).
        """
        if name in self._module_dict:
            logger.error("duplicate {} registration {}", self.kind, name)
            raise RegistryError(
                f"{self.kind} {name!r} already registered: {self._module_dict[name]}"
            )
        self._module_dict[name] = obj
        logger.debug("registered {} {}", self.kind, name)
        return obj

    def get(self, key: str) -> Any:
        """Resolve ``key`` (locally, then parents, then ``scope.Name``).

        Raises:
            RegistryError: for an unknown key (also a ``KeyError``), listing what is registered.
        """
        try:
            return super().get(key)
        except RegistryError as err:
            logger.error("unknown {} {!r}; registered={}", self.kind, key, sorted(self.keys()))
            raise RegistryError(
                f"unknown {self.kind} {key!r}; registered={sorted(self.keys())}"
            ) from err

    def __contains__(self, key: str) -> bool:
        # Membership tests are not errors: bypass the logging in :meth:`get`.
        try:
            Registry.get(self, key)
        except RegistryError:
            return False
        return True

    @property
    def items(self) -> dict[str, Any]:
        """The live ``name -> object`` mapping (mutations are visible)."""
        return self._module_dict

    def __iter__(self) -> Iterator[str]:
        return iter(self._module_dict)
