"""One name may map to several implementations, resolved by variant at runtime."""

from __future__ import annotations

from typing import Any

from loguru import logger

from hale_core.exceptions import RegistryError

VariantKey = frozenset[str] | None


class VariantRegistry:
    """Register multiple implementations under one name, scoped by variant.

    Resolution order for ``resolve(name, variant)``:
      1. entry whose variant set contains ``variant``
      2. entry registered with ``variants=None`` (generic fallback)
      3. skip (return ``None``) when nothing matches
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.items: dict[str, list[tuple[VariantKey, type]]] = {}

    def add(
        self,
        name: str,
        cls: type,
        *,
        variants: tuple[str, ...] | None = None,
        attr: str | None = "metric_name",
    ) -> type:
        """Register ``cls`` under ``name`` for ``variants`` (``None`` = generic fallback).

        Args:
            name: metric/component name.
            cls: the implementation.
            variants: training variants it applies to.
            attr: attribute set on ``cls`` to the registered name (``None`` to skip).

        Raises:
            RegistryError: if ``name`` is already registered for the same variants.
        """
        key: VariantKey = frozenset(variants) if variants is not None else None
        entries = self.items.setdefault(name, [])
        for existing, _ in entries:
            if existing == key:
                raise RegistryError(
                    f"{self.kind} {name!r} already registered for variants={variants}"
                )
        entries.append((key, cls))
        if attr is not None:
            setattr(cls, attr, name)
        logger.debug("registered {} {} variants={}", self.kind, name, variants)
        return cls

    def resolve(self, name: str, variant: str) -> type | None:
        """The implementation of ``name`` for ``variant`` (specific, else generic, else ``None``).

        Raises:
            RegistryError: if ``name`` is not registered at all.
        """
        entries = self.items.get(name)
        if entries is None:
            logger.error("unknown {} {!r}; registered={}", self.kind, name, sorted(self.items))
            raise RegistryError(f"unknown {self.kind} {name!r}; registered={sorted(self.items)}")
        specific = [
            cls for variants, cls in entries if variants is not None and variant in variants
        ]
        generic = [cls for variants, cls in entries if variants is None]
        return (specific or generic or [None])[0]

    def instantiate(self, names: list[str], variant: str) -> list[Any]:
        """Instantiate every name that has an implementation for ``variant``; others are skipped."""
        built: list[Any] = []
        for name in names:
            cls = self.resolve(name, variant)
            if cls is None:
                logger.info("skipping {} {} (not defined for variant {})", self.kind, name, variant)
                continue
            built.append(cls())
        return built
