"""Registry of fine-tunable language-model backbones."""

from __future__ import annotations

from hale_core.nn.llm.spec import LLMBackboneSpec
from hale_core.registry import NamedRegistry

LLM_BACKBONES = NamedRegistry("llm_backbone")
#: Old names kept working (Hale-VLM configs written before the shared registry).
_ALIASES: dict[str, str] = {"deepseek-r1-qwen-7b": "deepseek-r1-distill-qwen-7b"}


def register_llm_backbone(spec: LLMBackboneSpec) -> LLMBackboneSpec:
    """Add ``spec`` under ``spec.name`` (raises on duplicates)."""
    return LLM_BACKBONES.add(spec.name, spec)


def get_llm_backbone_spec(name: str) -> LLMBackboneSpec:
    """The spec registered as ``name`` (or one of its aliases)."""
    _ensure_catalog()
    return LLM_BACKBONES.get(_ALIASES.get(name, name))


def list_llm_backbones(source: str | None = None) -> list[str]:
    """Registered names, optionally only ``hf`` or ``hale`` backbones."""
    _ensure_catalog()
    return sorted(n for n in LLM_BACKBONES if source is None or LLM_BACKBONES.get(n).source == source)


def is_registered(name: str) -> bool:
    _ensure_catalog()
    return _ALIASES.get(name, name) in LLM_BACKBONES


def _ensure_catalog() -> None:
    # Lazy: the catalogue registers on first lookup, avoiding import-order coupling.
    from hale_core.nn.llm import catalog  # noqa: F401
