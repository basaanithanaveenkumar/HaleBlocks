"""Encode arbitrary (nested) training samples into WebDataset entries and back."""

from __future__ import annotations

import dataclasses
import enum
import importlib
import io
import json
import sys
from pathlib import PurePath
from typing import Any

import torch
from loguru import logger

from hale_core.exceptions import DataError

#: Name of the entry describing the sample's structure (not a valid field path, see _field_name).
META_ENTRY = "hale.meta.json"
#: v2 adds ``enum`` and ``path`` nodes; v1 shards (a subset) still decode.
CODEC_VERSION = 2
SUPPORTED_VERSIONS = (1, 2)


def _numpy():
    """The numpy module if it is loaded, else ``None`` (numpy is optional for HaleBlocks).

    A sample can only contain numpy objects if numpy was already imported, so there is no need
    to import it here.
    """
    return sys.modules.get("numpy")


class SampleCodec:
    """Lossless, self-describing serialisation of one dataset sample.

    A sample can be a leaf or any nesting of ``dict`` / ``list`` / ``tuple`` / dataclass. Every
    leaf becomes one tar member whose extension says how it was encoded:

    ==========================  =========  =======================================
    leaf type                   extension  encoding
    ==========================  =========  =======================================
    ``torch.Tensor``            ``.pth``   ``torch.save`` (dtype, shape, values)
    ``numpy.ndarray``           ``.npy``   ``numpy.save``
    ``str``                     ``.txt``   UTF-8
    ``bytes``                   ``.bin``   raw
    ``PIL.Image.Image``         ``.png``   lossless PNG
    JSON values (int, float,    ``.json``  JSON
    bool, None, lists/dicts of
    them)
    ==========================  =========  =======================================

    ``enum.Enum`` members and ``pathlib`` paths are small values recorded directly in the
    structure (no tar member): enums are re-created as members of their class, so a
    ``StrEnum`` does not silently come back as a plain ``str``. A path is stored as a path,
    **not** the file it points to - convert media to tensors / PIL images in the dataset if the
    shards must be self-contained.

    A ``hale.meta.json`` entry records the container structure so :meth:`decode` rebuilds the
    exact same Python object (tuples stay tuples, dataclasses are re-instantiated).

    Member names are ``<key>.<field path>.<ext>`` where the field path joins nested keys with
    ``__`` - e.g. ``000042.images.pth``, ``000042.cameras__intrinsics.pth``.
    """

    version = CODEC_VERSION

    # ------------------------------------------------------------------ encoding
    def encode(self, sample: Any, key: str) -> dict[str, Any]:
        """``sample`` -> ``{"__key__": key, "<field>.<ext>": bytes, ..., META_ENTRY: bytes}``."""
        entries: dict[str, Any] = {"__key__": key}
        structure = self._encode_node(sample, (), entries)
        entries[META_ENTRY] = json.dumps({"version": self.version, "structure": structure}).encode()
        return entries

    def _encode_node(self, node: Any, path: tuple[str, ...], entries: dict[str, Any]) -> Any:
        # Before the str / JSON checks: a StrEnum or IntEnum is also a str / int.
        if isinstance(node, enum.Enum):
            cls = type(node)
            return {
                "kind": "enum",
                "class": f"{cls.__module__}:{cls.__qualname__}",
                "name": node.name,
            }
        if isinstance(node, PurePath):
            return {
                "kind": "path",
                "class": f"{type(node).__module__}:{type(node).__qualname__}",
                "value": str(node),
            }
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            cls = type(node)
            fields = {f.name: getattr(node, f.name) for f in dataclasses.fields(node)}
            return {
                "kind": "dataclass",
                "class": f"{cls.__module__}:{cls.__qualname__}",
                "fields": {
                    k: self._encode_node(v, path + (k,), entries) for k, v in fields.items()
                },
            }
        if isinstance(node, dict) and not self._is_json(node):
            if not all(isinstance(k, str) for k in node):
                raise TypeError(
                    f"dict keys must be str to be stored in WebDataset, got {list(node)[:3]}"
                )
            return {
                "kind": "dict",
                "fields": {k: self._encode_node(v, path + (k,), entries) for k, v in node.items()},
            }
        if isinstance(node, (list, tuple)) and not self._is_json(node):
            return {
                "kind": "tuple" if isinstance(node, tuple) else "list",
                "items": [
                    self._encode_node(v, path + (str(i),), entries) for i, v in enumerate(node)
                ],
            }
        ext, payload = self._encode_leaf(node)
        name = f"{self._field_name(path)}.{ext}"
        entries[name] = payload
        return {"kind": "leaf", "entry": name}

    @staticmethod
    def _field_name(path: tuple[str, ...]) -> str:
        if not path:
            return "value"
        for part in path:
            if "." in part or "__" in part:
                raise DataError(f"field name {part!r} may not contain '.' or '__'")
        return "__".join(path)

    def _encode_leaf(self, value: Any) -> tuple[str, bytes]:
        if isinstance(value, torch.Tensor):
            buf = io.BytesIO()
            # contiguous + detached + cpu: what we store is exactly the values, not views or graphs
            torch.save(value.detach().cpu().contiguous(), buf)
            return "pth", buf.getvalue()
        np = _numpy()
        if np is not None and isinstance(value, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, value, allow_pickle=False)
            return "npy", buf.getvalue()
        if isinstance(value, str):
            return "txt", value.encode("utf-8")
        if isinstance(value, (bytes, bytearray)):
            return "bin", bytes(value)
        if np is not None and isinstance(value, np.generic):
            return "json", json.dumps(value.item()).encode()
        if self._is_json(value):
            return "json", json.dumps(value).encode()
        try:
            from PIL import Image

            if isinstance(value, Image.Image):
                buf = io.BytesIO()
                value.save(buf, format="PNG")
                return "png", buf.getvalue()
        except ImportError:  # pragma: no cover - pillow is optional
            logger.debug("Pillow not installed; image values cannot be PNG-encoded")
        raise TypeError(f"cannot store a {type(value).__name__} in WebDataset (no codec)")

    @staticmethod
    def _is_json(value: Any) -> bool:
        """True for values JSON represents exactly (not tuples: they would come back as lists,
        not enums: they would come back as their plain values)."""
        if isinstance(value, enum.Enum):
            return False
        if value is None or isinstance(value, (bool, int, float, str)):
            return True
        if isinstance(value, list):
            return all(SampleCodec._is_json(v) for v in value)
        if isinstance(value, dict):
            return all(isinstance(k, str) and SampleCodec._is_json(v) for k, v in value.items())
        return False

    # ------------------------------------------------------------------ decoding
    def decode(self, entries: dict[str, Any]) -> Any:
        """Inverse of :meth:`encode` (accepts the raw dict yielded by ``webdataset``)."""
        meta = json.loads(entries[META_ENTRY])
        if meta.get("version") not in SUPPORTED_VERSIONS:
            raise DataError(
                f"shards were written with codec v{meta.get('version')}, this reads "
                f"v{SUPPORTED_VERSIONS}"
            )
        return self._decode_node(meta["structure"], entries)

    def _decode_node(self, node: dict[str, Any], entries: dict[str, Any]) -> Any:
        kind = node["kind"]
        if kind == "leaf":
            return self._decode_leaf(node["entry"], entries[node["entry"]])
        if kind == "dict":
            return {k: self._decode_node(v, entries) for k, v in node["fields"].items()}
        if kind in ("list", "tuple"):
            items = [self._decode_node(v, entries) for v in node["items"]]
            return tuple(items) if kind == "tuple" else items
        if kind == "dataclass":
            cls = self._import_class(node["class"])
            return cls(**{k: self._decode_node(v, entries) for k, v in node["fields"].items()})
        if kind == "enum":
            return self._import_class(node["class"])[node["name"]]
        if kind == "path":
            return self._import_class(node["class"])(node["value"])
        raise DataError(f"unknown node kind {kind!r}")

    @staticmethod
    def _import_class(reference: str) -> Any:
        """``"package.module:Outer.Inner"`` -> the class."""
        module, qualname = reference.split(":")
        cls: Any = importlib.import_module(module)
        for part in qualname.split("."):
            cls = getattr(cls, part)
        return cls

    @staticmethod
    def _decode_leaf(name: str, payload: bytes) -> Any:
        ext = name.rsplit(".", 1)[-1]
        if ext == "pth":
            return torch.load(io.BytesIO(payload), weights_only=True)
        if ext == "npy":
            import numpy as np

            return np.load(io.BytesIO(payload), allow_pickle=False)
        if ext == "txt":
            return payload.decode("utf-8")
        if ext == "bin":
            return payload
        if ext == "json":
            return json.loads(payload)
        if ext == "png":
            from PIL import Image

            with Image.open(io.BytesIO(payload)) as image:
                return image.copy()  # plain in-memory Image (not a file-backed PngImageFile)
        raise ValueError(f"unknown entry extension {ext!r}")
