"""Stable fingerprints for "which data would this source produce?"."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def fingerprint(**fields: Any) -> str:
    """SHA-256 of ``fields`` as canonical JSON (sorted keys; non-JSON values via ``str``).

    Pass every setting that changes the *samples* (dataset names, limits, image size, tokenizer
    ...) and nothing that does not (learning rate, batch size), so shards are rebuilt exactly
    when their content would differ.
    """
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
