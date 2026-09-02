"""Bounded on-disk media cache with LRU eviction."""

from __future__ import annotations

import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger


class MediaCache:
    """Store downloaded media with a byte budget; evict least-recently-used files."""

    def __init__(self, root: Path, *, max_bytes: int = 2 * 1024**3) -> None:
        self.root = root.expanduser().resolve()
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._entries: OrderedDict[Path, int] = OrderedDict()
        self.root.mkdir(parents=True, exist_ok=True)

    def _touch(self, path: Path, size: int) -> None:
        if path in self._entries:
            self._entries.move_to_end(path)
            return
        self._entries[path] = size
        self._evict_if_needed()

    def _current_bytes(self) -> int:
        return sum(self._entries.values())

    def _evict_if_needed(self) -> None:
        while self._entries and self._current_bytes() > self.max_bytes:
            old_path, old_size = self._entries.popitem(last=False)
            if old_path.exists():
                old_path.unlink(missing_ok=True)
            logger.debug("evicted cache file {} ({:.1f} MB)", old_path, old_size / 1e6)

    def reserve(self, dataset: str, sample_idx: int, suffix: str) -> Path:
        safe = dataset.replace("/", "_")
        path = self.root / safe / f"{sample_idx:08d}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def store_bytes(self, path: Path, payload: bytes) -> Path:
        path.write_bytes(payload)
        with self._lock:
            self._touch(path, len(payload))
        return path

    def store_file(self, src: Path, dest: Path) -> Path:
        if src.resolve() == dest.resolve():
            size = dest.stat().st_size
        else:
            shutil.copy2(src, dest)
            size = dest.stat().st_size
        with self._lock:
            self._touch(dest, size)
        return dest

    def resolve_local(self, value: str | Path, *, dest: Path) -> Path | None:
        path = Path(value)
        if path.exists():
            return self.store_file(path, dest)
        parsed = urlparse(str(value))
        if parsed.scheme in ("http", "https"):
            return self._download_url(str(value), dest)
        return None

    def _download_url(self, url: str, dest: Path) -> Path:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "URL media download requires httpx; install with: uv add httpx"
            ) from exc

        dest.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            with dest.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        with self._lock:
            self._touch(dest, dest.stat().st_size)
        return dest
