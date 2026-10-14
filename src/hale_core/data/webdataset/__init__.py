"""Convert any PyTorch dataset to WebDataset tar shards and stream it back - behind one flag.

Pipeline::

    dataset ──WebDatasetConverter──► <root>/<name>/shard-*.tar + manifest.json
                                              │
              WebDatasetSource ◄──────────────┘  (split by rank & worker, shuffle, decode)
                     │
                     ▼
    torch IterableDataset with __len__  ──►  the usual DataLoader / Trainer

:class:`WebDatasetAdapter` ties it together: ``adapter.prepare(dataset, "train")`` returns the
original dataset when ``data.webdataset.enabled`` is false and the streamed WebDataset when it is
true. :class:`SampleCodec` round-trips tensors, numpy arrays, strings, bytes, JSON values, PIL
images and nested dicts / lists / tuples / dataclasses of them.

``webdataset`` is imported lazily (``pip install hale-blocks[webdataset]``).
"""

from hale_core.data.webdataset.adapter import WebDatasetAdapter, maybe_webdataset
from hale_core.data.webdataset.converter import WebDatasetConverter
from hale_core.data.webdataset.manifest import ShardManifest
from hale_core.data.webdataset.sample_codec import SampleCodec
from hale_core.data.webdataset.source import WebDatasetSource

__all__ = [
    "SampleCodec",
    "ShardManifest",
    "WebDatasetAdapter",
    "WebDatasetConverter",
    "WebDatasetSource",
    "maybe_webdataset",
]
