"""Built-in collators and DataLoader factories — registered in their registries.

Import as a side effect to activate all registrations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.data import DataLoader, Dataset

    from hale_core.config.sections.dataloader import DataLoaderConfig


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------


def _default_collate(batch: list) -> Any:
    """PyTorch's default collate (torch.utils.data.default_collate)."""
    from torch.utils.data.dataloader import default_collate

    return default_collate(batch)


def _pad_sequence_collate(batch: list) -> Any:
    """Pad variable-length 1-D sequences to the longest in the batch."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if isinstance(batch[0], dict):
        keys = batch[0].keys()
        return {
            k: pad_sequence([item[k] for item in batch], batch_first=True)
            if isinstance(batch[0][k], torch.Tensor)
            else [item[k] for item in batch]
            for k in keys
        }
    return pad_sequence(batch, batch_first=True)


# ---------------------------------------------------------------------------
# DataLoader factories
# (signature: factory(dataset, cfg: DataLoaderConfig, batch_size: int, **kwargs))
# ---------------------------------------------------------------------------


def _default_dataloader(
    dataset: Dataset,
    cfg: DataLoaderConfig,
    batch_size: int,
    *,
    collate_fn: Any = None,
) -> DataLoader:
    import torch.distributed as dist
    from torch.utils.data import DataLoader, IterableDataset
    from torch.utils.data.distributed import DistributedSampler

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers and cfg.num_workers > 0,
        "drop_last": cfg.drop_last,
    }
    if isinstance(dataset, IterableDataset):
        # Streams (e.g. WebDataset) shuffle and split across ranks / workers themselves;
        # DataLoader forbids shuffle and samplers for them.
        pass
    elif dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # DDP / Ray: each rank must read a disjoint 1/world_size of the data. The trainer
        # calls sampler.set_epoch(epoch) so the shuffle changes every epoch.
        loader_kwargs["sampler"] = DistributedSampler(
            dataset, shuffle=cfg.shuffle, drop_last=cfg.drop_last
        )
    else:
        loader_kwargs["shuffle"] = cfg.shuffle
    if cfg.prefetch_factor is not None and cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn

    return DataLoader(dataset, **loader_kwargs)


# ---------------------------------------------------------------------------
# Registration (side-effect on import)
# ---------------------------------------------------------------------------


def _register_loaders() -> None:
    from hale_core.registry.plugins import COLLATORS_REGISTRY, DATALOADERS_REGISTRY

    if "default" not in COLLATORS_REGISTRY:
        COLLATORS_REGISTRY.add("default", _default_collate)
        COLLATORS_REGISTRY.add("pad_sequence", _pad_sequence_collate)

    if "default" not in DATALOADERS_REGISTRY:
        DATALOADERS_REGISTRY.add("default", _default_dataloader)


_register_loaders()
