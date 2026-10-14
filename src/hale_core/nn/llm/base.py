"""The single interface the VLM and VLA use to talk to any language backbone."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hale_core.nn.llm.spec import LLMBackboneSpec


@dataclass
class LanguageBackboneOutput:
    """Result of :meth:`LanguageBackbone.forward`.

    Attributes:
        logits: ``(B, L, V)`` next-token logits.
        hidden_states: ``(B, L, H)`` final (normalised) hidden states - what action / video
            heads of a VLA read at their query-token positions.
        loss: shifted causal-LM cross-entropy when ``labels`` were given (``-100`` ignored).
    """

    logits: Tensor
    hidden_states: Tensor
    loss: Tensor | None = None


class LanguageBackbone(nn.Module, ABC):
    """A causal language model used as the decoder of a VLM / VLA.

    Multimodal models never call the LM with token ids alone: they embed the text with
    :meth:`embed_tokens`, splice in image / state embeddings, and call :meth:`forward` with
    ``inputs_embeds``. Implementations must therefore accept embeddings directly and return
    the final hidden states as well as logits.

    Args:
        spec: registry entry this backbone was built from.
        tokenizer: the tokenizer matching the backbone's vocabulary.
    """

    def __init__(self, spec: LLMBackboneSpec, tokenizer: Any) -> None:
        super().__init__()
        self.spec = spec
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ geometry
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Width of the hidden states / input embeddings."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Rows of the input embedding table."""

    # ------------------------------------------------------------------ compute
    @abstractmethod
    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        """``(B, L)`` ids -> ``(B, L, H)`` input embeddings (no positions added)."""

    @abstractmethod
    def forward(
        self,
        *,
        inputs_embeds: Tensor | None = None,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> LanguageBackboneOutput:
        """Run the decoder causally over ``inputs_embeds`` (or ``input_ids``)."""

    # ------------------------------------------------------------------ vocabulary
    @abstractmethod
    def resize_token_embeddings(self, num_tokens: int) -> None:
        """Grow (or shrink) the input embeddings and LM head to ``num_tokens`` rows."""

    def add_special_tokens(self, tokens: Sequence[str]) -> list[int]:
        """Add ``tokens`` to the tokenizer (if missing), resize the embeddings, return their ids.

        New rows are initialised to the mean of the existing embeddings (Hewitt, 2021: keeps
        the new tokens' logits in-distribution instead of random outliers).
        """
        missing = [t for t in tokens if t not in self.tokenizer.get_vocab()]
        if missing:
            self.tokenizer.add_special_tokens({"additional_special_tokens": missing})
        if len(self.tokenizer) > self.vocab_size:
            self.resize_token_embeddings(len(self.tokenizer))
        return [int(self.tokenizer.convert_tokens_to_ids(t)) for t in tokens]

    # ------------------------------------------------------------------ fine-tuning
    @abstractmethod
    def default_lora_targets(self) -> list[str]:
        """Module names LoRA adapts when the config does not list them."""

    @abstractmethod
    def apply_lora(self, r: int, alpha: int, dropout: float, targets: Sequence[str]) -> None:
        """Insert LoRA adapters into ``targets`` and freeze every other weight."""

    @abstractmethod
    def embedding_parameters(self) -> list[nn.Parameter]:
        """Input embedding and LM-head weights (trained with ``lora.train_embeddings``)."""

    def parameter_counts(self) -> tuple[int, int]:
        """``(trainable, total)`` parameter counts (shared tensors counted once)."""
        seen: set[int] = set()
        trainable = total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
            trainable += p.numel() if p.requires_grad else 0
        return trainable, total

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def causal_lm_loss(logits: Tensor, labels: Tensor) -> Tensor:
        """Cross-entropy of position ``t`` predicting ``labels[t + 1]`` (``-100`` ignored)."""
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    @staticmethod
    def _grown_rows(weight: Tensor, num_tokens: int) -> Tensor:
        """``weight`` resized to ``num_tokens`` rows; new rows = mean of the old ones."""
        old = weight.size(0)
        if num_tokens <= old:
            return weight[:num_tokens].clone()
        extra = weight.mean(0, keepdim=True).expand(num_tokens - old, -1)
        return torch.cat([weight, extra], 0)
