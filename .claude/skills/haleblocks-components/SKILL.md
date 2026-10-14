---
name: haleblocks-components
description: Add or modify a HaleBlocks neural component — attention variant, FFN/MoE, norm, positional scheme, transformer stack, backbone, token loss, optimizer or logger — through the registry/factory pattern, with tests. Use when implementing a new layer type or architecture (e.g. MLA, linear attention, a new MoE router, a new stack) in hale_core.
---

# Adding components

Every component family has a `NamedRegistry` plus a `build_*` factory. Register, extend
the factory's kwargs if needed, expose any config field, and test.

## Attention

```python
# src/hale_core/nn/layers/attention/my_attn.py
class MyAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, **kw): ...
    def forward(self, x, attn_mask=None, key_padding_mask=None, positions=None): ...
```

- Accept **exactly** `(x, attn_mask, key_padding_mask, positions)`. `TransformerLayer`
  and `DiTBlock` call attention with these keyword arguments.
- Masks: bool, `True` = blocked. `attn_mask` is `[T, T]` (or `[2n, 2n]` for block-causal);
  `key_padding_mask` is `[B, T]`.
- Register in `attention/factory.py` (`register_attention("my")(MyAttention)`) and add a
  branch in `build_attention`. Add `"my"` to `ModelConfig.attn_impl`'s `Literal`.

## FFN / MoE

Signature `(d_model, d_ff, dropout, ...)`, forward `[B, T, D] -> [B, T, D]`. Register in
`ffn/factory.py` and extend `ModelConfig.ffn_type`. The built-in `DeepseekMoE` has
`n_shared` SwiGLU experts plus `n_experts` routed experts with a noisy top-k router and **no
auxiliary load-balancing loss**. If you add one, return it through a side channel
(for example a module attribute that the trainer reads), not by changing the forward
signature.

## Stack / backbone

- A **stack** maps hidden states to hidden states (`forward(h, t_emb, attn_mask,
  key_padding_mask, positions)`).
- A **backbone** subclasses `SequenceBackbone`, sets `self.stack` and `self.lm_head`, and
  calls `self._tie_head(self.lm_head)`.
- Register it in `nn/backbones/factory.py::_register_builtins` and add the name to
  `ModelConfig.arch`. If it needs new constructor arguments, add them to
  `backbone_kwargs` and to `ModelConfig`.

## Token loss

```python
from hale_core.nn.losses import register_token_loss
@register_token_loss("my_loss")
def my_loss(logits, targets, *, ignore_index=-100, **kw) -> torch.Tensor: ...  # per-token or reduced, see reduce.py
```

## Optimizer / logger / trainer

`register_optimizer(name)` on a `BaseOptimizer` subclass implementing
`_make_optimizer(params, config)`; `register_logger(name)`; `register_trainer(name)`
(subclass `training.trainer.Trainer`).

## Tests (required)

Add a unit test next to the family (`tests/unit/test_core_blocks.py`, `test_masks.py`,
`test_backbones.py`, `test_losses.py`): shape, mask correctness (e.g. gradient from future
positions is zero for causal), and a parameter count. If the component should be reachable
from `build_backbone`, add a case to `tests/smoke/test_forward_passes.py`.
