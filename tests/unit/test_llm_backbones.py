"""Fine-tunable LLM backbones: registry, HF + HaloDiffusionLLM wrappers, LoRA / full / frozen.

Runs offline: Hugging Face architectures are instantiated as tiny random models from their
config classes, and a word-level tokenizer is built in memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from hale_core.config.sections import LLMBackboneConfig  # noqa: E402
from hale_core.nn.backbones import build_backbone  # noqa: E402
from hale_core.nn.llm import (  # noqa: E402
    build_language_backbone,
    configure_finetuning,
    get_llm_backbone_spec,
    list_llm_backbones,
)
from hale_core.nn.llm.hale_backbone import HaleSequenceLMBackbone  # noqa: E402
from hale_core.nn.llm.hf_backbone import HFCausalLMBackbone  # noqa: E402

#: model_type of every registered HF backbone (checked against the Hub when the catalogue was written).
MODEL_TYPES = {
    "smollm2": "llama", "qwen2.5": "qwen2", "qwen3": "qwen3", "llama3": "llama", "gemma2": "gemma2",
    "gemma3": "gemma3_text", "phi3": "phi3", "phi4": "phi3", "mistral": "mistral", "deepseek-r1": "qwen2",
}
VOCAB = [f"w{i}" for i in range(60)]


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {t: i for i, t in enumerate(["<pad>", "<unk>", "<eos>", *VOCAB])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>", eos_token="<eos>"
    )


def tiny_hf_model(model_type: str, vocab: int):
    config = transformers.AutoConfig.for_model(
        model_type, vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(config)


def tiny_hf_backbone(name: str = "qwen3-0.6b") -> HFCausalLMBackbone:
    spec = get_llm_backbone_spec(name)
    tok = tiny_tokenizer()
    return HFCausalLMBackbone(spec, tiny_hf_model(MODEL_TYPES[spec.family], len(tok)), tok)


# ---------------------------------------------------------------------------- registry
def test_registry_holds_only_causal_finetunable_backbones() -> None:
    hf, hale = list_llm_backbones("hf"), list_llm_backbones("hale")
    assert len(hf) == 21 and hale == ["hale-autoregressive", "hale-mtp"]
    assert {get_llm_backbone_spec(n).family for n in hf} == set(MODEL_TYPES)
    assert all(get_llm_backbone_spec(n).params_b <= 8.2 for n in hf)
    assert get_llm_backbone_spec("deepseek-r1-qwen-7b").name == "deepseek-r1-distill-qwen-7b"  # alias
    with pytest.raises(KeyError):
        get_llm_backbone_spec("hale-diffusion")


@pytest.mark.parametrize("name", list_llm_backbones("hf"))
def test_default_lora_targets_exist_in_the_architecture(name: str) -> None:
    spec = get_llm_backbone_spec(name)
    model = tiny_hf_model(MODEL_TYPES[spec.family], 64)
    leaves = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    assert set(spec.lora_targets) <= leaves, set(spec.lora_targets) - leaves


def test_hale_backbone_needs_a_checkpoint() -> None:
    with pytest.raises(ValidationError):
        LLMBackboneConfig(name="hale-autoregressive")


# ---------------------------------------------------------------------------- Hugging Face
def test_hf_inputs_embeds_matches_input_ids_and_gives_hidden_states() -> None:
    llm = tiny_hf_backbone().eval()
    ids = torch.randint(3, 60, (2, 7))
    a = llm(input_ids=ids, labels=ids)
    b = llm(inputs_embeds=llm.embed_tokens(ids), labels=ids)
    assert torch.allclose(a.logits, b.logits, atol=1e-5)
    assert b.hidden_states.shape == (2, 7, llm.hidden_size) and torch.isfinite(b.loss)


@pytest.mark.parametrize("finetune, train_embeddings", [("lora", False), ("lora", True), ("full", False), ("frozen", False)])
def test_hf_finetuning_modes(finetune: str, train_embeddings: bool) -> None:
    llm = tiny_hf_backbone("smollm2-135m").eval()
    ids = llm.add_special_tokens(["<image>", "<halo_action>"])
    assert ids == [llm.tokenizer.convert_tokens_to_ids(t) for t in ("<image>", "<halo_action>")]
    assert llm.vocab_size == len(llm.tokenizer)
    x = torch.randint(3, 60, (2, 6))
    before = llm(input_ids=x).logits
    cfg = LLMBackboneConfig(name="smollm2-135m", finetune=finetune,
                            lora={"r": 4, "alpha": 8, "train_embeddings": train_embeddings})
    configure_finetuning(llm, cfg)
    names = {n for n, p in llm.named_parameters() if p.requires_grad}
    if finetune == "frozen":
        assert not names
    elif finetune == "full":
        assert names == {n for n, _ in llm.named_parameters()}
    else:
        assert names and all("lora_" in n or ("embed" in n or "lm_head" in n) for n in names)
        assert any("embed_tokens" in n for n in names) == train_embeddings
        assert torch.allclose(llm(input_ids=x).logits, before, atol=1e-5)  # LoRA B = 0 at init
        llm.train()
        llm(input_ids=x, labels=x).loss.backward()
        assert all(p.grad is not None for n, p in llm.named_parameters() if "lora_B" in n)


# ---------------------------------------------------------------------------- HaloDiffusionLLM
def write_hale_checkpoint(tmp: Path, variant: str = "autoregressive", **model) -> Path:
    tok = tiny_tokenizer()
    tok.save_pretrained(tmp / "tok")
    model_cfg = dict(d_model=32, n_heads=4, n_layers=2, d_ff=64, max_length=32, dropout=0.0,
                     attn_type="causal", use_time_cond=False, block_size=None, attn_impl="gqa",
                     n_kv_heads=2, ffn_type="geglu", moe_num_experts=4, moe_top_k=2, moe_num_shared=1,
                     arch="transformer", sliding_window=None, local_global_ratio=None,
                     rope_theta_local=None, rope_theta_global=None, p_rope=None, qk_norm=False)
    model_cfg.update(model)
    torch.manual_seed(0)
    from types import SimpleNamespace

    bb = build_backbone(len(tok), SimpleNamespace(**model_cfg), **({"tie_embeddings": False} if variant == "mtp" else {}))
    state = {f"backbone.{k}": v for k, v in bb.state_dict().items()}
    if variant == "mtp":
        state["extra_heads.0.weight"] = torch.zeros(len(tok), 32)
    path = tmp / f"{variant}.pt"
    torch.save({"variant": variant, "vocab_size": len(tok), "model_state_dict": state,
                "config": {"model": model_cfg, "data": {"tokenizer_name": str(tmp / "tok")}}}, path)
    return path, bb


@pytest.mark.parametrize("variant, impl", [("autoregressive", "gqa"), ("autoregressive", "mha"), ("mtp", "gqa")])
def test_hale_checkpoint_round_trip_matches_native_forward(tmp_path: Path, variant: str, impl: str) -> None:
    path, native = write_hale_checkpoint(tmp_path, variant, attn_impl=impl)
    llm, ids = build_language_backbone(
        LLMBackboneConfig(name=f"hale-{variant}", checkpoint=str(path), dtype="float32", finetune="lora",
                          lora={"r": 4, "alpha": 8}),
        special_tokens=["<image>"],
    )
    assert isinstance(llm, HaleSequenceLMBackbone) and ids["<image>"] == llm.vocab_size - 1
    x = torch.randint(3, 60, (2, 9))
    llm.eval(), native.eval()
    out = llm(inputs_embeds=llm.embed_tokens(x), attention_mask=torch.ones_like(x), labels=x)
    # LoRA starts at zero and the old vocabulary rows are untouched: same logits as the checkpoint.
    assert torch.allclose(out.logits[..., : native.vocab_size], native(x), atol=1e-5)
    assert out.hidden_states.shape == (2, 9, 32) and torch.isfinite(out.loss)
    trainable = [n for n, p in llm.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    llm.train()
    out.loss.backward()


def test_hale_rejects_non_causal_checkpoints(tmp_path: Path) -> None:
    path, _ = write_hale_checkpoint(tmp_path, "autoregressive", attn_type="bidirectional")
    with pytest.raises(ValueError, match="causal"):
        build_language_backbone(LLMBackboneConfig(name="hale-autoregressive", checkpoint=str(path)))
    diff = tmp_path / "diff.pt"
    torch.save({"variant": "diffusion"}, diff)
    with pytest.raises(ValueError, match="cannot be a backbone"):
        build_language_backbone(LLMBackboneConfig(name="hale-autoregressive", checkpoint=str(diff)))
