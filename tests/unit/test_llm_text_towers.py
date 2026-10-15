"""Text towers of natively multimodal checkpoints as fine-tunable LLM backbones.

Runs offline: each architecture (Qwen3.5, Gemma 4, Ministral 3, Qwen3-VL / Cosmos-Reason2,
LFM2.5-VL, SmolVLM2) is instantiated as a tiny random ``*ForConditionalGeneration`` from the
configs in ``tests/fixtures/llm_multimodal`` (Hub tiny-random configs with a 128-token vocabulary).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")

from hale_core.config.sections import LLMBackboneConfig  # noqa: E402
from hale_core.nn.llm import configure_finetuning, get_llm_backbone_spec, list_llm_backbones  # noqa: E402
from hale_core.nn.llm.hf_backbone import HFCausalLMBackbone  # noqa: E402
from hale_core.nn.llm.hf_text_tower import TextTowerForCausalLM, extract_text_tower  # noqa: E402

from tests.unit.test_llm_backbones import tiny_tokenizer  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm_multimodal"
#: Registry family -> fixture architecture.
FAMILY_FIXTURE = {
    "qwen3.5": "qwen3_5", "gemma4": "gemma4", "ministral3": "mistral3", "cosmos-reason2": "qwen3_vl",
    "lfm2.5-vl": "lfm2_vl", "smolvlm2": "smolvlm",
}
MULTIMODAL = [n for n in list_llm_backbones("hf") if get_llm_backbone_spec(n).is_multimodal_checkpoint]
#: One registered backbone per architecture.
ONE_PER_FAMILY = {get_llm_backbone_spec(n).family: n for n in sorted(MULTIMODAL)}


def tiny_multimodal(fixture: str):
    """A random ``*ForConditionalGeneration`` of the fixture's architecture (eval mode)."""
    payload = json.loads((FIXTURES / f"{fixture}.json").read_text())
    config = transformers.AutoConfig.for_model(payload.pop("model_type"), **payload)
    torch.manual_seed(0)
    return transformers.AutoModelForImageTextToText.from_config(config).eval()


def saved_checkpoint(tmp_path: Path, fixture: str) -> Path:
    """A tiny multimodal checkpoint + word-level tokenizer saved like a Hub repo."""
    tiny_multimodal(fixture).save_pretrained(tmp_path)
    tiny_tokenizer().save_pretrained(tmp_path)
    return tmp_path


def test_every_multimodal_family_is_covered() -> None:
    assert set(ONE_PER_FAMILY) == set(FAMILY_FIXTURE)
    assert len(MULTIMODAL) == 21
    for name in MULTIMODAL:
        spec = get_llm_backbone_spec(name)
        assert spec.source == "hf" and spec.params_b <= 9.0


@pytest.mark.parametrize("family", sorted(FAMILY_FIXTURE))
def test_text_tower_reproduces_the_multimodal_model_on_text(family: str) -> None:
    full = tiny_multimodal(FAMILY_FIXTURE[family])
    ids = torch.randint(3, 100, (2, 9))
    with torch.no_grad():
        expected = full(input_ids=ids).logits
        tower = extract_text_tower(full)
        got = tower(input_ids=ids).logits
    torch.testing.assert_close(got, expected)
    assert not any(k in n for n, _ in tower.named_parameters() for k in ("vision", "visual", "audio"))
    if family == "cosmos-reason2":  # qwen3_vl_text has no *ForCausalLM in transformers
        assert isinstance(tower, TextTowerForCausalLM)
    else:
        assert type(tower).__name__.endswith("ForCausalLM")


@pytest.mark.parametrize("family", sorted(FAMILY_FIXTURE))
def test_lora_targets_cover_every_decoder_layer(family: str) -> None:
    """Hybrid decoders: the default targets must reach linear-attention / conv layers too."""
    tower = extract_text_tower(tiny_multimodal(FAMILY_FIXTURE[family]))
    targets = set(get_llm_backbone_spec(ONE_PER_FAMILY[family]).lora_targets)
    linears = [n for n, m in tower.named_modules() if isinstance(m, torch.nn.Linear)]
    assert targets <= {n.rsplit(".", 1)[-1] for n in linears}
    layers = {int(m.group(1)) for n in linears if (m := re.search(r"layers\.(\d+)\.", n))}
    adapted = {int(m.group(1)) for n in linears
               if n.rsplit(".", 1)[-1] in targets and (m := re.search(r"layers\.(\d+)\.", n))}
    assert adapted == layers


@pytest.mark.parametrize("family", sorted(FAMILY_FIXTURE))
def test_backbone_loads_text_tower_adds_tokens_and_finetunes(tmp_path: Path, family: str) -> None:
    name = ONE_PER_FAMILY[family]
    spec = get_llm_backbone_spec(name)
    llm = HFCausalLMBackbone.from_pretrained(spec, str(saved_checkpoint(tmp_path, FAMILY_FIXTURE[family])),
                                             dtype="float32", attn_implementation="eager")
    assert not any(k in n for n, _ in llm.named_parameters() for k in ("vision", "visual", "audio"))
    special = llm.add_special_tokens(["<image>", "<halo_action>"])
    configure_finetuning(llm, LLMBackboneConfig(name=name, finetune="lora", lora={"r": 4, "alpha": 8}))
    assert any("lora_" in n for n, p in llm.named_parameters() if p.requires_grad)

    ids = torch.randint(3, 60, (2, 8))
    ids[:, 1] = special[0]
    embeds = llm.embed_tokens(ids)
    embeds[:, 1] = torch.randn_like(embeds[:, 1])  # a spliced image embedding
    llm.train()
    out = llm(inputs_embeds=embeds, attention_mask=torch.ones_like(ids), labels=ids, token_ids=ids)
    assert out.hidden_states.shape == (2, 8, llm.hidden_size) and torch.isfinite(out.loss)
    out.loss.backward()
    assert all(p.grad is not None for n, p in llm.named_parameters() if "lora_B" in n)


def test_gemma4_per_layer_embeddings_need_token_ids_once_embeddings_are_spliced(tmp_path: Path) -> None:
    llm = HFCausalLMBackbone.from_pretrained(get_llm_backbone_spec("gemma4-e2b"),
                                             str(saved_checkpoint(tmp_path, "gemma4")), dtype="float32").eval()
    assert llm.per_layer_vocab == 128
    ids = torch.randint(3, 60, (1, 6))
    with torch.no_grad():
        reference = llm(input_ids=ids).logits
        exact = llm(inputs_embeds=llm.embed_tokens(ids), token_ids=ids).logits
        torch.testing.assert_close(exact, reference)  # token_ids reproduce the input_ids path
        spliced = llm.embed_tokens(ids)
        spliced[:, 2] += 0.5
        with pytest.raises(RuntimeError, match="inputs_embeds"):
            llm(inputs_embeds=spliced)  # Gemma cannot recover the ids from a spliced sequence
        new_id = llm.add_special_tokens(["<image>"])[0]
        assert new_id >= 0
        ids[:, 2] = new_id  # beyond the per-layer table -> padding id
        assert torch.isfinite(llm(inputs_embeds=llm.embed_tokens(ids), token_ids=ids).logits).all()


def test_generic_text_tower_generates(tmp_path: Path) -> None:
    llm = HFCausalLMBackbone.from_pretrained(get_llm_backbone_spec("cosmos-reason2-2b"),
                                             str(saved_checkpoint(tmp_path, "qwen3_vl")), dtype="float32").eval()
    prompt = torch.randint(3, 60, (1, 5))
    out = llm.generate(input_ids=prompt, max_new_tokens=4, do_sample=False, pad_token_id=0)
    assert out.shape == (1, 9) and torch.equal(out[:, :5], prompt)
