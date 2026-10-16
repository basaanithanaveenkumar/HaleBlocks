"""DeepSeek-style MoE: noisy top-k router + shared experts (HaloBlocks / HALO-WAM)."""

from hale_core.nn.layers.ffn.moe.deepseek_mo_e import DeepseekMoE
from hale_core.nn.layers.ffn.moe.noise_best_k_router import NoiseBestKRouter
from hale_core.nn.layers.ffn.moe.swi_glu_expert import SwiGLUExpert

__all__ = ["DeepseekMoE", "NoiseBestKRouter", "SwiGLUExpert"]
