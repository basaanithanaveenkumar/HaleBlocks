"""Normalize HF rows into :class:`LLMSample`."""

from __future__ import annotations

from typing import Any

from hale_core.data.kinds import TrainingStage
from hale_core.data.llm.types import LLMDatasetSpec, LLMSample


def _first_str(row: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for key in fields:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _format_conversations(messages: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    prompt_parts: list[str] = []
    response_parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role") or msg.get("from") or "").lower()
        content = msg.get("content") or msg.get("value") or msg.get("text")
        if not isinstance(content, str):
            continue
        if role in {"user", "human", "instruction"}:
            prompt_parts.append(content)
        elif role in {"assistant", "gpt", "model"}:
            response_parts.append(content)
    prompt = "\n".join(prompt_parts) if prompt_parts else None
    response = "\n".join(response_parts) if response_parts else None
    return prompt, response


def default_row_parser(row: dict[str, Any], spec: LLMDatasetSpec, *, sample_idx: int) -> LLMSample:
    text = _first_str(row, spec.text_fields)
    prompt = _first_str(row, spec.prompt_fields)
    response = _first_str(row, spec.response_fields)
    chosen = _first_str(row, spec.chosen_fields)
    rejected = _first_str(row, spec.rejected_fields)

    conversations = None
    for key in spec.conversation_fields:
        value = row.get(key)
        if isinstance(value, list):
            conversations = value
            conv_prompt, conv_response = _format_conversations(value)
            prompt = prompt or conv_prompt
            response = response or conv_response
            break

    if spec.stage == TrainingStage.PRETRAIN and text is None and prompt and response:
        text = f"{prompt}\n{response}"
    elif text is None and prompt and response:
        text = None
    elif text is None and response:
        text = response

    return LLMSample(
        dataset=spec.name,
        stage=spec.stage,
        domain=spec.domain,
        text=text,
        prompt=prompt,
        response=response,
        conversations=conversations,
        chosen=chosen,
        rejected=rejected,
        metadata={"row_id": row.get("id"), "index": sample_idx},
    )
