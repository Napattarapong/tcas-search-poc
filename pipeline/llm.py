"""Shared LLM helpers: GLM glm-5.1 via Hyper-Extract (Anthropic-compatible z.ai)."""
import os
import sys

from langchain_anthropic import ChatAnthropic

from . import config as C  # noqa: E402


def get_llm():
    # ChatAnthropic reads ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL from env (z.ai endpoint).
    # timeout caps each request so a hung call bails instead of stalling the run.
    return ChatAnthropic(
        model=C.GLM_MODEL,
        temperature=0,
        max_tokens=8192,
        timeout=180,
    )


def extract_chunks(list_model, instruction_fn, chunks, label="chunk"):
    """Run with_structured_output(list_model) over each text chunk; merge items.

    list_model: a pydantic model with an `items: List[...]` field (FLAT item schema).
    instruction_fn(chunk_text) -> full prompt string.
    chunks: list[str].
    Returns list of item dicts.
    """
    struct = get_llm().with_structured_output(list_model)
    out = []
    for i, ch in enumerate(chunks, 1):
        try:
            res = struct.invoke(instruction_fn(ch))
            out.extend(res.items)
            print(f"  [{label} {i}/{len(chunks)}] +{len(res.items)} (total {len(out)})",
                  file=sys.stderr, flush=True)
        except Exception as e:
            print(f"  [{label} {i}/{len(chunks)}] ERROR {type(e).__name__}: {str(e)[:120]}",
                  file=sys.stderr, flush=True)
    return out
