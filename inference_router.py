# inference_router.py — Routes local-Llama-style chat completions to either
# Groq cloud or the local llama_cpp model based on config.INFERENCE_BACKEND.
#
# Contract: `route_chat_completion(ai_core, messages, max_tokens, temperature, ...)`
# returns a dict shaped exactly like llama_cpp's create_chat_completion result, so
# every existing call site continues to do response["choices"][0]["message"]["content"].
#
# Behavior:
#   INFERENCE_BACKEND="local"  -> always local (legacy path).
#   INFERENCE_BACKEND="groq"   -> Groq first, fallback per GROQ_FALLBACK_TO_LOCAL:
#       "false"      -> no fallback; raise on Groq error.
#       "true"       -> local Llama must already be loaded; fall back silently.
#       "lazy_load"  -> on first Groq failure, load local Llama into VRAM and use
#                       it from then on for this session (one-time cost).

import os
from typing import Any, Optional

import config
from groq_client import GroqClient, GroqInferenceError


_groq_singleton: Optional[GroqClient] = None
_lazy_load_attempted = False


def get_groq_client() -> GroqClient:
    """Lazily build (and cache) the Groq client. Raises GroqInferenceError if unusable."""
    global _groq_singleton
    if _groq_singleton is None:
        _groq_singleton = GroqClient(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL,
            timeout=config.GROQ_TIMEOUT,
        )
    return _groq_singleton


def _ensure_local_llm_loaded(ai_core) -> bool:
    """Lazy-load the local Llama into VRAM on first Groq failure. Returns True
    if the local model is ready to serve a call after this returns."""
    global _lazy_load_attempted
    if ai_core.llm is not None:
        return True
    if _lazy_load_attempted:
        return ai_core.llm is not None
    _lazy_load_attempted = True
    print("   [INFERENCE] Groq failed and lazy_load is set — loading local Llama into VRAM (one-time)...")
    try:
        ai_core._init_llm(force=True)
    except Exception as e:
        print(f"   [INFERENCE] Local Llama lazy-load FAILED: {e}")
        return False
    return ai_core.llm is not None


def _call_local(ai_core, messages, max_tokens, temperature, top_p, repeat_penalty, stop):
    """Synchronous local Llama call under the AI_Core inference_lock."""
    if ai_core.llm is None:
        raise RuntimeError("local Llama not loaded (INFERENCE_BACKEND=groq with no fallback)")
    with ai_core.inference_lock:
        return ai_core.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
            stream=False,
        )


def route_chat_completion(
    ai_core,
    messages: list,
    max_tokens: int,
    temperature: float = 0.75,
    top_p: float = 1.0,
    repeat_penalty: float = 1.1,
    stop: Optional[list] = None,
) -> dict:
    """Synchronous router. Returns a llama_cpp-shaped response dict. Designed to be
    called via asyncio.to_thread from async callers, exactly like the local path was."""
    backend = (config.INFERENCE_BACKEND or "groq").lower()

    if backend == "local":
        return _call_local(ai_core, messages, max_tokens, temperature, top_p, repeat_penalty, stop)

    # Groq path
    try:
        client = get_groq_client()
        return client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
    except GroqInferenceError as e:
        fallback_mode = (config.GROQ_FALLBACK_TO_LOCAL or "false").lower()
        if fallback_mode == "false":
            print(f"   [GROQ_FAIL] {e} — no fallback, re-raising.")
            raise

        if fallback_mode == "lazy_load" and ai_core.llm is None:
            if not _ensure_local_llm_loaded(ai_core):
                print(f"   [GROQ_FAIL] {e} — lazy_load failed too. Re-raising.")
                raise

        if ai_core.llm is None:
            print(f"   [GROQ_FAIL] {e} — local Llama not loaded for fallback. Re-raising.")
            raise

        print(f"   [GROQ_FAIL] {e} — falling back to local Llama.")
        return _call_local(ai_core, messages, max_tokens, temperature, top_p, repeat_penalty, stop)
