# groq_client.py — Cloud Llama backend for triage / classification / response inference.
#
# Wraps the Groq Python SDK in a SYNCHRONOUS chat_completion() that returns a dict
# matching llama_cpp's create_chat_completion shape, so existing call sites that
# read response["choices"][0]["message"]["content"] keep working unchanged.
#
# Routing/fallback lives in inference_router.py — this module only does the API call.

import os
import time
from typing import Any, Optional

try:
    from groq import Groq
    from groq import APIError, APIConnectionError, APITimeoutError, RateLimitError
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None
    APIError = APIConnectionError = APITimeoutError = RateLimitError = Exception
    GROQ_AVAILABLE = False


class GroqInferenceError(RuntimeError):
    """Raised on any Groq backend failure. Carries the original error for logging."""
    def __init__(self, message: str, original: Optional[BaseException] = None):
        super().__init__(message)
        self.original = original


class GroqClient:
    """Thin wrapper around groq.Groq with a llama_cpp-shaped return.

    Default model: llama-3.1-8b-instant (matches the local GGUF for behavioral parity).
    Default timeout: 5s (this backend serves triage/classification — must stay fast).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        timeout: float = 5.0,
    ):
        if not GROQ_AVAILABLE:
            raise GroqInferenceError("groq SDK not installed. Run: pip install groq")
        key = api_key or os.getenv("GROQ_API_KEY", "")
        if not key:
            raise GroqInferenceError("GROQ_API_KEY is empty — set it in .env")
        self.model = model
        self.timeout = timeout
        try:
            self._client = Groq(api_key=key, timeout=timeout)
        except Exception as e:
            raise GroqInferenceError(f"Groq client init failed: {e}", e)

    def chat_completion(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.75,
        top_p: float = 1.0,
        stop: Optional[list] = None,
    ) -> dict:
        """Synchronous chat completion. Returns a llama_cpp-shaped dict:
            {"choices": [{"message": {"content": "..."}}]}

        Raises GroqInferenceError on any failure (timeout, rate-limit, network, API error).
        """
        t0 = time.time()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=False,
            )
        except APITimeoutError as e:
            raise GroqInferenceError(f"timeout after {self.timeout}s", e)
        except RateLimitError as e:
            raise GroqInferenceError(f"rate limited: {e}", e)
        except APIConnectionError as e:
            raise GroqInferenceError(f"connection error: {e}", e)
        except APIError as e:
            raise GroqInferenceError(f"API error: {e}", e)
        except Exception as e:
            raise GroqInferenceError(f"{type(e).__name__}: {e}", e)

        try:
            content = resp.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError) as e:
            raise GroqInferenceError(f"malformed Groq response: {e}", e)

        elapsed_ms = int((time.time() - t0) * 1000)
        _in  = getattr(getattr(resp, "usage", None), "prompt_tokens",     0)
        _out = getattr(getattr(resp, "usage", None), "completion_tokens", 0)
        # Record cost telemetry
        try:
            from kira.brain.cost_tracker import cost_tracker as _ct
            _ct.record(model=self.model, input_tokens=_in, output_tokens=_out, purpose="triage")
        except Exception:
            pass
        return {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "_groq_meta": {
                "model": self.model,
                "latency_ms": elapsed_ms,
                "prompt_tokens":     _in,
                "completion_tokens": _out,
            },
        }
