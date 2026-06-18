# gpu_telemetry.py — lightweight [VRAM] telemetry instrument (diagnostic, NOT a fix).
#
# Logs Kira's OWN GPU footprint so the regression hunt works from numbers, not
# guesses. The MEANINGFUL figure is PROCESS-level via NVML: it captures this pid's
# total GPU memory — CTranslate2 (the Whispers) + ggml (the local Llama, if loaded)
# + torch — ISOLATED from VTube Studio / OBS / browser, which is exactly what
# nvidia-smi's total-GPU number can't separate.
#
# torch.cuda.memory_allocated()/reserved() are ALSO logged (cheap, and requested),
# but they only see PyTorch TENSORS — faster-whisper and llama.cpp allocate CUDA
# memory OUTSIDE torch, so these usually read ~0. That's expected, not a bug; the
# kira_process= number is the real footprint.
#
# Every call is cheap (a couple torch reads + one NVML query) and NEVER raises.

import os
import time

try:
    import torch
except Exception:
    torch = None

_nvml = None
_nvml_state = "uninit"   # uninit -> "ok" | "unavailable"


def _ensure_nvml() -> None:
    global _nvml, _nvml_state
    if _nvml_state != "uninit":
        return
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml = pynvml
        _nvml_state = "ok"
    except Exception as e:
        _nvml_state = "unavailable"
        # Loud once (silent-failure principle): say the per-process number is off.
        print(f"   [VRAM] NVML unavailable — per-process number disabled, torch-only: {e}")


def _process_and_card_mb():
    """(this_pid_used_MB, card_used_MB, card_total_MB) via NVML, summed across GPUs,
    or (None, None, None) if NVML is unavailable. this_pid captures CT2 + ggml + torch."""
    if _nvml_state != "ok":
        return None, None, None
    try:
        pid = os.getpid()
        proc_used = card_used = card_total = 0
        for i in range(_nvml.nvmlDeviceGetCount()):
            h = _nvml.nvmlDeviceGetHandleByIndex(i)
            mem = _nvml.nvmlDeviceGetMemoryInfo(h)
            card_used += mem.used
            card_total += mem.total
            for p in _nvml.nvmlDeviceGetComputeRunningProcesses(h):
                if p.pid == pid and getattr(p, "usedGpuMemory", None):
                    proc_used += p.usedGpuMemory
        MB = 1048576.0
        return proc_used / MB, card_used / MB, card_total / MB
    except Exception:
        return None, None, None


def log_vram(tag: str = "") -> None:
    """Emit one greppable `[VRAM]` line. Cheap; never raises into the app."""
    try:
        _ensure_nvml()
        seg = [f"   [VRAM] {time.strftime('%H:%M:%S')}"]
        if tag:
            seg.append(f"({tag})")
        proc_mb, card_mb, total_mb = _process_and_card_mb()
        if proc_mb is not None:
            seg.append(f"kira_process={proc_mb / 1024.0:.2f}GB")
            seg.append(f"card={card_mb / 1024.0:.1f}/{total_mb / 1024.0:.1f}GB")
        if torch is not None:
            try:
                if torch.cuda.is_available():
                    seg.append(
                        f"torch_alloc={torch.cuda.memory_allocated() / 1073741824.0:.2f}GB "
                        f"reserved={torch.cuda.memory_reserved() / 1073741824.0:.2f}GB"
                    )
            except Exception:
                pass
        print(" ".join(seg))
    except Exception:
        pass
