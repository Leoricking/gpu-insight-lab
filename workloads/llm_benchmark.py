"""
GPU Insight Lab - LLM Benchmark Workload
Placeholder for LLM inference throughput benchmarking.
Requires transformers/vLLM/llama.cpp - not required for core MVP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMBenchmarkResult:
    status: str = "not_implemented"
    model_name: str = ""
    tokens_per_second: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    total_tokens: int = 0
    gpu_memory_used_mb: Optional[float] = None
    error: Optional[str] = None
    note: str = (
        "LLM benchmarking requires an external inference engine "
        "(transformers, vLLM, or llama.cpp) which is not bundled with GPU Insight Lab v0.1.0. "
        "This placeholder is here to show integration capability."
    )


def run_llm_benchmark(
    model_name: str = "gpt2",
    num_tokens: int = 128,
    batch_size: int = 1,
    backend: str = "transformers",
) -> LLMBenchmarkResult:
    """
    Run LLM inference benchmark.
    Returns not_implemented status in v0.1.0.
    """
    result = LLMBenchmarkResult(model_name=model_name)

    if backend == "transformers":
        try:
            import transformers  # type: ignore  # noqa: PLC0415
            import torch  # type: ignore  # noqa: PLC0415
            import time  # noqa: PLC0415

            logger.info("Loading model %s with transformers", model_name)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            prompt = "GPU Insight Lab performance test: " * 10
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Warmup
            with torch.no_grad():
                _ = model.generate(inputs.input_ids, max_new_tokens=8)

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, max_new_tokens=num_tokens)
            t1 = time.perf_counter()

            generated = outputs.shape[1] - inputs.input_ids.shape[1]
            elapsed = t1 - t0
            result.status = "ok"
            result.tokens_per_second = generated / elapsed
            result.total_tokens = generated

            if device == "cuda":
                result.gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)

        except ImportError as exc:
            result.status = "not_implemented"
            result.error = f"transformers or torch not installed: {exc}"
        except Exception as exc:  # noqa: BLE001
            result.status = "error"
            result.error = str(exc)
    else:
        result.status = "not_implemented"
        result.error = f"Backend '{backend}' not supported in v0.1.0"

    return result
