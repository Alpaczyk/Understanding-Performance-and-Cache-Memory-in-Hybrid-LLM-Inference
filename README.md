# Understanding Performance and Cache Memory in Hybrid LLM Inference

Artifacts from my master's thesis (TU Eindhoven): a benchmarking and monitoring toolkit for studying how **hybrid LLM architectures** — models mixing attention layers with linear-recurrent (SSM/Mamba) layers — perform under realistic serving workloads, and how their cache memories fill up under load.

## What's in here

### `benchmark_cli.py` — LLM serving benchmark

An asynchronous load generator (aiohttp) for any OpenAI-compatible endpoint (vLLM, SGLang, ...). It sweeps **workload profiles × concurrency levels** and records latency and throughput to CSV.

| Profile | Workload | Target tokens | Stress |
|---|---|---|---|
| **A** | Chat (ShareGPT) | 128 in / 128 out | high concurrency |
| **B** | Code generation (Magicoder) | 512 in / 1024 out | decode-heavy |
| **C** | RAG / summarization (SCROLLS gov-report) | 4K in / 256 out | prefill-heavy |
| **D** | Long-context / agentic (LongBench) | 16K in / 512 out | extreme prefill |

Per request it measures TTFT, TPOT, wall time and token counts; per (profile, concurrency) cell it aggregates mean / p50 / p99 latency and throughput (tok/s). Prompts are sized token-precisely — real datasets are truncated and padded using the model's tokenizer (binary search) with a synthetic-prompt fallback. Architecture and engine labels (e.g. `ssm_hybrid`, `linear_hybrid`, `vllm`, `sglang`) are inferred from the endpoint and recorded per row.

```bash
python benchmark_cli.py --url https://<endpoint>/v1/chat/completions \
    --profiles A,B,C,D --concurrencies 1,4,16,32 --runs-per-cell 5
```

### `monitor.py` — cache memory monitor

Polls the engine's `/metrics` endpoint once per second while benchmarks run, and logs cache-memory behaviour:

- **vLLM** — KV-cache usage (GB and %), running / waiting requests
- **SGLang** — attention KV cache (full + sliding-window), Mamba/SSM cache state, token usage, queue depth, cache hit rate, generation throughput

### Data

`RQ1_all_data.xlsx` and `RQ2_all_data.xlsx` contain the collected measurements for the thesis' research questions.

## Requirements

`aiohttp` is required; `transformers` and `datasets` are optional (tokenizer accuracy / real dataset prompts) with automatic fallbacks.
