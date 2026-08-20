#!/usr/bin/env python
"""CLI runner for the RQ1 inference benchmark notebook.

Example:
    python rq1_benchmark_cli.py \
        --url https://example.inference.spike.tue.nl/v1/chat/completions \
        --profiles A,B,C,D \
        --concurrencies 1,4,16,32 \
        --runs-per-cell 5 \
        --warmup-requests 3
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp


HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}

PROFILES: dict[str, dict[str, Any]] = {
    "A": {
        "name": "Chat / Conversational",
        "dataset": "sharegpt",
        "input_toks": 128,
        "output_toks": 128,
        "stress": "High concurrency",
        "description": "ShareGPT Vicuna - first human turn as prompt",
    },
    "B": {
        "name": "Code Generation",
        "dataset": "magicoder",
        "input_toks": 512,
        "output_toks": 1024,
        "stress": "Decode-heavy",
        "description": "Magicoder OSS Instruct - instruction field as prompt",
    },
    "C": {
        "name": "RAG / Summarization",
        "dataset": "scrolls",
        "input_toks": 4096,
        "output_toks": 256,
        "stress": "Prefill-heavy",
        "description": "SCROLLS gov_report - document truncated to target length",
    },
    "D": {
        "name": "Long-context / Agentic",
        "dataset": "longbench",
        "input_toks": 16384,
        "output_toks": 512,
        "stress": "Extreme prefill",
        "description": "LongBench NarrativeQA / RepoBench-P",
    },
}

FIELDNAMES = [
    "model",
    "arch",
    "engine",
    "profile",
    "profile_name",
    "dataset",
    "concurrency",
    "input_tokens_target",
    "output_tokens_target",
    "run",
    "throughput_tps",
    "mean_ttft_ms",
    "p50_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p50_tpot_ms",
    "p99_tpot_ms",
    "batch_wall_s",
    "success_rate",
    "n",
    "mean_actual_input_toks",
    "mean_actual_output_toks",
    "timestamp",
]

_DATASET_CACHE: dict[str, list[str]] = {}
_TOKENIZER_CACHE: dict[str, Any] = {}


@dataclass
class RequestMetrics:
    ttft_ms: float
    tpot_ms: float
    wall_s: float
    tokens_out: int
    tokens_in: int


@dataclass
class BenchmarkConfig:
    url: str
    model_id: str
    profiles: list[str]
    concurrencies: list[int]
    runs_per_cell: int
    warmup_requests: int
    output: Path
    model_label: str
    arch: str
    engine: str
    cooldown_s: float
    connect_timeout_s: float
    total_timeout_s: float
    seed: int
    dataset_prewarm: bool
    max_connections: int
    request_max_tokens_warmup: int
    ignore_eos: bool
    logprobs: bool


def parse_csv_values(raw: str, *, cast: Any = str) -> list[Any]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list cannot be empty")
    return [cast(item) for item in values]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def infer_chat_url(url: str) -> str:
    cleaned = url.rstrip("/")
    if cleaned.endswith("/v1/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return cleaned + "/chat/completions"
    if "/v1/" not in cleaned:
        return cleaned + "/v1/chat/completions"
    return cleaned


def models_url_from_chat_url(url: str) -> str:
    if "/v1/" in url:
        return url.split("/v1/", 1)[0].rstrip("/") + "/v1/models"
    return url.rstrip("/") + "/v1/models"


def infer_label_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or "custom_model"
    return host.split(".")[0].replace("-", "_")


def infer_engine_from_url(url: str) -> str:
    label = infer_label_from_url(url).lower()
    for engine in ("sglang", "vllm"):
        if engine in label:
            return engine
    return "custom"


def infer_arch_from_model_id(model_id: str) -> str:
    lowered = model_id.lower()
    if any(name in lowered for name in ("qwen", "nemotron")):
        return "ssm_hybrid"
    if "ling" in lowered:
        return "linear_hybrid"
    return "unknown"


async def discover_model_id(url: str, timeout_s: float) -> str:
    models_url = models_url_from_chat_url(url)
    timeout = aiohttp.ClientTimeout(connect=min(10, timeout_s), total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(models_url, headers=HEADERS) as resp:
            if resp.status != 200:
                err = await resp.text()
                raise RuntimeError(
                    f"Could not discover model id from {models_url}: HTTP {resp.status}: {err[:200]}"
                )
            payload = await resp.json()
    data = payload.get("data", [])
    if not data or not data[0].get("id"):
        raise RuntimeError(f"No model ids found in {models_url}")
    return data[0]["id"]


def get_tokenizer(model_id: str) -> Any:
    if model_id not in _TOKENIZER_CACHE:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                trust_remote_code=True,
            )
            _TOKENIZER_CACHE[model_id] = tok
            print(f"Tokenizer loaded: {model_id}")
        except Exception as exc:
            print(
                f"Tokenizer unavailable for {model_id} ({exc}); "
                "falling back to whitespace token counting"
            )
            _TOKENIZER_CACHE[model_id] = None
    return _TOKENIZER_CACHE[model_id]


def count_tokens(text: str, tokenizer: Any) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_tokens(text: str, target_tokens: int, tokenizer: Any) -> str:
    if target_tokens <= 0:
        return ""
    if tokenizer is None:
        return " ".join(text.split()[:target_tokens])
    if count_tokens(text, tokenizer) <= target_tokens:
        return text

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid], tokenizer) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def pad_to_tokens(text: str, target_tokens: int, tokenizer: Any) -> str:
    current = count_tokens(text, tokenizer)
    if current >= target_tokens:
        return text

    filler_lines = []
    while current < target_tokens:
        remaining = target_tokens - current
        line = f"[pad:{uuid.uuid4()}]"
        line_toks = count_tokens(line + "\n", tokenizer)
        if line_toks > remaining:
            filler_lines.append(" " * remaining)
            break
        filler_lines.append(line)
        current += line_toks

    padded = text + "\n" + "\n".join(filler_lines)
    while count_tokens(padded, tokenizer) > target_tokens:
        padded = padded[:-1]
    return padded


def fit_to_tokens(text: str, target_tokens: int, tokenizer: Any) -> str:
    return pad_to_tokens(truncate_to_tokens(text, target_tokens, tokenizer), target_tokens, tokenizer)


def synthetic_prompt(target_tokens: int, tokenizer: Any) -> str:
    seed = f"[uid={uuid.uuid4()}] Continue for {target_tokens} tokens:\n"
    body = "The quick brown fox jumps over the lazy dog. " * (target_tokens + 10)
    return fit_to_tokens(seed + body, target_tokens, tokenizer)


def load_sharegpt(n: int = 200) -> list[str]:
    key = "sharegpt"
    if key not in _DATASET_CACHE:
        try:
            from datasets import load_dataset

            ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
            prompts = []
            for ex in ds:
                convs = ex.get("conversations", [])
                human = next(
                    (
                        c["value"]
                        for c in convs
                        if c.get("from") in ("human", "user") and c.get("value")
                    ),
                    None,
                )
                if human:
                    prompts.append(human)
                if len(prompts) >= n:
                    break
            _DATASET_CACHE[key] = prompts
            print(f"ShareGPT loaded {len(prompts)} prompts")
        except Exception as exc:
            print(f"ShareGPT unavailable ({exc}); using synthetic fallback")
            _DATASET_CACHE[key] = []
    return _DATASET_CACHE[key]


def load_magicoder(n: int = 200) -> list[str]:
    key = "magicoder"
    if key not in _DATASET_CACHE:
        try:
            from datasets import load_dataset

            ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train", streaming=True)
            prompts = []
            for ex in ds:
                instr = ex.get("instruction") or ex.get("problem", "")
                if instr:
                    prompts.append(instr)
                if len(prompts) >= n:
                    break
            _DATASET_CACHE[key] = prompts
            print(f"Magicoder loaded {len(prompts)} prompts")
        except Exception as exc:
            print(f"Magicoder unavailable ({exc}); using synthetic fallback")
            _DATASET_CACHE[key] = []
    return _DATASET_CACHE[key]


def load_govreport(n: int = 100) -> list[str]:
    key = "scrolls"
    if key not in _DATASET_CACHE:
        try:
            from datasets import load_dataset

            ds = load_dataset("ccdv/govreport-summarization", split="train", streaming=True)
            prompts = []
            for ex in ds:
                text = ex.get("document") or ex.get("report", "")
                if text:
                    prompts.append(text)
                if len(prompts) >= n:
                    break
            _DATASET_CACHE[key] = prompts
            print(f"GovReport loaded {len(prompts)} documents")
        except Exception as exc:
            print(f"GovReport unavailable ({exc}); using synthetic fallback")
            _DATASET_CACHE[key] = []
    return _DATASET_CACHE[key]


def load_longbench(n: int = 100) -> list[str]:
    key = "longbench"
    if key not in _DATASET_CACHE:
        try:
            from datasets import load_dataset

            ds = load_dataset("THUDM/LongBench-v2", split="train", streaming=True)
            prompts = []
            for ex in ds:
                ctx = ex.get("context", "")
                question = ex.get("question", "")
                text = (ctx + "\n\n" + question).strip() if ctx else question
                if text:
                    prompts.append(text)
                if len(prompts) >= n:
                    break
            _DATASET_CACHE[key] = prompts
            print(f"LongBench-v2 loaded {len(prompts)} examples")
        except Exception as exc:
            print(f"LongBench-v2 unavailable ({exc}); using synthetic fallback")
            _DATASET_CACHE[key] = []
    return _DATASET_CACHE[key]


LOADERS = {
    "sharegpt": load_sharegpt,
    "magicoder": load_magicoder,
    "scrolls": load_govreport,
    "longbench": load_longbench,
}


def get_prompt(profile_key: str, model_id: str, rng: random.Random) -> str:
    profile = PROFILES[profile_key]
    target = int(profile["input_toks"])
    tokenizer = get_tokenizer(model_id)
    loader = LOADERS.get(str(profile["dataset"]))
    pool = loader() if loader else []

    if not pool:
        return synthetic_prompt(target, tokenizer)

    uid_prefix = f"[uid={uuid.uuid4()}]\n"
    uid_toks = count_tokens(uid_prefix, tokenizer)
    text = fit_to_tokens(rng.choice(pool), max(target - uid_toks, 0), tokenizer)
    return uid_prefix + text


def prewarm_datasets(profiles: list[str]) -> None:
    print("Pre-loading datasets...")
    for profile_key in profiles:
        profile = PROFILES[profile_key]
        loader = LOADERS.get(str(profile["dataset"]))
        pool = loader() if loader else []
        print(
            f"Profile {profile_key} ({profile['name']}): {len(pool)} examples "
            f"| target {profile['input_toks']} in / {profile['output_toks']} out tokens"
        )


async def send_streaming_request(
    session: aiohttp.ClientSession,
    config: BenchmarkConfig,
    prompt: str,
    output_toks: int,
    req_id: int,
    label: str,
) -> RequestMetrics | None:
    payload = {
        "model": config.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_toks,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": config.ignore_eos,
    }
    if config.logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 1

    t_start = time.perf_counter()
    t_first_tok = None
    token_count = 0
    prompt_tokens_actual = 0
    completion_tokens_actual = 0

    try:
        async with session.post(config.url, json=payload, headers=HEADERS) as resp:
            if resp.status != 200:
                err = await resp.text()
                print(f"[{label} req {req_id}] HTTP {resp.status}: {err[:200]}")
                return None

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or line == "data: [DONE]" or not line.startswith("data: "):
                    continue

                if t_first_tok is None:
                    t_first_tok = time.perf_counter()

                try:
                    chunk = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens_actual = int(usage.get("prompt_tokens") or 0)
                    completion_tokens_actual = int(usage.get("completion_tokens") or 0)

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning")
                    ):
                        token_count += 1

        t_end = time.perf_counter()
        if t_first_tok is None or token_count == 0:
            print(f"[{label} req {req_id}] No tokens received")
            return None

        decode_s = t_end - t_first_tok
        actual_out = completion_tokens_actual if completion_tokens_actual > 0 else token_count
        return RequestMetrics(
            ttft_ms=(t_first_tok - t_start) * 1000,
            tpot_ms=(decode_s / max(token_count - 1, 1)) * 1000,
            wall_s=t_end - t_start,
            tokens_out=actual_out,
            tokens_in=prompt_tokens_actual,
        )
    except Exception as exc:
        print(f"[{label} req {req_id}] {exc}")
        return None


async def run_batch(
    config: BenchmarkConfig,
    concurrency: int,
    profile_key: str,
    rng: random.Random,
) -> dict[str, Any]:
    profile = PROFILES[profile_key]
    output_toks = int(profile["output_toks"])
    label = f"{config.model_label}/{config.engine}/P{profile_key}"
    print(
        f"  > [{label}] profile={profile_key} concurrency={concurrency} "
        f"in~{profile['input_toks']} out={output_toks}"
    )

    timeout = aiohttp.ClientTimeout(
        connect=config.connect_timeout_s,
        total=config.total_timeout_s,
    )
    connector = aiohttp.TCPConnector(limit=config.max_connections)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for warmup_idx in range(config.warmup_requests):
            warmup_prompt = get_prompt(profile_key, config.model_id, rng)
            await send_streaming_request(
                session,
                config,
                warmup_prompt,
                config.request_max_tokens_warmup,
                -(warmup_idx + 1),
                label + "_warmup",
            )

        prompts = [get_prompt(profile_key, config.model_id, rng) for _ in range(concurrency)]
        batch_t0 = time.perf_counter()
        tasks = [
            send_streaming_request(session, config, prompts[i], output_toks, i, label)
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)
        batch_t1 = time.perf_counter()

    valid = [result for result in results if result is not None]
    if not valid:
        print(f"  All {concurrency} requests failed.")
        return {}

    wall_time = batch_t1 - batch_t0
    total_out = sum(result.tokens_out for result in valid)
    ttfts = [result.ttft_ms for result in valid]
    tpots = [result.tpot_ms for result in valid]
    tokens_in_list = [result.tokens_in for result in valid]
    tokens_out_list = [result.tokens_out for result in valid]

    return {
        "throughput_tps": round(total_out / wall_time if wall_time > 0 else 0.0, 2),
        "mean_ttft_ms": round(statistics.fmean(ttfts), 2),
        "p50_ttft_ms": round(percentile(ttfts, 50), 2),
        "p99_ttft_ms": round(percentile(ttfts, 99), 2),
        "mean_tpot_ms": round(statistics.fmean(tpots), 2),
        "p50_tpot_ms": round(percentile(tpots, 50), 2),
        "p99_tpot_ms": round(percentile(tpots, 99), 2),
        "batch_wall_s": round(wall_time, 3),
        "success_rate": round(len(valid) / concurrency, 3),
        "n": len(valid),
        "mean_actual_input_toks": round(statistics.fmean(tokens_in_list), 1),
        "mean_actual_output_toks": round(statistics.fmean(tokens_out_list), 1),
    }


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


async def sweep(config: BenchmarkConfig) -> list[dict[str, Any]]:
    rng = random.Random(config.seed)
    rows = []

    if config.dataset_prewarm:
        prewarm_datasets(config.profiles)

    print("=" * 72)
    print(f"Endpoint: {config.url}")
    print(f"Model id: {config.model_id}")
    print(
        f"Grid: {len(config.profiles)} profiles x {len(config.concurrencies)} concurrencies "
        f"x {config.runs_per_cell} runs"
    )
    print("=" * 72)

    for profile_key in config.profiles:
        profile = PROFILES[profile_key]
        print(f"\nProfile {profile_key}: {profile['name']}")

        for concurrency in config.concurrencies:
            run_metrics = []
            for run_idx in range(config.runs_per_cell):
                cell = await run_batch(config, concurrency, profile_key, rng)
                if not cell:
                    continue

                row = {
                    "model": config.model_label,
                    "arch": config.arch,
                    "engine": config.engine,
                    "profile": profile_key,
                    "profile_name": profile["name"],
                    "dataset": profile["dataset"],
                    "concurrency": concurrency,
                    "input_tokens_target": profile["input_toks"],
                    "output_tokens_target": profile["output_toks"],
                    "run": run_idx,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    **cell,
                }
                rows.append(row)
                run_metrics.append(row)
                append_csv_row(config.output, row)

            if run_metrics:
                tputs = [r["throughput_tps"] for r in run_metrics]
                ttfts = [r["mean_ttft_ms"] for r in run_metrics]
                tpots = [r["mean_tpot_ms"] for r in run_metrics]
                print(
                    f"    conc={concurrency:3d} "
                    f"TPS={statistics.fmean(tputs):7.1f} "
                    f"TTFT={statistics.fmean(ttfts):6.0f}ms "
                    f"TPOT={statistics.fmean(tpots):6.1f}ms"
                )

            if config.cooldown_s > 0:
                print(f"  Cooling down {config.cooldown_s:g} s...")
                await asyncio.sleep(config.cooldown_s)

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RQ1 OpenAI-compatible streaming inference benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", required=True, help="Endpoint base URL, /v1 URL, or /v1/chat/completions URL.")
    parser.add_argument(
        "--profiles",
        type=lambda raw: parse_csv_values(raw.upper()),
        default=["A", "B", "C", "D"],
        help="Comma-separated profile keys: A,B,C,D.",
    )
    parser.add_argument(
        "--concurrencies",
        type=lambda raw: parse_csv_values(raw, cast=int),
        default=[1, 4, 16, 32],
        help="Comma-separated simultaneous request counts.",
    )
    parser.add_argument("--runs-per-cell", type=int, default=5)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--model-id", help="Model id for request payloads and tokenizer loading. Defaults to first /v1/models id.")
    parser.add_argument("--model-label", help="Short label for CSV rows. Defaults to hostname-derived label.")
    parser.add_argument("--engine", help="Engine label for CSV rows. Defaults to inferred vllm/sglang/custom.")
    parser.add_argument("--arch", help="Architecture label for CSV rows. Defaults to inferred value from model id.")
    parser.add_argument("--output", type=Path, default=Path("rq1_raw_results.csv"), help="CSV output path.")
    parser.add_argument("--cooldown-s", type=float, default=15.0, help="Pause after each concurrency cell.")
    parser.add_argument("--connect-timeout-s", type=float, default=10.0)
    parser.add_argument("--total-timeout-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-connections", type=int, default=256)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument("--no-dataset-prewarm", action="store_true", help="Load datasets lazily instead of before the sweep.")
    parser.add_argument("--ignore-eos", action="store_true", help="Ask compatible servers to ignore EOS.")
    parser.add_argument("--no-logprobs", action="store_true", help="Do not request logprobs/top_logprobs.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    unknown_profiles = sorted(set(args.profiles) - set(PROFILES))
    if unknown_profiles:
        raise SystemExit(f"Unknown profile(s): {', '.join(unknown_profiles)}. Valid profiles: {', '.join(PROFILES)}")
    if any(concurrency < 1 for concurrency in args.concurrencies):
        raise SystemExit("All concurrencies must be >= 1")
    if args.runs_per_cell < 1:
        raise SystemExit("--runs-per-cell must be >= 1")
    if args.warmup_requests < 0:
        raise SystemExit("--warmup-requests must be >= 0")
    if args.connect_timeout_s <= 0 or args.total_timeout_s <= 0:
        raise SystemExit("Timeout values must be positive")
    
async def isAlive(url: str, timeout_s: float)  -> bool:
    models_url = models_url_from_chat_url(url)
    timeout = aiohttp.ClientTimeout(connect=min(10, timeout_s), total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(models_url, headers=HEADERS) as resp:
            if resp.status == 404:
                raise RuntimeError(
                    f"Model with given URL: {models_url} do not exist. Remember to use https."
                )
            if resp.status != 200:
                return False
            else:
                return True    


async def async_main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    url = infer_chat_url(args.url)
    if not await isAlive(url, args.connect_timeout_s):
        raise SystemExit("Model is loading. Wait around 10 minutes and run the program again.")
    model_id = args.model_id
    if not model_id:
        print("Discovering model id from /v1/models...")
        model_id = await discover_model_id(url, args.connect_timeout_s)

    config = BenchmarkConfig(
        url=url,
        model_id=model_id,
        profiles=args.profiles,
        concurrencies=args.concurrencies,
        runs_per_cell=args.runs_per_cell,
        warmup_requests=args.warmup_requests,
        output=args.output,
        model_label=args.model_label or infer_label_from_url(url),
        arch=args.arch or infer_arch_from_model_id(model_id),
        engine=args.engine or infer_engine_from_url(url),
        cooldown_s=args.cooldown_s,
        connect_timeout_s=args.connect_timeout_s,
        total_timeout_s=args.total_timeout_s,
        seed=args.seed,
        dataset_prewarm=not args.no_dataset_prewarm,
        max_connections=args.max_connections,
        request_max_tokens_warmup=args.warmup_max_tokens,
        ignore_eos=args.ignore_eos,
        logprobs=not args.no_logprobs,
    )

    rows = await sweep(config)
    print(f"\nSweep complete. {len(rows)} rows collected -> {config.output}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
