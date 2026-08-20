import requests
import time
import re
# The metrics endpoint is hosted on the same server, just under /metrics

MANUAL_TOTAL_KV_CACHE_GB_VLLM =  66.71

MANUAL_TOTAL_KV_CACHE_GB_SGLANG = 35.94

MANUAL_TOTAL_MAMBA_CACHE_GB_SGLANG = 32.31

MANUAL_TOTAL_INTERNAL_CACHE_GB_SGLANG = 68.25

metrics = ""

import time
from datetime import datetime

LOG_FILE = "vllm_vram_monitor.log"

def monitor_vllm_vram():
    print(f"Monitoring internal vLLM memory. Polling every 1 second...\n")
    print(f"Logging to: {LOG_FILE}\n")

    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"=== Monitor started at {datetime.now().isoformat()} ===\n")

        while True:
            poll_start = time.perf_counter()
            timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            try:
                response = requests.get(metrics, timeout=5)
                lines    = response.text.split('\n')

                kv_cache_usage_line  = next((l for l in lines if l.startswith("vllm:kv_cache_usage_perc{")),  None)
                requests_running_line = next((l for l in lines if l.startswith("vllm:num_requests_running{")), None)
                config_info_line     = next((l for l in lines if l.startswith("vllm:cache_config_info{")),    None)
                requests_waiting_line = next((l for l in lines if l.startswith("vllm:num_requests_waiting{")), None)
                waiting_val = requests_waiting_line.split(" ")[-1]
                if kv_cache_usage_line:
                    usage_fraction = float(kv_cache_usage_line.split(" ")[-1])
                    usage_percent  = usage_fraction * 100
                    total_gb       = MANUAL_TOTAL_KV_CACHE_GB_VLLM

                    if config_info_line:
                        match = re.search(r'kv_cache_memory_bytes="([^"]+)"', config_info_line)
                        if match and match.group(1) != "None":
                            total_gb = float(match.group(1)) / (1024 ** 3)

                    gb_filled   = usage_fraction * total_gb
                    running_val = requests_running_line.split(" ")[-1] if requests_running_line else "0"

                    line = (
                        f"[{timestamp}] "
                        f"VLLM KV Cache Filled: {gb_filled:.2f} GB / {total_gb:.2f} GB "
                        f"({usage_percent:.2f}%) | Requests running: {float(running_val):.0f} "
                        f"Waiting: {float(waiting_val):.0f}"
                    )
                else:
                    line = f"[{timestamp}] No KV cache metric found in response"

            except Exception as e:
                line = f"[{timestamp}] Error fetching/parsing metrics: {e}"

            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            # Sleep for exactly 1 second minus the time already spent this iteration
            elapsed = time.perf_counter() - poll_start
            time.sleep(max(0.0, 1.0 - elapsed))

def monitor_sglang_vram():
    print("Monitoring internal SGLang memory. Polling every 1 second...\n")
    print(f"Logging to: {LOG_FILE}\n")

    def get_metric(lines, metric_name):
        line = next((l for l in lines if l.startswith(metric_name + "{")), None)
        if line is None:
            return None
        return float(line.split(" ")[-1])

    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"=== SGLang monitor started at {datetime.now().isoformat()} ===\n")

        while True:
            poll_start = time.perf_counter()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            try:
                response = requests.get(metrics, timeout=5)
                lines = response.text.split("\n")
                full_token_usage = get_metric(lines, "sglang:full_token_usage") or 0.0
                swa_token_usage = get_metric(lines, "sglang:swa_token_usage") or 0.0
                mamba_usage = get_metric(lines, "sglang:mamba_usage") or 0.0
                token_usage = get_metric(lines, "sglang:token_usage") or 0.0

                used_tokens = get_metric(lines, "sglang:num_used_tokens") or 0.0
                max_tokens = get_metric(lines, "sglang:max_total_num_tokens") or 0.0
                running_reqs = get_metric(lines, "sglang:num_running_reqs") or 0.0
                queue_reqs = get_metric(lines, "sglang:num_queue_reqs") or 0.0
                cache_hit_rate = get_metric(lines, "sglang:cache_hit_rate") or 0.0
                gen_throughput = get_metric(lines, "sglang:gen_throughput") or 0.0

                # From your startup log:
                # KV Cache: K size 12.37 GB + V size 12.37 GB
                total_kv_gb = MANUAL_TOTAL_KV_CACHE_GB_SGLANG

                # Mamba Cache: conv_state 0.51 GB + ssm_state 21.80 GB
                total_mamba_gb = MANUAL_TOTAL_MAMBA_CACHE_GB_SGLANG

                total_internal_cache_gb = total_kv_gb + total_mamba_gb

                # Attention KV cache usage
                full_kv_filled_gb = full_token_usage * total_kv_gb
                swa_kv_filled_gb = swa_token_usage * total_kv_gb
                attention_kv_filled_gb = full_kv_filled_gb + swa_kv_filled_gb

                # Mamba recurrent/cache-state usage
                mamba_filled_gb = mamba_usage * total_mamba_gb

                # Combined internal cache usage
                internal_cache_filled_gb = attention_kv_filled_gb + mamba_filled_gb
                internal_cache_usage_percent = (
                    internal_cache_filled_gb / total_internal_cache_gb * 100
                    if total_internal_cache_gb > 0
                    else 0.0
                )

                line = (
                    f"[{timestamp}] "
                    f"SGLang Internal Cache Filled: "
                    f"{internal_cache_filled_gb:.2f} GB / {total_internal_cache_gb:.2f} GB "
                    f"({internal_cache_usage_percent:.2f}%) | "

                    f"Attention KV: "
                    f"{attention_kv_filled_gb:.2f} GB / {total_kv_gb:.2f} GB "
                    f"(full={full_token_usage * 100:.2f}%, "
                    f"swa={swa_token_usage * 100:.2f}%) | "

                    f"Mamba Cache: "
                    f"{mamba_filled_gb:.2f} GB / {total_mamba_gb:.2f} GB "
                    f"({mamba_usage * 100:.2f}%) | "

                    f"Tokens used: {int(used_tokens)} / {int(max_tokens)} | "
                    f"Running: {int(running_reqs)} | "
                    f"Queued: {int(queue_reqs)} | "
                    f"Cache hit: {cache_hit_rate:.2f} | "
                    f"Gen throughput: {gen_throughput:.2f} tok/s"
                )

            except Exception as e:
                line = f"[{timestamp}] Error fetching/parsing SGLang metrics: {e}"

            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            elapsed = time.perf_counter() - poll_start
            time.sleep(max(0.0, 1.0 - elapsed))


if __name__ == "__main__":
    monitor_vllm_vram()