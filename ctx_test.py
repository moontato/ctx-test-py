#!/usr/bin/env python3
"""
ctx_test.py — Find the practical context limit of a llama-server instance.

Sends increasingly large prompts (linear steps) and records memory usage at
each step via jtop. Stops on first failure or when memory exceeds a threshold.

Usage:
    python3 ctx_test.py [--url URL] [--start N] [--step N] [--max N] [--threshold N]

Requirements:
    pip install jetson-stats requests --break-system-packages
"""

from __future__ import annotations
import argparse
import hashlib
import json
import time
import sys
import requests
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

LLAMA_URL    = "http://127.0.0.1:8080"
MODEL_NAME   = ""          # required in router mode — set to your model alias
START_TOKENS = 50_000       # first test size
STEP_TOKENS  = 25_000       # increment per step
MAX_TOKENS   = 525_000     # hard ceiling (your ctx-size)
MEM_THRESHOLD_GB = 54.0    # stop before OOM — set to ~90% of your total RAM
REQUEST_TIMEOUT  = 600     # seconds to wait for a single completion

# Repeated chunk for building prompts — "the " = 2 tokens reliably
FILL_CHUNK = "the quick brown fox jumps over the lazy dog . " * 1
CHUNK_TOKENS = 10  # approximate tokens per chunk above

# ── jtop memory helper ────────────────────────────────────────────────────────

def get_memory_gb():
    """Return (used_gb, gpu_shared_gb, free_gb) from jtop."""
    try:
        from jtop import jtop
        with jtop() as jetson:
            if jetson.ok():
                ram = jetson.memory["RAM"]
                # values are in KB
                used_kb   = ram["used"]
                shared_kb = ram["shared"]
                free_kb   = ram["free"]
                return (
                    used_kb   / 1024 / 1024,
                    shared_kb / 1024 / 1024,
                    free_kb   / 1024 / 1024,
                )
    except ModuleNotFoundError:
        print("  [jtop error: module not found — run with system python3, or use --no-jtop]")
    except Exception as e:
        print(f"  [jtop error: {e}]")
    return (0.0, 0.0, 0.0)

# ── llama-server helpers ──────────────────────────────────────────────────────

def tokenize(text: str) -> int:
    """Return actual token count from llama-server /tokenize endpoint."""
    try:
        body = {"content": text}
        if MODEL_NAME:
            body["model"] = MODEL_NAME
        r = requests.post(
            f"{LLAMA_URL}/tokenize",
            json=body,
            timeout=30,
        )
        r.raise_for_status()
        return len(r.json().get("tokens", []))
    except Exception as e:
        print(f"  [tokenize error: {e}]")
        return 0

def build_prompt(target_tokens: int) -> str:
    """Build a prompt of approximately target_tokens tokens."""
    # Rough estimate first, then trim
    reps = (target_tokens // CHUNK_TOKENS) + 10
    return FILL_CHUNK * reps

def send_completion(prompt: str, timeout: int = REQUEST_TIMEOUT) -> dict | None:
    """
    Send a /completion request with n_predict=1.
    Returns the response dict on success, None on failure.
    """
    try:
        body = {
            "prompt":        prompt,
            "n_predict":     1,
            "cache_prompt":  False,
            "temperature":   0.0,
        }
        if MODEL_NAME:
            body["model"] = MODEL_NAME
        r = requests.post(
            f"{LLAMA_URL}/completion",
            json=body,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        print(f"  [timeout after {timeout}s]")
    except requests.exceptions.HTTPError as e:
        print(f"  [HTTP error: {e.response.status_code} {e.response.text[:200]}]")
    except Exception as e:
        print(f"  [request error: {e}]")
    return None

def check_health() -> bool:
    try:
        r = requests.get(f"{LLAMA_URL}/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global LLAMA_URL, MODEL_NAME
    parser = argparse.ArgumentParser(description="Test llama-server context limits")
    parser.add_argument("--url",       default=LLAMA_URL,          help="llama-server base URL")
    parser.add_argument("--model",     default=MODEL_NAME,         help="Model alias (required in router mode)")
    parser.add_argument("--start",     type=int, default=START_TOKENS,   help="Starting token count")
    parser.add_argument("--step",      type=int, default=STEP_TOKENS,    help="Token increment per step")
    parser.add_argument("--max",       type=int, default=MAX_TOKENS,     help="Maximum tokens to test")
    parser.add_argument("--threshold", type=float, default=MEM_THRESHOLD_GB, help="Stop if memory exceeds this GB")
    parser.add_argument("--no-jtop",   action="store_true",         help="Skip jtop (if not on Jetson)")
    args = parser.parse_args()

    LLAMA_URL  = args.url
    MODEL_NAME = args.model

    print(f"\n{'='*70}")
    print(f"  llama-server context limit test")
    print(f"  URL: {LLAMA_URL}")
    print(f"  Range: {args.start:,} → {args.max:,} tokens, step {args.step:,}")
    print(f"  Memory stop threshold: {args.threshold} GB")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    if not check_health():
        print("ERROR: llama-server is not responding at /health. Is it running?")
        sys.exit(1)

    # Print header
    col = "{:<10} {:<12} {:<12} {:<12} {:<10} {:<8}"
    print(col.format("Tokens", "Used GB", "GPU sh GB", "Free GB", "Time (s)", "Status"))
    print("-" * 70)

    results = []
    last_success = 0

    target = args.start
    while target <= args.max:

        # Build and verify prompt token count
        print(f"  tokenizing {target:,} token prompt...", end="", flush=True)
        prompt = build_prompt(target)
        actual_tokens = tokenize(prompt)
        print(f" got {actual_tokens:,}", flush=True)

        # Trim or pad to get closer to target
        if actual_tokens > target + 500:
            # Trim by removing chunks from the end
            while actual_tokens > target and len(prompt) > FILL_CHUNK:
                prompt = prompt[:-len(FILL_CHUNK)]
                actual_tokens -= CHUNK_TOKENS
        
        # Memory before request
        if not args.no_jtop:
            used_before, _, _ = get_memory_gb()
        else:
            used_before = 0.0

        # Check threshold before even sending
        total_before = used_before
        if total_before >= args.threshold:
            print(f"\nSTOP: Memory {total_before:.1f} GB already at/above threshold "
                  f"{args.threshold} GB before sending request.")
            break

        # Send request and time it
        t_start = time.time()
        response = send_completion(prompt)
        elapsed = time.time() - t_start

        # Memory after request
        if not args.no_jtop:
            used_after, gpu_after, free_after = get_memory_gb()
        else:
            used_after = gpu_after = free_after = 0.0

        total_after = used_after

        if response is not None:
            tokens_evaluated = response.get("tokens_evaluated", actual_tokens)
            status = "OK"
            last_success = tokens_evaluated
            print(col.format(
                f"{tokens_evaluated:,}",
                f"{used_after:.2f}",
                f"{gpu_after:.2f}",
                f"{free_after:.2f}",
                f"{elapsed:.1f}",
                status,
            ))
            results.append({
                "target_tokens":    target,
                "actual_tokens":    tokens_evaluated,
                "used_gb":          used_after,
                "gpu_shared_gb":    gpu_after,
                "free_gb":          free_after,
                "total_gb":         total_after,
                "elapsed_s":        elapsed,
                "success":          True,
            })
        else:
            status = "FAIL"
            print(col.format(
                f"{target:,}",
                f"{used_after:.2f}",
                f"{gpu_after:.2f}",
                f"{free_after:.2f}",
                f"{elapsed:.1f}",
                status,
            ))
            results.append({
                "target_tokens":    target,
                "actual_tokens":    actual_tokens,
                "used_gb":          used_after,
                "gpu_shared_gb":    gpu_after,
                "free_gb":          free_after,
                "total_gb":         total_after,
                "elapsed_s":        elapsed,
                "success":          False,
            })
            print(f"\nSTOP: Request failed at {target:,} tokens.")
            break

        # Check threshold after request
        if total_after >= args.threshold:
            print(f"\nSTOP: Memory {total_after:.1f} GB hit threshold "
                  f"{args.threshold} GB after {target:,} token request.")
            break

        target += args.step

    # Summary
    print(f"\n{'='*70}")
    print(f"  Results summary")
    print(f"{'='*70}")
    print(f"  Last successful context: {last_success:,} tokens")

    if results:
        successful = [r for r in results if r["success"]]
        failed     = [r for r in results if not r["success"]]
        if successful:
            peak_mem = max(r["total_gb"] for r in successful)
            print(f"  Peak memory (used + GPU sh): {peak_mem:.2f} GB")
            print(f"  Memory at baseline ({args.start:,} tokens): "
                  f"{successful[0]['total_gb']:.2f} GB")
            if len(successful) > 1:
                growth = successful[-1]["total_gb"] - successful[0]["total_gb"]
                print(f"  Memory growth over test: {growth:.2f} GB")
        if failed:
            print(f"  Failed at: {failed[0]['target_tokens']:,} tokens")

    # Save results to JSON
    out_file = f"ctx_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump({
            "config": {
                "url":       args.url,
                "start":     args.start,
                "step":      args.step,
                "max":       args.max,
                "threshold": args.threshold,
                "timestamp": datetime.now().isoformat(),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Full results saved to: {out_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()