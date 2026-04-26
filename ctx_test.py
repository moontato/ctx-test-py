#!/usr/bin/env python3
"""
ctx_test.py — Find the practical context limit of a llama-server instance.

Sends increasingly large prompts (linear steps) and records memory usage at
each step via jtop. Stops on first failure or when memory exceeds a threshold.

Usage:
    ./ctx_test.py --config myconfig.ini
    ./ctx_test.py --url http://127.0.0.1:8080 --model my-alias
    ./ctx_test.py --config myconfig.ini --step 16000  # CLI overrides ini

Requirements:
    pip install requests --break-system-packages
    pip install jetson-stats --break-system-packages  (Jetson only)
    pip install Pillow --break-system-packages         (image modes only)
"""

from __future__ import annotations
import argparse
import base64
import configparser
import io
import json
import socket
import time
import sys
import requests
from requests.adapters import HTTPAdapter
from datetime import datetime

# Internal prompt-building constants — not user config
FILL_CHUNK   = "the quick brown fox jumps over the lazy dog . "
CHUNK_TOKENS = 10  # approximate tokens per repetition of FILL_CHUNK

# Defaults for optional settings
_DEFAULTS = {
    "start":        8_000,
    "step":         8_000,
    "max":          256_000,
    "threshold":    58.0,
    "timeout":      1800,
    "image_width":  448,
    "image_height": 448,
}

# ── jtop memory helper ────────────────────────────────────────────────────────

def get_memory_gb():
    """Return (used_gb, gpu_shared_gb, free_gb) from jtop."""
    try:
        from jtop import jtop
        with jtop() as jetson:
            if jetson.ok():
                ram = jetson.memory["RAM"]
                # values are in KB; free = tot - used matches jtop header display
                used_kb   = ram["used"]
                shared_kb = ram["shared"]
                free_kb   = ram["tot"] - ram["used"]
                return (
                    used_kb   / 1024 / 1024,
                    shared_kb / 1024 / 1024,
                    free_kb   / 1024 / 1024,
                )
    except ModuleNotFoundError:
        print("  [jtop error: module not found — install jetson-stats or use --no-jtop]")
    except Exception as e:
        print(f"  [jtop error: {e}]")
    return (0.0, 0.0, 0.0)

# ── llama-server helpers ──────────────────────────────────────────────────────

LLAMA_URL  = ""
MODEL_NAME = ""


def tokenize(text: str, images: list[str] | None = None, timeout: int = 120) -> int:
    """Return actual token count from llama-server /tokenize endpoint."""
    try:
        body = {"content": text}
        if MODEL_NAME:
            body["model"] = MODEL_NAME
        if images:
            body["image_data"] = [{"data": b64, "id": i + 1} for i, b64 in enumerate(images)]
        r = requests.post(f"{LLAMA_URL}/tokenize", json=body, timeout=timeout)
        r.raise_for_status()
        return len(r.json().get("tokens", []))
    except Exception as e:
        print(f"  [tokenize error: {e}]")
        return 0


def build_prompt(target_tokens: int, reserve_tokens: int = 0) -> str:
    """Build a text prompt of approximately (target_tokens - reserve_tokens) tokens."""
    reps = ((target_tokens - reserve_tokens) // CHUNK_TOKENS) + 10
    return FILL_CHUNK * reps


def make_test_image(width: int = 448, height: int = 448) -> str:
    """Generate a solid grey PNG and return it as a base64 string."""
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow is required for image modes — pip install Pillow")
        sys.exit(1)
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class _KeepaliveAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs["socket_options"] = [
            (socket.SOL_SOCKET,  socket.SO_KEEPALIVE,  1),
            (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE,  60),
            (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10),
            (socket.IPPROTO_TCP, socket.TCP_KEEPCNT,   6),
        ]
        super().init_poolmanager(*args, **kwargs)


_session = requests.Session()
_session.mount("http://",  _KeepaliveAdapter())
_session.mount("https://", _KeepaliveAdapter())


def send_completion(
    prompt: str,
    images: list[str] | None = None,
    cache: bool = False,
    timeout: int = _DEFAULTS["timeout"],
) -> dict | None:
    try:
        body = {
            "prompt":       prompt,
            "n_predict":    1,
            "cache_prompt": cache,
            "temperature":  0.0,
        }
        if MODEL_NAME:
            body["model"] = MODEL_NAME
        if images:
            body["image_data"] = [{"data": b64, "id": i + 1} for i, b64 in enumerate(images)]
        r = _session.post(f"{LLAMA_URL}/completion", json=body, timeout=timeout)
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

# ── Config loading ────────────────────────────────────────────────────────────

def load_ini(path: str) -> dict:
    cp = configparser.ConfigParser()
    if not cp.read(path):
        print(f"ERROR: could not read config file: {path}")
        sys.exit(1)
    section = "ctx_test"
    if section not in cp:
        print(f"ERROR: config file must contain a [{section}] section")
        sys.exit(1)
    return dict(cp[section])

# ── Step runner ───────────────────────────────────────────────────────────────

def run_step(
    prompt: str,
    images: list[str] | None,
    target: int,
    actual_tokens: int,
    incremental: bool,
    threshold: float,
    timeout: int,
    no_jtop: bool,
    col: str,
) -> dict | None:
    """Run a single completion step. Returns result dict or None on threshold stop."""
    if not no_jtop:
        used_before, _, _ = get_memory_gb()
    else:
        used_before = 0.0

    if used_before >= threshold:
        print(f"\nSTOP: Memory {used_before:.1f} GB already at/above threshold "
              f"{threshold} GB before sending request.")
        return None

    t_start = time.time()
    response = send_completion(prompt, images=images, cache=incremental, timeout=timeout)
    elapsed = time.time() - t_start

    if not no_jtop:
        used_after, gpu_after, free_after = get_memory_gb()
    else:
        used_after = gpu_after = free_after = 0.0

    success = response is not None
    tokens_evaluated = response.get("tokens_evaluated", actual_tokens) if success else actual_tokens
    status = "OK" if success else "FAIL"

    print(col.format(
        f"{tokens_evaluated:,}",
        f"{used_after:.2f}",
        f"{gpu_after:.2f}",
        f"{free_after:.2f}",
        f"{elapsed:.1f}",
        status,
    ))

    return {
        "target_tokens": target,
        "actual_tokens": tokens_evaluated,
        "used_gb":       used_after,
        "gpu_shared_gb": gpu_after,
        "free_gb":       free_after,
        "total_gb":      used_after,
        "elapsed_s":     elapsed,
        "success":       success,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global LLAMA_URL, MODEL_NAME

    parser = argparse.ArgumentParser(
        description="Test llama-server context limits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file format (--config myconfig.ini):
  [ctx_test]
  url          = http://127.0.0.1:8080
  model        = my-model-alias
  start        = 8000
  step         = 8000
  max          = 256000
  threshold    = 58.0
  timeout      = 1800
  image_width  = 448
  image_height = 448

CLI flags override values in the config file.
'url' and 'model' are required — from config file or flags.
        """,
    )
    parser.add_argument("--config",       help="Path to .ini config file")
    parser.add_argument("--url",          help="llama-server base URL")
    parser.add_argument("--model",        help="Model alias (required in router mode)")
    parser.add_argument("--start",        type=int,   help=f"Starting token count (default: {_DEFAULTS['start']:,})")
    parser.add_argument("--step",         type=int,   help=f"Token increment per step (default: {_DEFAULTS['step']:,})")
    parser.add_argument("--max",          type=int,   help=f"Maximum tokens to test (default: {_DEFAULTS['max']:,})")
    parser.add_argument("--threshold",    type=float, help=f"Stop if used memory exceeds this GB (default: {_DEFAULTS['threshold']})")
    parser.add_argument("--timeout",      type=int,   help=f"Request timeout in seconds (default: {_DEFAULTS['timeout']})")
    parser.add_argument("--image-width",  type=int,   help=f"Synthetic image width in pixels (default: {_DEFAULTS['image_width']})")
    parser.add_argument("--image-height", type=int,   help=f"Synthetic image height in pixels (default: {_DEFAULTS['image_height']})")
    parser.add_argument("--no-jtop",      action="store_true", help="Skip jtop memory readings (if not on Jetson)")
    args = parser.parse_args()

    ini = load_ini(args.config) if args.config else {}

    url          = args.url          if args.url          is not None else ini.get("url")
    model        = args.model        if args.model        is not None else ini.get("model", "")
    start        = args.start        if args.start        is not None else int(ini.get("start",        _DEFAULTS["start"]))
    step         = args.step         if args.step         is not None else int(ini.get("step",         _DEFAULTS["step"]))
    max_tok      = args.max          if args.max          is not None else int(ini.get("max",          _DEFAULTS["max"]))
    threshold    = args.threshold    if args.threshold    is not None else float(ini.get("threshold",  _DEFAULTS["threshold"]))
    timeout      = args.timeout      if args.timeout      is not None else int(ini.get("timeout",      _DEFAULTS["timeout"]))
    image_width  = args.image_width  if args.image_width  is not None else int(ini.get("image_width",  _DEFAULTS["image_width"]))
    image_height = args.image_height if args.image_height is not None else int(ini.get("image_height", _DEFAULTS["image_height"]))

    if not url:
        parser.error("'url' is required — pass --url or set it in a --config .ini file")
    if not model:
        parser.error("'model' is required — pass --model or set it in a --config .ini file")

    LLAMA_URL  = url
    MODEL_NAME = model

    print(f"\n{'='*70}")
    print(f"  llama-server context limit test")
    print(f"  URL: {LLAMA_URL}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Range: {start:,} → {max_tok:,} tokens, step {step:,}")
    print(f"  Memory stop threshold: {threshold} GB")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    if not check_health():
        print("ERROR: llama-server is not responding at /health. Is it running?")
        sys.exit(1)

    _PREFILL_MAP = {"cold": "1", "incremental": "2"}
    _CONTENT_MAP = {"text": "1", "image+text": "2", "multi-image": "3"}

    ini_prefill = ini.get("prefill_mode", "").lower()
    if ini_prefill in _PREFILL_MAP:
        choice = _PREFILL_MAP[ini_prefill]
        print(f"Prefill mode: {ini_prefill} (from config)")
    else:
        print("Prefill mode:")
        print("  [1] Cold        — rebuild prompt each step, no cache (worst-case memory)")
        print("  [2] Incremental — extend prompt each step, reuse KV cache (faster)")
        while True:
            choice = input("Choice [1/2]: ").strip()
            if choice in ("1", "2"):
                break
    incremental = choice == "2"

    ini_content = ini.get("content_mode", "").lower()
    if ini_content in _CONTENT_MAP:
        content_choice = _CONTENT_MAP[ini_content]
        print(f"Content mode: {ini_content} (from config)")
    else:
        print()
        print("Content mode:")
        print("  [1] Text only")
        print("  [2] Image + text  — one synthetic image + text fill per request")
        print("  [3] Multi-image   — scale image count each step, no text (worst-case)")
        while True:
            content_choice = input("Choice [1/2/3]: ").strip()
            if content_choice in ("1", "2", "3"):
                break
    print()

    use_image   = content_choice in ("2", "3")
    multi_image = content_choice == "3"

    # Generate image and measure its token cost via a real completion request
    image_b64    = None
    image_tokens = 0
    if use_image:
        print(f"  Generating {image_width}×{image_height} test image...", end="", flush=True)
        image_b64 = make_test_image(image_width, image_height)
        print(" measuring token cost...", end="", flush=True)
        resp = send_completion("[img-1]", images=[image_b64], timeout=timeout)
        if resp is None:
            print("\nERROR: could not measure image token cost — check model supports vision")
            sys.exit(1)
        image_tokens = resp.get("tokens_evaluated", 0)
        print(f" {image_tokens:,} tokens per image\n")

    col = "{:<10} {:<12} {:<12} {:<12} {:<10} {:<8}"
    print(col.format("Tokens", "Used GB", "GPU sh GB", "Free GB", "Time (s)", "Status"))
    print("-" * 70)

    results      = []
    last_success = 0
    text_prompt  = ""

    target = start
    while target <= max_tok:

        print(f"  tokenizing {target:,} token prompt...", end="", flush=True)

        # ── multi-image: scale image count, no text ───────────────────────────
        if multi_image:
            num_images  = max(1, target // image_tokens)
            imgs        = [image_b64] * num_images
            full_prompt = "\n".join(f"[img-{i+1}]" for i in range(num_images))
            actual_tokens = tokenize(full_prompt, images=imgs, timeout=timeout)
            print(f" got {actual_tokens:,} ({num_images} images)", flush=True)

            result = run_step(
                full_prompt, imgs, target, actual_tokens,
                False, threshold, timeout, args.no_jtop, col,
            )
            if result is None:
                break
            result["num_images"] = num_images
            results.append(result)

            if not result["success"]:
                print(f"\nSTOP: Request failed at {target:,} tokens ({num_images} images).")
                break

            last_success = result["actual_tokens"]

            if len(results) >= 2:
                prev      = results[-2]
                gb_delta  = results[-1]["used_gb"] - prev["used_gb"]
                tok_delta = results[-1]["actual_tokens"] - prev["actual_tokens"]
                if gb_delta > 0 and tok_delta > 0:
                    gb_per_tok    = gb_delta / tok_delta
                    headroom      = threshold - result["used_gb"]
                    predicted_max = int(result["actual_tokens"] + headroom / gb_per_tok)
                    print(f"  Predicted max: ~{predicted_max:,} tokens (~{predicted_max // image_tokens} images) "
                          f"({gb_per_tok*1000:.3f} MB/tok, {headroom:.2f} GB headroom)")

            if result["used_gb"] >= threshold:
                print(f"\nSTOP: Memory {result['used_gb']:.1f} GB hit threshold "
                      f"{threshold} GB after {target:,} token request.")
                break

        # ── single mode: text-only or image+text ─────────────────────────────
        else:
            if use_image:
                if incremental:
                    prev_tokens  = results[-1]["actual_tokens"] if results else 0
                    extra_tokens = target - prev_tokens - image_tokens
                    text_prompt += FILL_CHUNK * ((extra_tokens // CHUNK_TOKENS) + 10)
                else:
                    text_prompt = build_prompt(target, reserve_tokens=image_tokens)
                full_prompt   = "[img-1]\n" + text_prompt
                actual_tokens = tokenize(full_prompt, images=[image_b64], timeout=timeout)
            else:
                if incremental:
                    extra_tokens = target - (results[-1]["actual_tokens"] if results else 0)
                    text_prompt += FILL_CHUNK * ((extra_tokens // CHUNK_TOKENS) + 10)
                else:
                    text_prompt = build_prompt(target)
                full_prompt   = text_prompt
                actual_tokens = tokenize(full_prompt, timeout=timeout)
            print(f" got {actual_tokens:,}", flush=True)

            if actual_tokens > target + 500:
                while actual_tokens > target and len(text_prompt) > len(FILL_CHUNK):
                    text_prompt    = text_prompt[:-len(FILL_CHUNK)]
                    actual_tokens -= CHUNK_TOKENS
                full_prompt = ("[img-1]\n" + text_prompt) if use_image else text_prompt

            result = run_step(
                full_prompt, [image_b64] if use_image else None,
                target, actual_tokens,
                incremental, threshold, timeout, args.no_jtop, col,
            )
            if result is None:
                break

            results.append(result)

            if not result["success"]:
                print(f"\nSTOP: Request failed at {target:,} tokens.")
                break

            last_success = result["actual_tokens"]

            if len(results) >= 2:
                prev      = results[-2]
                gb_delta  = results[-1]["used_gb"] - prev["used_gb"]
                tok_delta = results[-1]["actual_tokens"] - prev["actual_tokens"]
                if gb_delta > 0 and tok_delta > 0:
                    gb_per_tok    = gb_delta / tok_delta
                    headroom      = threshold - result["used_gb"]
                    predicted_max = int(result["actual_tokens"] + headroom / gb_per_tok)
                    print(f"  Predicted max: ~{predicted_max:,} tokens "
                          f"({gb_per_tok*1000:.3f} MB/tok, {headroom:.2f} GB headroom)")

            if result["used_gb"] >= threshold:
                print(f"\nSTOP: Memory {result['used_gb']:.1f} GB hit threshold "
                      f"{threshold} GB after {target:,} token request.")
                break

        target += step

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Results summary")
    print(f"{'='*70}")

    print(f"  Last successful context: {last_success:,} tokens")
    if multi_image and results:
        successful = [r for r in results if r["success"]]
        if successful:
            print(f"  Max images processed: {successful[-1].get('num_images', '?')}")
    successful = [r for r in results if r["success"]]
    failed     = [r for r in results if not r["success"]]
    if successful:
        peak_mem = max(r["total_gb"] for r in successful)
        print(f"  Peak memory (used GB): {peak_mem:.2f} GB")
        print(f"  Memory at baseline ({start:,} tokens): {successful[0]['total_gb']:.2f} GB")
        if len(successful) > 1:
            growth = successful[-1]["total_gb"] - successful[0]["total_gb"]
            print(f"  Memory growth over test: {growth:.2f} GB")
    if failed:
        print(f"  Failed at: {failed[0]['target_tokens']:,} tokens")

    out_file = f"ctx_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump({
            "config": {
                "url":          url,
                "model":        model,
                "start":        start,
                "step":         step,
                "max":          max_tok,
                "threshold":    threshold,
                "timeout":      timeout,
                "image_width":  image_width,
                "image_height": image_height,
                "timestamp":    datetime.now().isoformat(),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Full results saved to: {out_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
