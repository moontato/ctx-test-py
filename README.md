# ctx-test-py

Finds the practical context length limit of a llama-server instance by sending progressively larger prompts and recording memory usage at each step. Stops when a request fails, times out, or memory exceeds a safety threshold.

Designed for Jetson Orin (unified CPU/GPU memory), but works on any machine running llama-server.

## Requirements

- Python 3.7+
- `requests`
- `jetson-stats` (Jetson only — skip with `--no-jtop` on other hardware)

## Install

```bash
# Clone
git clone https://github.com/moontato/ctx-test-py
cd ctx-test-py

# Install dependencies (system Python — recommended on Jetson)
pip3 install requests --break-system-packages

# Jetson only
pip3 install jetson-stats --break-system-packages
sudo systemctl enable --now jtop.service

# Or use a venv with system site-packages (gives access to system jtop)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install requests
```

## Configuration

Settings can be provided via a `.ini` file, CLI flags, or both. CLI flags always override the config file. `url` and `model` are required from one source or the other.

### Config file

Copy and edit the included sample:

```bash
cp ctx_test.ini myconfig.ini
```

```ini
[ctx_test]
url       = http://127.0.0.1:8080
model     = my-model-alias

# Optional — these are the defaults if omitted
# start     = 8000       # first test size in tokens
# step      = 8000       # increment per step
# max       = 256000     # hard ceiling
# threshold = 58.0       # stop if used memory (GB) exceeds this
# timeout   = 300        # seconds to wait per request
```

Find your model alias:

```bash
curl http://127.0.0.1:8080/v1/models | python3 -m json.tool
```

## Adding to PATH

To call the script from anywhere as `ctx_test`:

```bash
# Make executable
chmod +x /path/to/ctx-test-py/ctx_test.py

# Symlink into a directory already on your PATH
sudo ln -s /path/to/ctx-test-py/ctx_test.py /usr/local/bin/ctx_test
```

Then from anywhere:

```bash
ctx_test --config /path/to/myconfig.ini
```

## Usage

```bash
# Make executable (optional)
chmod +x ctx_test.py

# Run with config file
./ctx_test.py --config myconfig.ini

# Run with CLI flags only
./ctx_test.py --url http://127.0.0.1:8080 --model my-model-alias

# CLI flags override config file values
./ctx_test.py --config myconfig.ini --step 16000 --threshold 60.0

# Start from a specific token count (skip lower sizes already tested)
./ctx_test.py --config myconfig.ini --start 64000

# Skip jtop memory readings (non-Jetson machines)
./ctx_test.py --config myconfig.ini --no-jtop

# Full options
./ctx_test.py --help
```

### Test modes

At startup the script asks you to choose a test mode:

**[1] Cold prefill** — rebuilds the prompt from scratch each step with `cache_prompt: false`. Measures worst-case memory: what a brand-new conversation at that context size would cost. Slower.

**[2] Incremental** — extends the prompt from the previous step with `cache_prompt: true`. The server reuses the KV cache for the prefix, only prefilling new tokens each step. Faster, and shows incremental memory growth.

## Output

Results are printed as a table and saved to a timestamped JSON file:

```
Tokens     Used GB      GPU sh GB    Free GB      Time (s)   Status
----------------------------------------------------------------------
50,101     28.26        23.32        22.39        106.1      OK
75,101     30.32        25.34        20.33        174.4      OK
```

- **Used GB** — RAM in use (`RAM.used` from jtop)
- **GPU sh GB** — memory mapped to GPU (`RAM.shared` from jtop)
- **Free GB** — remaining available memory
- **Status** — `OK` or `FAIL`

### Interpreting results

- Memory growth between steps should be small if scratch buffers are reused. Large jumps (>2 GB) indicate new allocator scratch allocations that won't be released until llama-server restarts.
- `elapsed_s` grows non-linearly with context size.
- **FAIL with no prior OKs** usually means a config issue: wrong model alias, server not in router mode, or `--ctx-size` too low. Test with `--start 1000` first.

### Failure modes

Check llama-server logs to distinguish FAIL causes:

```bash
journalctl -u llama-server -n 50
```

| Symptom | Likely cause |
|---|---|
| HTTP 400 | Prompt exceeds `--ctx-size`, or missing `model` field |
| HTTP 503 | No available slots (another request running) |
| Connection error | llama-server crashed (OOM, segfault) |
| Timeout | Prefill took longer than `timeout` — increase it |

## After running

The script leaves llama-server in a high-memory state (CUDA scratch high-water mark). Restart before resuming normal use:

```bash
sudo systemctl restart llama-server
```
