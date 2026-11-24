#!/usr/bin/env python3
"""Kaggle TPU v5e-8 smoke test for the llama-tqp-mas-tpu tree.

Run this in a Kaggle notebook with accelerator = TPU v5e-8.
It verifies a TPU is actually attached, downloads xbill9's prebuilt
llama-server + ggml-tpu backend (v0.2.0), fetches a small Q4_0 GGUF,
serves it on device TPU0 and prints the decode speed.

No build is needed for this check. It tests the TPU + backend path first,
before spending time building the TurboQuant/MAS overlay on the TPU VM.
"""
import glob
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.request

WORK = pathlib.Path(os.environ.get("TPU_PROBE_WORK", "/kaggle/working/tpu-probe"))
WORK.mkdir(parents=True, exist_ok=True)
RELEASE = "https://github.com/xbill9/ggml-tpu/releases/download/v0.2.0/ggml-tpu-98107ca-llama.cpp-7182ef71e-linux-x86_64.tar.gz"
REPO_MODEL = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-0.5b-instruct-q4_0.gguf"
PORT = 8090


def log(*a):
    print("[tpu-probe]", *a, flush=True)


def check_tpu():
    import jax
    devices = jax.devices()
    log("jax devices:", [(d.platform, d.device_kind, str(d)) for d in devices])
    tpu = [d for d in devices if d.platform == "tpu"]
    if not tpu:
        raise RuntimeError("No TPU device attached - Kaggle did not grant the v5e-8 accelerator")
    log("TPU OK:", tpu[0])
    return tpu[0]


def download(url, dest):
    if dest.exists():
        return dest
    log("downloading", url)
    urllib.request.urlretrieve(url, str(dest))
    return dest


def find_libtpu():
    hits = []
    for root in ("/usr/local/lib", "/usr/lib", str(pathlib.Path.home())):
        hits += glob.glob(root + "/**/libtpu/libtpu.so", recursive=True)
        hits += glob.glob(root + "/**/libtpu.so", recursive=True)
    try:
        import libtpu
        hits.insert(0, str(pathlib.Path(libtpu.__file__).parent / "libtpu.so"))
    except Exception:
        pass
    hits = [h for h in hits if pathlib.Path(h).exists()]
    if not hits:
        raise RuntimeError("libtpu.so not found; install jax[tpu] (preinstalled on Kaggle TPU notebooks)")
    log("libtpu:", hits[0])
    return hits[0]


def main():
    check_tpu()
    tgz = download(RELEASE, WORK / "backend.tgz")
    releasedir = WORK / "release"
    if not (releasedir / "llama-server").exists():
        releasedir.mkdir(exist_ok=True)
        log("extracting backend")
        with subprocess.Popen(["tar", "xzf", str(tgz), "-C", str(releasedir)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
            out = p.communicate()[0]
            if p.returncode:
                raise RuntimeError(out[-2000:])
    root = next((p for p in releasedir.iterdir() if p.is_dir()), releasedir)
    server = root / "llama-server"
    server.chmod(0o755)
    log("backend dir:", root)

    from huggingface_hub import hf_hub_download
    log("fetching model", REPO_MODEL, MODEL_FILE)
    model = hf_hub_download(repo_id=REPO_MODEL, filename=MODEL_FILE, local_dir=str(WORK / "models"))
    log("model:", model)

    cache = WORK / "programs"
    cache.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(root) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    env["GGML_BACKEND_PATH"] = str(root / "libggml-tpu.so")
    env["GGML_TPU_EXECUTOR"] = "pjrt"
    env["GGML_TPU_LIBTPU"] = find_libtpu()
    env["GGML_TPU_CACHE_DIR"] = str(cache)
    env["GGML_TPU_QUANT"] = "bf16"
    env["GGML_TPU_PRECISION"] = "DEFAULT"
    env["GGML_TPU_VERBOSE"] = "1"

    cmd = [str(server), "-m", str(model), "-dev", "TPU0", "-ngl", "99", "-fa", "off",
           "-c", "1024", "--host", "127.0.0.1", "--port", str(PORT), "--parallel", "1", "--metrics"]
    log("launching:", " ".join(cmd))
    logfile = open(WORK / "server.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logfile, stderr=subprocess.STDOUT)
    try:
        base = "http://127.0.0.1:%d" % PORT
        deadline = time.time() + 180
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("server exited early; tail of server.log:\n" + (WORK / "server.log").read_text(errors="replace")[-3000:])
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError("server did not become healthy in 180s")
        log("server healthy")

        payload = json.dumps({"prompt": "The capital of France is", "max_tokens": 24, "temperature": 0}).encode()
        req = urllib.request.Request(base + "/v1/completions", data=payload, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.load(r)
        dt = time.time() - t0
        text = resp["choices"][0]["text"]
        timings = resp.get("timings", {})
        log("RESPONSE:", repr(text))
        log("request seconds: %.1f" % dt)
        log("decode tok/s:", timings.get("predicted_per_second"))
        log("tokens:", timings.get("predicted_n"))
        log("PROBE RESULT: PASS" if "Paris" in text else "PROBE RESULT: CHECK OUTPUT (no 'Paris' in response)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        logfile.close()
        print("--- server.log tail ---", flush=True)
        print((WORK / "server.log").read_text(errors="replace")[-3000:], flush=True)


if __name__ == "__main__":
    main()
