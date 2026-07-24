#!/usr/bin/env python
"""Download MUSE-News / MUSE-Books into the shared HF cache.

Core configs for unlearning eval + forget/retain corpora:
  knowmem (knowledge-mem QA), verbmem (verbatim), privleak (MIA), raw (corpora),
  train (retain training corpus).
News-only scal/sust (scaling ablations, large) are skipped by default.
"""
import os

os.environ.setdefault("HF_HOME", "/rdata/minsoo3.kim/hf_home")
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset  # noqa: E402

CONFIGS = ["knowmem", "verbmem", "privleak", "raw", "train"]
REPOS = ["muse-bench/MUSE-Books", "muse-bench/MUSE-News"]


def main():
    for repo in REPOS:
        for cfg in CONFIGS:
            try:
                ds = load_dataset(repo, cfg)
                splits = {k: len(v) for k, v in ds.items()}
                print(f"[ok] {repo}:{cfg} splits={splits}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {repo}:{cfg} -> {type(e).__name__}: {str(e)[:140]}", flush=True)
    print("MUSE download done", flush=True)


if __name__ == "__main__":
    main()
