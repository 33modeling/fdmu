#!/usr/bin/env python
"""Download every dataset/encoder the gate experiment needs into ~/rdata/hf_home.

- locuslab/TOFU  configs: full, forget10_perturbed  (TOFU adapter)
- jinzhuoran/RWKU configs: forget_target, forget_level1/2, neighbor_level1/2 (RWKU adapter)
- sentence-transformers/all-MiniLM-L6-v2 (knn_embed selector)
Run inside the repo venv.
"""
import os

os.environ.setdefault("HF_HOME", "/rdata/minsoo3.kim/hf_home")
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")

from datasets import load_dataset  # noqa: E402

TOFU = [("full", "train"), ("forget10_perturbed", "train")]
RWKU = [
    ("forget_target", "train"),
    ("forget_level1", "test"),
    ("forget_level2", "test"),
    ("neighbor_level1", "test"),
    ("neighbor_level2", "test"),
]


def grab(repo, config, split):
    print(f"[dl] {repo}:{config} ...", flush=True)
    ds = load_dataset(repo, config)[split]
    print(f"     {repo}:{config} rows={len(ds)}", flush=True)


def main():
    for cfg, split in TOFU:
        grab("locuslab/TOFU", cfg, split)
    for cfg, split in RWKU:
        grab("jinzhuoran/RWKU", cfg, split)
    # sentence encoder for knn_embed selector
    print("[dl] sentence-transformers/all-MiniLM-L6-v2 ...", flush=True)
    from sentence_transformers import SentenceTransformer

    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("all downloads complete", flush=True)


if __name__ == "__main__":
    main()
