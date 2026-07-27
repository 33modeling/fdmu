"""1.5B TOFU gate experiment: the cheapest real-data forecast of Tables 1-2.

One forget10 author on Qwen2.5-1.5B-Instruct: SFT-memorize the request
universe, seal audit-fold scores, run independent generators (NPO/GradDiff/
RMU), unseal, fill a mini Table 1; then compare NPO / NPO+transplant / ours
for a mini Table 2 on the untouched random audit fold.

GPU (H100) run:   python experiments/gate_1p5b/gate.py --device cuda
CPU smoke:        python experiments/gate_1p5b/gate.py --smoke
Artifacts land in runs/gate_<tag>/ (tables + JSON + seal ledger).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from rsus.analysis.prediction import table1_rows  # noqa: E402
from rsus.blocks import mlp_down_last_layers, set_trainable_  # noqa: E402
from rsus.data.base import collate  # noqa: E402
from rsus.data.indexed import build_indexed_request  # noqa: E402
from rsus.data.tofu import (  # noqa: E402
    FORGET10_FIRST_AUTHOR,
    idk_variants,
    load_tofu_examples,
    load_tofu_paraphrases,
    tofu_request,
)
from rsus.evalx.metrics import mean_recall  # noqa: E402
from rsus.generators.s2s import S2SConfig, run_s2s_trajectory  # noqa: E402
from rsus.evalx.protection import evaluate_protection  # noqa: E402
from rsus.generators import TrajectoryConfig, run_trajectory  # noqa: E402
from rsus.generators.ours import OursConfig, run_ours_trajectory  # noqa: E402
from rsus.generators.repaired import RepairedConfig, run_engine_repaired  # noqa: E402
from rsus.losses import seq_mean_answer_nll  # noqa: E402
from rsus.partition import PartitionParams, build_partition, make_folds  # noqa: E402
from rsus.probe.base import ProbeSpec, get_scorer, scorer_names  # noqa: E402
from rsus.sealing import seal_scores, unseal  # noqa: E402
from rsus.sft_cache import contract_sha256, resolve_sft_cache_path  # noqa: E402
from rsus.stage1 import Stage1Config, calibrate_floor  # noqa: E402
from rsus.stage2 import Stage2Config  # noqa: E402

GENERATORS = ["npo", "graddiff", "rmu"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--model-id", default="", help="stable campaign alias stored in the manifest")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument(
        "--dataset",
        default="tofu",
        choices=[
            "tofu",
            "rwku",
            "wmdp_bio_mmlu",
            "muse_news",
            "muse_books",
            "pistol",
        ],
        help=(
            "registered request adapter; --author is its integer request index "
            "and --candidate-authors is its frozen retained-group index pool"
        ),
    )
    p.add_argument(
        "--pistol-sample",
        type=int,
        choices=[1, 2],
        default=2,
        help="official PISTOL sample used by --dataset pistol (default: 2)",
    )
    p.add_argument("--author", type=int, default=FORGET10_FIRST_AUTHOR)
    p.add_argument("--universe-authors", type=int, default=30)
    p.add_argument(
        "--candidate-authors",
        default="",
        help="exact retained-author pool, e.g. '1,3,5-9'; campaign use prevents "
             "development/audit candidate overlap",
    )
    p.add_argument("--block-last-n", type=int, default=8)
    p.add_argument(
        "--trainable-scope",
        default="full",
        choices=["full", "probe_block"],
        help="parameters updated by SFT and every T1 objective. 'probe_block' makes "
             "the parent update support identical to the block measured by gradient probes; "
             "recommended for one-GPU fp32 7B channel-matrix runs",
    )
    p.add_argument("--eta", type=float, default=3e-4)
    p.add_argument("--probe-norm-eta", type=float, default=3e-3,
                   help="FD radius for fd_norm (larger than --eta): random projections "
                        "give small g.v, so at 3e-4 the fp32 loss-difference cancellation "
                        "inflates the squared estimate ~4x; 3e-3 recovers ||g||^2 (fidelity gate)")
    p.add_argument("--probe-dirs", type=int, default=64,
                   help="random directions K for norm-estimating scorers (fd_norm): "
                        "2K forward sweeps, relative estimator variance ~2/K "
                        "(default 64 = the frozen fd_norm operating point, "
                        "prereg FREEZE-2026-07-21-channel-interaction)")
    p.add_argument("--probe-seed", type=int, default=0,
                   help="seed for the probe's shared random directions "
                        "(fixed at 0 for all sealed runs to date; expose for "
                        "direction-seed robustness checks)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sft-lr", type=float, default=1e-5)
    p.add_argument("--sft-steps", type=int, default=400)
    p.add_argument("--sft-target-loss", type=float, default=0.8)
    p.add_argument("--sft-eval-every", type=int, default=100,
                   help="full memorization-set gate interval; avoids stopping on a lucky minibatch")
    p.add_argument(
        "--sft-cache",
        default="auto",
        help=(
            "request-level CPU state cache: 'auto' (default) stores under "
            "runs/sft_cache using the full SFT contract, an explicit .pt path "
            "uses that file, and 'off' disables reuse"
        ),
    )
    p.add_argument(
        "--require-sft-target",
        action="store_true",
        help="abort before scoring if the full memorization set remains above "
             "--sft-target-loss; enabled by sealed channel-matrix campaigns",
    )
    p.add_argument("--gen-lr", type=float, default=2e-6)
    p.add_argument("--gen-steps", type=int, default=60)
    p.add_argument("--gen-steps-per", default="",
                   help="per-generator step overrides, e.g. 'npo=240,rmu=120' "
                        "(unlisted generators keep --gen-steps)")
    p.add_argument("--gen-lr-per", default="",
                   help="per-generator lr overrides, e.g. 'npo=6e-6' "
                        "(unlisted generators keep --gen-lr)")
    p.add_argument("--gen-beta-per", default="",
                   help="per-generator beta overrides, e.g. 'npo=0.1,simnpo=4.5'")
    p.add_argument("--gen-forget-weight-per", default="",
                   help="per-generator forget-term weights, e.g. 'npo=0.125,simnpo=0.125'")
    p.add_argument("--gen-retain-weight-per", default="",
                   help="per-generator retain-term weights, e.g. 'npo=1,simnpo=1'")
    p.add_argument("--gen-rmu-alpha-per", default="",
                   help="per-generator representation retain weights (RMU/RepNoise/RR)")
    p.add_argument("--gen-rmu-c-per", default="",
                   help="per-generator representation target magnitudes (RMU/RepNoise)")
    p.add_argument(
        "--gen-rep-retain-mode",
        default="fixed",
        choices=["fixed", "stream_cached"],
        help="representation retain reference: legacy fixed minibatch or a cached "
             "development-pool stream (recommended for the 7B campaign)",
    )
    p.add_argument("--generators", default="npo,graddiff,rmu",
                   help="Table-1 T1 generators whose damage is predicted (each becomes an "
                        "objective column). Output/loss-gradient channel: ga,graddiff,npo,"
                        "simnpo,idkdpo,gru; representation channel: rmu,repnoise,"
                        "circuit_breakers. Drop near-static arms only by a predeclared rule.")
    p.add_argument(
        "--predictors",
        default="fd,fd_norm,knn_feature,knn_embed,knn_lexical,grad_norm,random_rank",
        help="comma-separated sealed predictor roster; knn_embed is skipped with an explicit "
             "log entry if sentence-transformers is unavailable",
    )
    p.add_argument(
        "--require-all-predictors",
        action="store_true",
        help="fail before writing any seal if a requested predictor dependency is unavailable",
    )
    p.add_argument(
        "--sentence-encoder",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="frozen local path or model id for knn_embed; recorded in the manifest",
    )
    p.add_argument("--extra-predictors", default="",
                   help="comma-separated additional registered scorers to run and "
                        "seal alongside the default roster, e.g. 'jvp,vmap_graddot,grad_cosine'")
    p.add_argument("--attn-impl", default="",
                   help="attention implementation passed to from_pretrained "
                        "(forced to 'eager' when jvp is requested: SDPA kernels "
                        "do not support forward-mode AD)")
    p.add_argument("--gen-ckpt-every", type=int, default=10)
    p.add_argument(
        "--gen-recall-gate",
        type=float,
        default=0.10,
        help="direct forgetting threshold; T1 damage is measured at the first "
             "saved checkpoint at or below this value",
    )
    p.add_argument("--s1-lr", type=float, default=1e-5)
    p.add_argument("--s1-max-steps", type=int, default=600)
    p.add_argument("--partition-predictor", default="fd",
                   help="scored predictor whose profile builds the protect partition "
                        "for ours/npo_transplant (e.g. grad_norm, knn_feature). "
                        "fd is the preregistered default; it anticorrelates with damage "
                        "on 1.5B TOFU, mis-targeting the protect pool.")
    p.add_argument("--s1-recall-gate", type=float, default=0.0,
                   help="if >0, stage-1 (ours/s2s) must also push forget argmax recall "
                        "to <= this before exiting — aligns its stopping point with the "
                        "T2 common criterion (recall_max=0.10). 0 = floor-only exit.")
    p.add_argument("--s2-steps", type=int, default=80)
    p.add_argument("--s2-delta-seq", type=float, default=1e-2,
                   help="stage-2 forget-drift budget D_seq (RUNBOOK D4 placeholder; the "
                        "binding constraint on repair depth — rf25c froze at +0.14 nats "
                        "of repair when this was exhausted)")
    p.add_argument("--s2-delta-tok", type=float, default=1e-1,
                   help="stage-2 token-level drift budget (D4 placeholder)")
    p.add_argument("--s2-eta2", type=float, default=0.0,
                   help="stage-2 repair step size (0 = Stage2Config default 5e-3, a raw "
                        "momentum-SGD step ported from the CPU toy world; repair1 showed "
                        "it inflicts 18-130 nats of audit damage on 1.5B — try 1e-4/1e-5)")
    p.add_argument("--t2-steps", type=int, default=0,
                   help="step budget for single-stage T2 arms (0 = inherit --gen-steps)")
    p.add_argument("--t2-lr", type=float, default=0.0,
                   help="lr for single-stage T2 arms (0 = inherit --gen-lr)")
    p.add_argument("--t2-lr-per", default="",
                   help="per-method T2 lr overrides, e.g. 'npo=8e-6,simnpo=8e-6' "
                        "(unlisted methods keep --t2-lr / --gen-lr)")
    p.add_argument("--beta", type=float, default=0.0,
                   help="NPO/SimNPO/IdkDPO temperature for T1 generators and T2 arms "
                        "(0 = keep TrajectoryConfig default 1.0; NPO paper standard is 0.1 — "
                        "beta=1.0 kills ascent pressure ~2 nats above reference)")
    p.add_argument("--pool-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument(
        "--t2-roster",
        default="ga,graddiff,npo,simnpo,idkdpo,rmu,gru,s2s,npo_transplant,ours",
        help="comma-separated Table-2 methods",
    )
    p.add_argument("--smoke", action="store_true", help="tiny random model, CPU-sized budgets")
    p.add_argument("--run-tag", default="", help="suffix for the run directory (rerun without clobbering seals)")
    p.add_argument("--out-dir", default="",
                   help="explicit run directory (preferred by campaign launchers); overrides model/tag naming")
    return p.parse_args()


def apply_smoke(a):
    a.universe_authors = 4
    a.block_last_n = 1
    a.sft_steps = 60
    a.sft_lr = 5e-3
    a.sft_target_loss = 1.2
    a.gen_lr = 1e-3
    a.gen_steps = 10
    a.gen_ckpt_every = 5
    a.s1_lr = 2e-3
    a.s1_max_steps = 250
    a.s2_steps = 20
    a.pool_size = 8
    a.batch_size = 4


def parse_int_ranges(raw: str) -> list[int]:
    values: list[int] = []
    for item in (part.strip() for part in raw.split(",") if part.strip()):
        if "-" in item:
            lo_raw, hi_raw = item.split("-", 1)
            lo, hi = int(lo_raw), int(hi_raw)
            if hi < lo:
                raise ValueError(f"descending integer range is not allowed: {item!r}")
            values.extend(range(lo, hi + 1))
        else:
            values.append(int(item))
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate ids in integer range list: {raw!r}")
    return values


def load_model(a, tokenizer):
    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    if a.smoke:
        torch.manual_seed(a.seed)
        cfg = Qwen2Config(
            vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=512, pad_token_id=tokenizer.pad_token_id,
        )
        return Qwen2ForCausalLM(cfg).to(dtype).eval()
    kw = {"torch_dtype": dtype}
    if a.attn_impl:
        kw["attn_implementation"] = a.attn_impl
    m = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    return m.to(a.device).eval()


def clear_cuda_cache() -> None:
    """Release tensors owned by completed scorers/trajectories.

    Gate runs intentionally serialize GPU work: the SFT target is stored on
    CPU and every scorer or trajectory receives one disposable model.  The
    collection before ``empty_cache`` is important because otherwise a model
    caught in a short Python reference cycle can survive into the next arm.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _mean_example_nll(model, examples, batch_size: int, device: str) -> float:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = collate(examples[start:start + batch_size])
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            values.extend(seq_mean_answer_nll(model, batch).detach().float().cpu().tolist())
    if not values:
        raise ValueError("cannot evaluate SFT target on an empty example set")
    return sum(values) / len(values)


def sft(model, examples, a, log, trainable_block=None) -> dict[str, float | int | bool]:
    # The foreach AdamW implementation holds an additional tensor list about
    # the size of the parameters during opt.step().  The scalar implementation
    # has the same update rule and a lower transient peak, which matters for a
    # full-model 7B update on one 80GB GPU.
    trainable = set_trainable_(model, trainable_block)
    opt = torch.optim.AdamW(trainable.values(), lr=a.sft_lr, foreach=False)
    log(f"  SFT trainable tensors={len(trainable)} params="
        f"{sum(p.numel() for p in trainable.values()):,} scope={a.trainable_scope}")
    gen = torch.Generator().manual_seed(a.seed)
    loss = None
    batch = None
    step = 0
    for step in range(1, a.sft_steps + 1):
        idx = torch.randperm(len(examples), generator=gen)[: a.batch_size]
        batch = {k: v.to(a.device) if torch.is_tensor(v) else v
                 for k, v in collate([examples[i] for i in idx.tolist()]).items()}
        loss = seq_mean_answer_nll(model, batch).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 20 == 0:
            log(f"  sft step {step} loss {float(loss.detach()):.3f}")
        if step % a.sft_eval_every == 0:
            interim = _mean_example_nll(model, examples, a.batch_size, a.device)
            log(f"  sft step {step} full-set mean NLL {interim:.3f}")
            if interim <= a.sft_target_loss:
                break
    model.zero_grad(set_to_none=True)
    del opt
    del loss
    del batch
    clear_cuda_cache()
    full_mean = _mean_example_nll(model, examples, a.batch_size, a.device)
    reached = full_mean <= a.sft_target_loss
    log(f"  SFT full-set mean NLL={full_mean:.3f} target={a.sft_target_loss:.3f} "
        f"reached={reached}")
    return {"steps": step, "full_mean_nll": full_mean, "target": a.sft_target_loss,
            "reached": reached}


def _sft_cache_contract(a, req, probe_block, trainable_keys) -> dict:
    state_scope = (
        "trainable_block" if a.trainable_scope == "probe_block" else "full_model"
    )
    return {
        "schema": "sft-cache-v3",
        "model": a.model,
        "dtype": a.dtype,
        "attn_impl": a.attn_impl,
        "smoke": a.smoke,
        "request": req.request_id,
        "candidate_universe_sha": req.universe.sha,
        "forget_sha": req.forget_sha,
        "trainable_scope": a.trainable_scope,
        "trainable_pattern": (
            probe_block.pattern if a.trainable_scope == "probe_block" else None
        ),
        "state_scope": state_scope,
        "state_keys": (
            sorted(trainable_keys) if state_scope == "trainable_block" else None
        ),
        "sft_lr": a.sft_lr,
        "sft_steps": a.sft_steps,
        "sft_target_loss": a.sft_target_loss,
        "sft_eval_every": a.sft_eval_every,
        "batch_size": a.batch_size,
        "seed": a.seed,
    }


@contextmanager
def _sft_cache_guard(path: Path, *, exclusive: bool):
    """Serialize readers/writers that share one request-level cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    guard_path = path.with_suffix(path.suffix + ".guard")
    with guard_path.open("a+b") as guard:
        fcntl.flock(
            guard.fileno(),
            fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
        )
        try:
            yield
        finally:
            fcntl.flock(guard.fileno(), fcntl.LOCK_UN)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Make renames durable when the backing filesystem supports directory fsync."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(descriptor)


def _sft_cache_local_stage(
    path: Path,
    state_bytes: int,
    token: str,
    log,
) -> Path | None:
    """Choose bounded node-local staging when it has enough free space."""
    configured = os.environ.get("SFT_CACHE_LOCAL_TMPDIR", "").strip()
    root = (
        Path(os.path.expandvars(os.path.expanduser(configured)))
        if configured
        else Path(tempfile.gettempdir()) / "fdmu-sft-cache"
    )
    required = state_bytes + 1024 * 2**20
    try:
        root.mkdir(parents=True, exist_ok=True)
        stale_before = time.time() - 24 * 60 * 60
        for stale in root.glob(f"{path.name}.*.stage"):
            try:
                stat = stale.stat()
            except FileNotFoundError:
                continue
            if stat.st_mtime <= stale_before:
                stale.unlink()
                log(
                    f"removed stale local SFT cache stage path={stale} "
                    f"age_hours={(time.time() - stat.st_mtime) / 3600:.1f} "
                    f"size_gib={stat.st_size / 2**30:.2f}"
                )
        free_bytes = shutil.disk_usage(root).free
    except OSError as error:
        log(
            f"SFT cache local staging unavailable path={root} "
            f"error={type(error).__name__}: {error}; using shared temporary"
        )
        return None
    if free_bytes < required:
        log(
            f"SFT cache local staging skipped path={root} "
            f"need_gib={required / 2**30:.2f} free_gib={free_bytes / 2**30:.2f}; "
            "using shared temporary"
        )
        return None
    stage = root / f"{path.name}.{token}.stage"
    log(
        f"SFT cache staging path={stage} "
        f"free_gib={free_bytes / 2**30:.2f}"
    )
    return stage


def _remove_stale_sft_temporaries_locked(path: Path, log) -> None:
    """Remove only old writer debris; live writers serialize outside the lock."""
    meta_path = path.with_suffix(path.suffix + ".json")
    candidates = {
        path.with_suffix(path.suffix + ".tmp"),
        meta_path.with_suffix(meta_path.suffix + ".tmp"),
        *path.parent.glob(f".{path.name}.*.tmp"),
        *meta_path.parent.glob(f".{meta_path.name}.*.tmp"),
    }
    stale_before = time.time() - 24 * 60 * 60
    for stale in candidates:
        try:
            stat = stale.stat()
        except FileNotFoundError:
            continue
        if stat.st_mtime > stale_before:
            continue
        size_gib = stat.st_size / 2**30
        stale.unlink()
        log(
            f"removed stale SFT cache temporary "
            f"path={stale} age_hours={(time.time() - stat.st_mtime) / 3600:.1f} "
            f"size_gib={size_gib:.2f}"
        )


def _quarantine_sft_cache_locked(path: Path, reason: str, log) -> Path:
    """Preserve a known-bad final cache pair while holding its exclusive lock."""
    meta_path = path.with_suffix(path.suffix + ".json")
    runs_root = Path(os.environ.get("CLUSTER_RUNS_ROOT", ROOT / "runs"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = (
        runs_root
        / "forensics"
        / "sft-cache-corrupt"
        / f"{stamp}__{path.stem}__{uuid.uuid4().hex[:8]}"
    )
    destination.mkdir(parents=True, exist_ok=False)
    moved = []
    for artifact in (path, meta_path):
        if artifact.exists():
            target = destination / artifact.name
            shutil.move(str(artifact), str(target))
            moved.append(str(target))
    manifest = {
        "schema": "sft-cache-quarantine-v1",
        "source_state": str(path),
        "source_metadata": str(meta_path),
        "reason": reason,
        "quarantined_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": moved,
    }
    (destination / "quarantine.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    log(
        f"QUARANTINED corrupt SFT cache reason={reason} "
        f"destination={destination}"
    )
    return destination


def _known_sft_cache_corruption(error: Exception) -> bool:
    if isinstance(error, (EOFError, pickle.UnpicklingError, UnicodeDecodeError)):
        return True
    if not isinstance(error, RuntimeError):
        return False
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "unexpected pos",
            "pytorchstreamreader failed",
            "failed finding central directory",
            "invalid header or archive is corrupted",
            "pickle data was truncated",
            "invalid magic number",
        )
    )


def _validate_sft_result(result) -> str | None:
    if not isinstance(result, dict):
        return "metadata has no valid sft_result"
    for field in ("steps", "full_mean_nll", "target", "reached"):
        if field not in result:
            return f"metadata sft_result is missing {field!r}"
    if not isinstance(result["steps"], int) or isinstance(result["steps"], bool):
        return "metadata sft_result has invalid steps"
    if not isinstance(result["full_mean_nll"], (int, float)):
        return "metadata sft_result has invalid full_mean_nll"
    if not isinstance(result["target"], (int, float)):
        return "metadata sft_result has invalid target"
    if not isinstance(result["reached"], bool):
        return "metadata sft_result has invalid reached"
    return None


def _sft_expected_state(model, contract: dict) -> tuple[dict, str | None]:
    state_scope = contract.get("state_scope", "full_model")
    if state_scope == "full_model":
        return dict(model.state_dict()), None
    if state_scope != "trainable_block":
        return {}, f"unsupported SFT cache state_scope {state_scope!r}"
    keys = contract.get("state_keys")
    if (
        not isinstance(keys, list)
        or not keys
        or not all(isinstance(key, str) and key for key in keys)
        or len(keys) != len(set(keys))
    ):
        return {}, "trainable-block cache contract has invalid state_keys"
    parameters = dict(model.named_parameters())
    unknown = sorted(set(keys) - set(parameters))
    if unknown:
        return {}, f"trainable-block cache contract has unknown keys {unknown[:5]}"
    return {name: parameters[name] for name in keys}, None


def _snapshot_sft_state(model, contract: dict) -> dict:
    """Clone only state that SFT is permitted to modify."""
    expected, error = _sft_expected_state(model, contract)
    if error is not None:
        raise RuntimeError(error)
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in expected.items()
    }


def _validate_sft_state_dict(model, state, contract: dict) -> str | None:
    """Check the scoped state contract before mutating the live model."""
    if not isinstance(state, dict):
        return f"loaded state is {type(state).__name__}, expected dict"
    expected, contract_error = _sft_expected_state(model, contract)
    if contract_error is not None:
        return contract_error
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        return (
            f"state-dict keys mismatch missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    for name, expected_tensor in expected.items():
        cached_tensor = state[name]
        if not torch.is_tensor(cached_tensor):
            return f"state-dict entry {name!r} is not a tensor"
        if tuple(cached_tensor.shape) != tuple(expected_tensor.shape):
            return (
                f"state-dict shape mismatch for {name!r}: "
                f"expected={tuple(expected_tensor.shape)} "
                f"actual={tuple(cached_tensor.shape)}"
            )
        if cached_tensor.dtype != expected_tensor.dtype:
            return (
                f"state-dict dtype mismatch for {name!r}: "
                f"expected={expected_tensor.dtype} actual={cached_tensor.dtype}"
            )
    return None


def _apply_sft_state(model, state: dict, contract: dict) -> None:
    state_scope = contract.get("state_scope", "full_model")
    model.load_state_dict(state, strict=state_scope == "full_model")


def _read_sft_cache_metadata_locked(
    path: Path,
    contract: dict,
    log,
) -> dict | None:
    """Validate the final pair without converting generic I/O into a cache miss."""
    meta_path = path.with_suffix(path.suffix + ".json")
    state_exists = path.exists()
    metadata_exists = meta_path.exists()
    if not state_exists and not metadata_exists:
        return None
    if not state_exists or not metadata_exists:
        _quarantine_sft_cache_locked(path, "incomplete final cache pair", log)
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        _quarantine_sft_cache_locked(
            path,
            f"invalid metadata JSON: {error}",
            log,
        )
        return None
    if not isinstance(metadata, dict):
        _quarantine_sft_cache_locked(path, "metadata is not a JSON object", log)
        return None
    if metadata.get("contract") != contract:
        _quarantine_sft_cache_locked(
            path,
            "SFT cache contract mismatch; preserving old pair before rebuild",
            log,
        )
        return None
    result_error = _validate_sft_result(metadata.get("sft_result"))
    if result_error is not None:
        _quarantine_sft_cache_locked(path, result_error, log)
        return None

    integrity = metadata.get("integrity")
    if integrity is None:
        log(f"loading legacy SFT cache without integrity metadata: {path}")
        return metadata
    if not isinstance(integrity, dict):
        _quarantine_sft_cache_locked(path, "integrity metadata is not an object", log)
        return None
    if integrity.get("algorithm") != "sha256":
        raise RuntimeError(
            f"unsupported SFT cache integrity algorithm at {path}: "
            f"{integrity.get('algorithm')!r}"
        )
    expected_size = integrity.get("size_bytes")
    expected_sha = integrity.get("sha256")
    if not isinstance(expected_size, int) or expected_size < 0:
        _quarantine_sft_cache_locked(path, "invalid integrity size_bytes", log)
        return None
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        _quarantine_sft_cache_locked(path, "invalid integrity sha256", log)
        return None
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        _quarantine_sft_cache_locked(
            path,
            f"size mismatch expected={expected_size} actual={actual_size}",
            log,
        )
        return None
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        _quarantine_sft_cache_locked(
            path,
            f"sha256 mismatch expected={expected_sha} actual={actual_sha}",
            log,
        )
        return None
    return metadata


def _load_sft_cache(model, path: Path, contract: dict, log) -> dict | None:
    with _sft_cache_guard(path, exclusive=True):
        metadata = _read_sft_cache_metadata_locked(path, contract, log)
        if metadata is None:
            return None
        try:
            try:
                state = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError as error:  # torch versions before weights_only
                message = str(error).lower()
                if "weights_only" not in message or "unexpected keyword" not in message:
                    raise
                state = torch.load(path, map_location="cpu")
        except Exception as error:
            if not _known_sft_cache_corruption(error):
                raise
            _quarantine_sft_cache_locked(
                path,
                f"{type(error).__name__}: {error}",
                log,
            )
            return None
        state_error = _validate_sft_state_dict(model, state, contract)
        if state_error is not None:
            del state
            _quarantine_sft_cache_locked(path, state_error, log)
            return None
    _apply_sft_state(model, state, contract)
    del state
    result = metadata["sft_result"]
    log(f"loaded validated development SFT cache {path} "
        f"(full-set mean NLL={result['full_mean_nll']:.3f})")
    return result


def _write_sft_cache(
    path: Path,
    contract: dict,
    result: dict,
    state: dict,
    log,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(path.suffix + ".json")
    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    tmp_state = path.with_name(f".{path.name}.{token}.tmp")
    tmp_meta = meta_path.with_name(f".{meta_path.name}.{token}.tmp")
    state_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in state.values()
        if torch.is_tensor(tensor)
    )
    local_stage = _sft_cache_local_stage(path, state_bytes, token, log)
    serialization_path = local_stage or tmp_state

    # Hold the cross-host lock only while inspecting or publishing final names.
    # A 14B serialization can take minutes and must not make other workers look
    # dead while no shared state is being changed.
    with _sft_cache_guard(path, exclusive=True):
        _remove_stale_sft_temporaries_locked(path, log)
        metadata = _read_sft_cache_metadata_locked(path, contract, log)
        if metadata is not None:
            log(f"reusing SFT cache written by a concurrent worker: {path}")
            return False

    free_bytes = shutil.disk_usage(path.parent).free
    log(
        f"writing SFT cache path={path} "
        f"estimated_gib={state_bytes / 2**30:.2f} "
        f"shared_free_gib={free_bytes / 2**30:.2f}"
    )
    if free_bytes < state_bytes + 1024 * 2**20:
        with _sft_cache_guard(path, exclusive=True):
            metadata = _read_sft_cache_metadata_locked(path, contract, log)
            if metadata is not None:
                log(f"reusing SFT cache written by a concurrent worker: {path}")
                return False
        raise RuntimeError(
            f"insufficient disk for SFT cache {path}: "
            f"need>{(state_bytes + 1024 * 2**20) / 2**30:.2f} GiB, "
            f"free={free_bytes / 2**30:.2f} GiB"
        )

    try:
        for attempt in range(1, 3):
            serialization_path.unlink(missing_ok=True)
            try:
                # The legacy sequential serializer never enters
                # inline_container.cc and is therefore suitable for very large
                # state dictionaries and filesystems with fragile seek support.
                log(
                    f"serializing SFT cache attempt={attempt}/2 "
                    f"path={serialization_path} format=torch-legacy"
                )
                torch.save(
                    state,
                    serialization_path,
                    _use_new_zipfile_serialization=False,
                )
                with serialization_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                break
            except Exception as error:
                serialization_path.unlink(missing_ok=True)
                if attempt == 2 or not _known_sft_cache_corruption(error):
                    raise
                log(
                    f"RETRY SFT cache serialization after known PyTorch "
                    f"container failure attempt={attempt}/2 "
                    f"error={type(error).__name__}: {error}"
                )
        else:  # pragma: no cover - the loop either breaks or raises
            raise AssertionError("unreachable SFT cache serialization state")

        source_size = serialization_path.stat().st_size
        source_sha = _sha256_file(serialization_path)

        # Avoid copying a second 14B snapshot to shared storage if another
        # worker published while this worker was serializing locally.
        with _sft_cache_guard(path, exclusive=True):
            metadata = _read_sft_cache_metadata_locked(path, contract, log)
            if metadata is not None:
                log(f"reusing SFT cache written by a concurrent worker: {path}")
                return False

        published_sha = source_sha
        if local_stage is not None:
            with serialization_path.open("rb") as source, tmp_state.open("xb") as target:
                shutil.copyfileobj(source, target, length=8 * 2**20)
                target.flush()
                os.fsync(target.fileno())
            copied_size = tmp_state.stat().st_size
            copied_sha = _sha256_file(tmp_state)
            if copied_size != source_size or copied_sha != source_sha:
                raise RuntimeError(
                    f"SFT cache staging copy verification failed for {path}: "
                    f"source_size={source_size} copied_size={copied_size} "
                    f"source_sha256={source_sha} copied_sha256={copied_sha}"
                )
            published_sha = copied_sha

        integrity = {
            "algorithm": "sha256",
            "size_bytes": tmp_state.stat().st_size,
            "sha256": published_sha,
            "serialization": "torch-legacy",
        }
        tmp_meta.write_text(
            json.dumps(
                {
                    "contract": contract,
                    "sft_result": result,
                    "integrity": integrity,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        with tmp_meta.open("rb") as handle:
            os.fsync(handle.fileno())

        with _sft_cache_guard(path, exclusive=True):
            metadata = _read_sft_cache_metadata_locked(path, contract, log)
            if metadata is not None:
                log(f"reusing SFT cache written by a concurrent worker: {path}")
                return False
            os.replace(tmp_state, path)
            _fsync_directory(path.parent)
            os.replace(tmp_meta, meta_path)
            _fsync_directory(path.parent)
    finally:
        serialization_path.unlink(missing_ok=True)
        tmp_state.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)
    log(f"wrote development SFT cache {path}")
    return True


def main():
    a = parse_args()
    if not 0.0 <= a.gen_recall_gate <= 1.0:
        raise ValueError("--gen-recall-gate must be in [0, 1]")
    if a.smoke:
        apply_smoke(a)
    requested_predictors = [x.strip() for x in a.predictors.split(",") if x.strip()]
    requested_predictors += [x.strip() for x in a.extra_predictors.split(",") if x.strip()]
    if "jvp" in requested_predictors and not a.attn_impl:
        a.attn_impl = "eager"  # SDPA has no forward-AD support (jvp would crash)
    tag = "smoke" if a.smoke else a.model.split("/")[-1]
    if a.run_tag:
        tag += f"_{a.run_tag}"
    out = Path(a.out_dir).resolve() if a.out_dir else ROOT / "runs" / f"gate_{tag}"
    if (out / "seal_ledger.jsonl").exists():
        sys.exit(
            f"{out} already contains sealed scores (seals are append-only by design). "
            "Pass --run-tag <name> for a fresh run directory."
        )
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "gate.log"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    torch.manual_seed(a.seed)
    log(f"host={platform.node()} model={a.model} dtype={a.dtype} "
        f"trainable_scope={a.trainable_scope} seed={a.seed}")
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    candidate_authors = parse_int_ranges(a.candidate_authors) if a.candidate_authors else None
    if a.dataset == "tofu" and not candidate_authors:
        log(f"loading TOFU (universe_authors={a.universe_authors}) ...")
        examples = load_tofu_examples(tokenizer)
        req = tofu_request(
            a.author,
            examples,
            universe_authors=a.universe_authors,
            seed=a.seed,
            candidate_authors=candidate_authors,
        )
    else:
        if not candidate_authors:
            sys.exit(
                f"--dataset {a.dataset} requires an explicit frozen "
                "--candidate-authors pool"
            )
        log(
            f"loading dataset={a.dataset} request_index={a.author} "
            f"retained_groups={len(candidate_authors)} ..."
        )
        req = build_indexed_request(
            a.dataset,
            tokenizer,
            a.author,
            list(candidate_authors),
            universe_groups=a.universe_authors,
            seed=a.seed,
            pistol_sample=a.pistol_sample,
        )
    by_id = {e.example_id: e for e in req.universe.examples}
    log(f"request {req.request_id}: |Df|={len(req.forget)} |C|={len(req.universe)}")

    t2_methods = [x.strip() for x in a.t2_roster.split(",") if x.strip()]

    # A T2 run needs only the scalar pre-injection floor, not a live reference
    # model.  Compute it first and release that model before loading the SFT
    # target.  Consequently base and model0 never coexist on the GPU.
    floor_m = None
    if t2_methods:
        log("calibrating the pre-injection forgetting floor ...")
        base = load_model(a, tokenizer)
        try:
            floor_m = calibrate_floor(base, req, a.batch_size)
        finally:
            del base
            clear_cuda_cache()

    model0 = load_model(a, tokenizer)
    probe_block = mlp_down_last_layers(model0, a.block_last_n)
    selected_probe_params = probe_block.select(model0)
    model_info = {
        "architecture": type(model0).__name__,
        "model_type": getattr(model0.config, "model_type", None),
        "num_hidden_layers": int(model0.config.num_hidden_layers),
        "hidden_size": int(model0.config.hidden_size),
        "total_parameters": sum(parameter.numel() for parameter in model0.parameters()),
        "probe_block_parameters": sum(
            parameter.numel() for parameter in selected_probe_params.values()
        ),
        "tokenizer": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
    }
    trainable_state_keys = (
        tuple(selected_probe_params)
        if a.trainable_scope == "probe_block"
        else ()
    )
    del selected_probe_params
    sft_examples = list(req.forget) + list(req.universe.examples)
    cache_contract = _sft_cache_contract(
        a,
        req,
        probe_block,
        trainable_state_keys,
    )
    cache_path = resolve_sft_cache_path(
        a.sft_cache,
        automatic_root=ROOT / "runs" / "sft_cache",
        model=str(a.model),
        model_id=str(a.model_id),
        dataset=str(a.dataset),
        request_id=req.request_id,
        contract=cache_contract,
    )
    if cache_path is not None:
        log(f"SFT cache path: {cache_path}")
    sft_result = (
        _load_sft_cache(model0, cache_path, cache_contract, log)
        if cache_path is not None else None
    )
    sft_cache_hit = sft_result is not None
    if sft_result is None:
        if cache_path is not None:
            log("SFT cache miss; memorizing the request universe ...")
        else:
            log("SFT cache disabled; memorizing the request universe ...")
        sft_result = sft(model0, sft_examples, a, log,
                         probe_block if a.trainable_scope == "probe_block" else None)
    if a.require_sft_target and not sft_result["reached"]:
        raise RuntimeError(
            "SFT memorization gate failed before susceptibility scoring: "
            f"full-set mean NLL={sft_result['full_mean_nll']:.3f} > "
            f"target={sft_result['target']:.3f}"
        )
    # For probe_block campaigns, only the parameters that SFT can modify are
    # cached and retained on CPU. Fresh models provide the unchanged base state.
    state0 = _snapshot_sft_state(model0, cache_contract)
    state0_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in state0.values()
    )
    log(
        f"SFT frozen state scope={cache_contract['state_scope']} "
        f"tensors={len(state0)} size_gib={state0_bytes / 2**30:.2f}"
    )
    if cache_path is not None and not sft_cache_hit:
        for cache_attempt in range(3):
            if _write_sft_cache(
                cache_path,
                cache_contract,
                sft_result,
                state0,
                log,
            ):
                break
            del state0
            concurrent_result = _load_sft_cache(
                model0,
                cache_path,
                cache_contract,
                log,
            )
            if concurrent_result is not None:
                sft_result = concurrent_result
                sft_cache_hit = True
                state0 = _snapshot_sft_state(model0, cache_contract)
                break
            state0 = _snapshot_sft_state(model0, cache_contract)
        else:
            raise RuntimeError(
                f"SFT cache did not stabilize after 3 attempts: {cache_path}"
            )
    del model0
    clear_cuda_cache()

    def fresh():
        """Load exactly one disposable GPU model from the CPU SFT snapshot."""
        m = load_model(a, tokenizer)
        state_error = _validate_sft_state_dict(m, state0, cache_contract)
        if state_error is not None:
            raise RuntimeError(f"in-memory SFT state is invalid: {state_error}")
        _apply_sft_state(m, state0, cache_contract)
        return m

    folds = make_folds({e.example_id: e.group for e in req.universe.examples}, 0.5, a.seed)
    audit_ids = {e.example_id for e in req.universe.examples if folds[e.group] == "audit"}
    disc_ids = sorted(set(by_id) - audit_ids)
    log(f"folds: {len(audit_ids)} audit / {len(disc_ids)} discovery candidates")

    # ---- predictors, sealed on the audit fold --------------------------------
    spec = ProbeSpec(block=probe_block, eta=a.eta, seed=a.probe_seed,
                     batch_size=a.batch_size, n_dirs=a.probe_dirs, norm_eta=a.probe_norm_eta)
    # fd_norm is the frozen headline gradient probe (prereg headline_probes);
    # it is in the default roster so a no-flags run cannot silently omit it.
    predictors = list(dict.fromkeys(requested_predictors))
    unknown = sorted(set(predictors) - set(scorer_names()))
    if unknown:
        raise ValueError(f"unknown requested predictors before sealing: {unknown}")
    sentence_transformers_version = None
    if "knn_embed" in predictors:
        try:
            import sentence_transformers
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            if a.require_all_predictors:
                raise RuntimeError(
                    "requested knn_embed but sentence-transformers is unavailable; "
                    "install/provision it before sealed audit"
                ) from error
            predictors.remove("knn_embed")
            log("sentence-transformers missing: skipping requested knn_embed row")
        else:
            from rsus.probe.baselines import set_embed_encoder

            sentence_transformers_version = importlib.metadata.version("sentence-transformers")
            sentence_model = SentenceTransformer(a.sentence_encoder, device="cpu")

            def encode_sentences(examples):
                texts = [example.text for example in examples]
                if not all(texts):
                    raise ValueError("knn_embed requires non-empty Example.text")
                return torch.as_tensor(
                    sentence_model.encode(
                        texts,
                        batch_size=a.batch_size,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                )

            set_embed_encoder(encode_sentences)
    scores_by_pred = {}
    profile_artifact_dir = out / "profile_artifacts"
    profile_artifact_dir.mkdir(parents=True, exist_ok=True)
    for pred in predictors:
        log(f"scoring: {pred}")
        scorer_model = fresh()
        try:
            prof = get_scorer(pred)(scorer_model, req, spec)
        finally:
            del scorer_model
            clear_cuda_cache()
        scores_by_pred[pred] = prof.scores
        profile_payload = {
            "schema": "paper-profile-v2",
            "request": req.request_id,
            "model_id": a.model_id or Path(a.model).name,
            "scorer": pred,
            "candidate_universe_sha": req.universe.sha,
            "probe": {
                "block": probe_block.pattern,
                "eta": spec.eta,
                "norm_eta": spec.norm_eta,
                "direction_count": spec.n_dirs,
                "direction_seed": spec.seed,
                "loss": spec.loss,
            },
            "cost": vars(prof.cost),
            "candidates": [
                {
                    "candidate_id": candidate_id,
                    "group": by_id[candidate_id].group,
                    "fold": "audit" if candidate_id in audit_ids else "discovery",
                    # Audit-fold scores exist ONLY behind the seal ledger; the
                    # plain artifact keeps identity/fold metadata but no value,
                    # so nothing can consume a sealed outcome pre-gate.
                    "score": (
                        None if candidate_id in audit_ids else prof.scores[candidate_id]
                    ),
                }
                for candidate_id in sorted(prof.scores)
            ],
            "artifacts": prof.artifacts,
        }
        (profile_artifact_dir / f"{pred}.json").write_text(
            json.dumps(profile_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        seal_scores(out / "seals", out / "seal_ledger.jsonl", req.request_id, pred,
                    {c: prof.scores[c] for c in audit_ids})

    # ---- independent generator trajectories ----------------------------------
    gen_cfg = TrajectoryConfig(max_steps=a.gen_steps, checkpoint_every=a.gen_ckpt_every,
                               lr=a.gen_lr, batch_size=a.batch_size, seed=a.seed,
                               trainable_pattern=(probe_block.pattern
                                                  if a.trainable_scope == "probe_block"
                                                  else None),
                               representation_retain_mode=a.gen_rep_retain_mode,
                               **({"beta": a.beta} if a.beta else {}))
    import dataclasses as _dc
    def parse_overrides(raw, cast):
        out = {}
        for item in (x.strip() for x in raw.split(",") if x.strip()):
            if item.count("=") != 1:
                raise ValueError(f"bad override {item!r}; expected objective=value")
            key, value = item.split("=")
            out[key.strip()] = cast(value)
        return out

    gen_steps_per = parse_overrides(a.gen_steps_per, int)
    gen_lr_per = parse_overrides(a.gen_lr_per, float)
    gen_beta_per = parse_overrides(a.gen_beta_per, float)
    gen_forget_weight_per = parse_overrides(a.gen_forget_weight_per, float)
    gen_retain_weight_per = parse_overrides(a.gen_retain_weight_per, float)
    gen_rmu_alpha_per = parse_overrides(a.gen_rmu_alpha_per, float)
    gen_rmu_c_per = parse_overrides(a.gen_rmu_c_per, float)
    retain_gen = [by_id[c] for c in disc_ids]
    gens = [x.strip() for x in a.generators.split(",") if x.strip()]
    idk_gen = idk_variants(tokenizer, list(req.forget)) if "idkdpo" in gens else None
    objective_cfgs = {}
    for g in gens:
        cfg_g = _dc.replace(
            gen_cfg,
            max_steps=gen_steps_per.get(g, a.gen_steps),
            lr=gen_lr_per.get(g, a.gen_lr),
            beta=gen_beta_per.get(g, gen_cfg.beta),
            forget_weight=gen_forget_weight_per.get(g, gen_cfg.forget_weight),
            retain_weight=gen_retain_weight_per.get(g, gen_cfg.retain_weight),
            rmu_alpha=gen_rmu_alpha_per.get(g, gen_cfg.rmu_alpha),
            rmu_c=gen_rmu_c_per.get(g, gen_cfg.rmu_c),
        )
        if g == "idkdpo":
            cfg_g = _dc.replace(cfg_g, idk_examples=idk_gen)
        objective_cfgs[g] = cfg_g

    manifest = {
        "schema": "channel-matrix-run-v1",
        "host": platform.node(),
        "request": req.request_id,
        "model": a.model,
        "model_id": a.model_id or Path(a.model).name,
        "model_info": model_info,
        "dtype": a.dtype,
        "seed": a.seed,
        "probe_seed": a.probe_seed,
        "candidate_universe_sha": req.universe.sha,
        "forget_sha": req.forget_sha,
        "candidate_authors": candidate_authors,
        "trainable_scope": a.trainable_scope,
        "trainable_pattern": gen_cfg.trainable_pattern,
        "probe_config": {
            "block_last_n": a.block_last_n,
            "eta": a.eta,
            "norm_eta": a.probe_norm_eta,
            "n_dirs": a.probe_dirs,
            "seed": a.probe_seed,
            "loss": spec.loss,
        },
        "predictors": predictors,
        "sentence_encoder": (
            {"model": a.sentence_encoder, "package_version": sentence_transformers_version}
            if "knn_embed" in predictors else None
        ),
        "objectives": gens,
        "sft": sft_result,
        "sft_cache": {
            "enabled": cache_path is not None,
            "hit": sft_cache_hit,
            "path": str(cache_path) if cache_path is not None else None,
            "contract_sha256": contract_sha256(cache_contract),
            "state_scope": cache_contract["state_scope"],
            "state_tensors": len(state0),
            "state_size_bytes": state0_bytes,
        },
        "objective_configs": {
            g: {
                field.name: getattr(cfg, field.name)
                for field in _dc.fields(cfg)
                if field.name != "idk_examples"
            }
            for g, cfg in objective_cfgs.items()
        },
        "implementation_variants": {
            "npo": "summed sequence-log-probability ratio plus retain GD",
            "simnpo": "length-normalized reference-free NPO plus retain GD",
            "gru": "minimum-deviation conflicting-gradient projection",
            "repnoise": "block-controlled RepNoise-style; not full all-layer MMD implementation",
            "circuit_breakers": "block-controlled RR loss; no LoRA or coefficient schedule",
            "representation_retain_mode": a.gen_rep_retain_mode,
        },
        "cli": vars(a),
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    markers = []
    damage_by_opt: dict[str, dict[str, dict[str, float]]] = {}
    reach_by_opt: dict[str, dict[str, object]] = {}
    for g in gens:
        log(f"generator: {g}")
        cfg_g = objective_cfgs[g]
        trajectory_model = fresh()
        try:
            rec = run_trajectory(
                trajectory_model, g, req, retain_gen, cfg_g,
                out_dir=out / f"traj_{g}",
                stop_at_recall=a.gen_recall_gate,
            )
        finally:
            del trajectory_model
            clear_cuda_cache()
        markers.append(out / f"traj_{g}" / "DONE")
        term = rec.terminal()
        reached = term.forget_recall <= a.gen_recall_gate
        dmg_all = rec.damage_at()
        rem = [dmg_all[c] for c in audit_ids]
        horizon = "first-reaching" if reached else "terminal-not-reached"
        log(
            f"  {horizon} step={term.step} recall={term.forget_recall:.3f} "
            f"gate={a.gen_recall_gate:.3f} "
            f"mean_audit_dmg={sum(rem)/len(rem):+.3f}"
        )
        reach_by_opt[g] = {
            "request": req.request_id,
            "reached": reached,
            "horizon": horizon,
            "step": term.step,
            "terminal_recall": term.forget_recall,
            "recall_gate": a.gen_recall_gate,
        }
        damage_by_opt[g] = {req.request_id: {c: dmg_all[c] for c in audit_ids}}

    with open(out / "generator_reach_status.json", "w", encoding="utf-8") as f:
        json.dump(reach_by_opt, f, indent=2, sort_keys=True)

    sealed = {
        pred: {req.request_id: unseal(out / "seals", out / "seal_ledger.jsonl",
                                      req.request_id, pred, markers)}
        for pred in predictors
    }
    k = max(5, round(0.1 * len(audit_ids)))
    rows = table1_rows(sealed, damage_by_opt, k=k)
    with open(out / "table1.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    log("\n=== Gate Table 1 (single request; audit fold) ===")
    log(f"{'predictor':14s}" + "".join(f"{g+'_rho':>14s}" for g in gens) + f"{'AUROC':>8s}{'Ovl@'+str(k):>8s}")
    for pred in predictors:
        r = rows[pred]
        log(f"{pred:14s}" + "".join(f"{r[f'{g}_rho']['mean']:14.3f}" for g in gens)
            + f"{r['auroc']['mean']:8.3f}{r['overlap']['mean']:8.3f}")

    if not t2_methods:
        log(f"\nartifacts in {out}")
        return

    # ---- protection (T2 mini) -------------------------------------------------
    part_scores = scores_by_pred[a.partition_predictor]
    from rsus.costs import CostRecord
    from rsus.probe.base import ScoreProfile

    part = build_partition(
        ScoreProfile(req.request_id, a.partition_predictor, part_scores, spec, CostRecord()),
        req, folds,
        PartitionParams(pool_size=a.pool_size, min_pool_size=4, tau_rem_abs_quantile=0.6,
                        seed=a.seed),
    )
    protect = [by_id[c] for c in part.protect]
    remote_stream = [by_id[c] for c in part.remote_stream]
    gen_r = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(len(disc_ids), generator=gen_r).tolist()
    retain_matched = [by_id[disc_ids[i]] for i in perm[: len(protect)]]  # size-matched

    assert floor_m is not None
    log(f"\npartition: |P|={len(protect)} fallback={part.fallback}; floor m={floor_m:.2f}")

    # paraphrase-recall audit at every snapshot (weights are not persisted)
    extra_eval = None
    if a.dataset == "tofu":
        try:
            paras = load_tofu_paraphrases(tokenizer)
            para_ex = [paras[e.example_id] for e in req.forget if e.example_id in paras]
            if para_ex:
                extra_eval = lambda m: {"para_recall": mean_recall(m, para_ex, a.batch_size)}  # noqa: E731
        except Exception as e:
            log(f"paraphrase audit unavailable: {e}")
    else:
        log(f"paraphrase audit unavailable: no paraphrase set for dataset={a.dataset}")

    s1_cfg = Stage1Config(lr=a.s1_lr, max_steps=a.s1_max_steps, eval_every=20,
                          batch_size=a.batch_size, seed=a.seed,
                          forget_recall_max=a.s1_recall_gate or None)
    s2_cfg = Stage2Config(max_steps=a.s2_steps, refresh_k=4,
                          delta_seq_sq=a.s2_delta_seq, delta_tok_sq=a.s2_delta_tok,
                          batch_size=a.batch_size,
                          **({"eta2": a.s2_eta2} if a.s2_eta2 else {}))
    try:
        idk = idk_variants(tokenizer, list(req.forget))
    except ValueError:
        # Non-QA forget probes (e.g. RWKU cloze) have no refusal variant; the
        # idkdpo protection methods below fail loudly instead of silently.
        idk = None

    t2 = {}
    t2_lr_per = {k: float(v) for k, v in
                 (kv.split("=") for kv in a.t2_lr_per.split(",") if kv.strip())}
    for method in t2_methods:
        log(f"protection method: {method}")
        two_stage = method in ("ours", "s2s") or method.endswith("_repaired")
        m = None
        try:
            m = fresh()
            if method == "ours":
                ocfg = OursConfig(stage1=s1_cfg, stage2=s2_cfg,
                                  batch_size=a.batch_size, stage2_snapshots=4)
                rec = run_ours_trajectory(m, mlp_down_last_layers(m, a.block_last_n), req,
                                          protect, remote_stream, floor_m, ocfg,
                                          extra_eval=extra_eval, log=log)
            elif method.endswith("_repaired"):
                import dataclasses as _dc

                engine = method[: -len("_repaired")]
                eng_cfg = _dc.replace(gen_cfg, max_steps=a.t2_steps or a.gen_steps,
                                      lr=t2_lr_per.get(engine, a.t2_lr or a.gen_lr))
                if engine == "idkdpo":
                    if idk is None:
                        raise ValueError("idkdpo repair engine needs QA-formatted forget probes")
                    eng_cfg = _dc.replace(eng_cfg, idk_examples=idk)
                rcfg = RepairedConfig(engine_cfg=eng_cfg, stage2=s2_cfg,
                                      batch_size=a.batch_size)
                rec = run_engine_repaired(m, mlp_down_last_layers(m, a.block_last_n), req,
                                          retain_matched, protect, remote_stream, floor_m,
                                          engine, rcfg, extra_eval=extra_eval, log=log)
            elif method == "s2s":
                scfg = S2SConfig(stage1=s1_cfg, stage2=s2_cfg,
                                 partition=PartitionParams(pool_size=a.pool_size, min_pool_size=4,
                                                           tau_rem_abs_quantile=0.6, seed=a.seed),
                                 batch_size=a.batch_size)
                rec = run_s2s_trajectory(m, mlp_down_last_layers(m, a.block_last_n), req,
                                         folds, floor_m, scfg, extra_eval=extra_eval)
            else:
                import dataclasses as _dc

                objective = "npo" if method == "npo_transplant" else method
                retain = protect if method == "npo_transplant" else retain_matched
                cfg_m = _dc.replace(gen_cfg, max_steps=a.t2_steps or a.gen_steps,
                                    lr=t2_lr_per.get(method, a.t2_lr or a.gen_lr))
                if method == "idkdpo":
                    if idk is None:
                        raise ValueError("idkdpo t2 method needs QA-formatted forget probes")
                    cfg_m = _dc.replace(cfg_m, idk_examples=idk)
                rec = run_trajectory(m, objective, req, retain, cfg_m, extra_eval=extra_eval)
            outp = evaluate_protection(rec, native_ids=audit_ids, utility_ids=set(),
                                       recall_max=0.10,
                                       mode="last" if two_stage else "first")
        except Exception as e:  # noqa: BLE001  (one arm must not sink the table)
            import traceback
            log(f"  ERROR {type(e).__name__}: {e}")
            log(traceback.format_exc())
            t2[method] = {"reached": False, "error": f"{type(e).__name__}: {e}"}
            continue
        finally:
            if m is not None:
                del m
            clear_cuda_cache()
        t2[method] = vars(outp)
        para = (outp.extra or {}).get("para_recall")
        log(f"  reach={outp.reached} step={outp.step} "
            f"audit mean dNLL={outp.native_mean if outp.native_mean is None else round(outp.native_mean, 3)} "
            f"CVaR={outp.native_cvar if outp.native_cvar is None else round(outp.native_cvar, 3)}"
            + (f" para_recall={para:.3f}" if para is not None else ""))
    with open(out / "table2.json", "w", encoding="utf-8") as f:
        json.dump(t2, f, indent=1)
    log(f"\nartifacts in {out}")


if __name__ == "__main__":
    main()
