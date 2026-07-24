# fdmu — Update-Conditioned Retain Susceptibility (FD probe + guarded repair)

> **2026-07-24 protocol warning:** `KDD_UnlearningFail.pdf` contains a newer
> Chapter 4 than this README, the checked-in LaTeX, and parts of the code. The
> repository is not yet claim-compatible with that PDF, especially for
> Equations (7)--(8), RQ1--RQ3 decisions, fractional CVaR, and native-metric
> non-inferiority. See
> [`docs/PDF_V4_CODE_AUDIT.md`](docs/PDF_V4_CODE_AUDIT.md) and
> [`docs/NEW_MODEL_CALIBRATION_GUIDE.md`](docs/NEW_MODEL_CALIBRATION_GUIDE.md)
> before running or porting the method. The sections below describe the legacy
> implementation and must not be treated as the July 24 paper contract.
> The staged v4 mechanics live in `src/rsus/repair.py` and the explicitly named
> `run_pdf_repair_from_reached` wrapper; they fail closed when `B_tok` or other
> required repair settings are not frozen.

> LLM **언러닝(삭제)** 시 어떤 *보존 데이터*가 부수적으로 망가지는지를
> **미리 예측**하고(Finite-Difference 프로브), 그 예측을 이용해 **보호적 복구**를
> 수행하는 연구 코드. 논문 *"When Does LLM Unlearning Fail? Probing
> Update-Conditioned Retain Susceptibility"* (`paper/` 참고)의 구현체.
>
> 이 README는 **실제 코드 기준**으로 작성됨. 개념/수식은 `paper/sections/*.tex`,
> 동결 상수는 `prereg/constants.yaml`이 최종 근거. (`DESIGN.md`는 아카이브된
> 초안이라 일부 파일명이 실제 트리와 다름 — 참고용으로만.)

---

## 1. 무엇을 하는가 (문제 정의)

삭제 요청 `q`는 잊어야 할 집합 `Df`(forget set)를 준다. 언러닝을 돌리면 `Df`는
잊히지만, **보존해야 할 후보 `C(q)`(retain universe) 중 일부가 함께 손상**된다.
손상량은 파라미터 업데이트 후의 손실 증가로 정의:

```
damage  d(x) = ℓ(x; θ_T) − ℓ(x; θ_0)      # 양수 = 손상
```

핵심 질문 두 가지:
- **RQ1 (예측):** 실제로 언러닝을 돌리기 *전에*, 어떤 `x∈C(q)`가 손상될지
  값싸게 예측할 수 있는가?
- **RQ2 (보호):** 그 예측(susceptibility profile)을 이용해 손상을 **미리 막는**
  복구(repair)를 할 수 있는가?

## 2. 핵심 아이디어 — FD(finite-difference) susceptibility probe

`Df`에 대한 1회 backward로 forget 그래디언트 방향 `ĝ`를 얻고, 파라미터를
`θ ± η·ĝ`로 **미세하게 흔들어** 각 후보 `x`의 손실이 얼마나 "흔들리는지"를
유한차분으로 측정한다 (paper `eq:loss-shake-identity`). 이 loss-shake가 클수록
그 후보는 forget 업데이트에 취약(susceptible)하다.

- `fd` — 정렬(alignment) 프로브: 1 backward(Df) + `θ±η·ĝ`에서 2회 batched forward.
- `fd_norm` — K개 랜덤방향 제곱평균 = grad-norm 추정(**backward-free**), bf16/대형에
  강건 (headline gradient probe, `prereg` 동결).
- 비교군: `jvp`(정확 forward-mode), `vmap_graddot`, `streaming_backward`.
- 베이스라인: `grad_norm`, `grad_cosine`, `knn_feature/embed/lexical`,
  `last_layer`, `random_dir/rank`.

## 3. 어떤 실험인가 (파이프라인)

메인 실행 드라이버: **`experiments/gate_1p5b/gate.py`**. 단일 삭제 요청에 대해
아래를 순서대로 수행하고 `runs/<...>/table1.json`, `table2.json`, 봉인 원장을 남긴다.

```
1. floor 보정      : 사전주입 forgetting floor m 계산 (reference 모델)
2. SFT 주입        : 요청 universe를 모델에 암기시킴 (θ0 정의)
3. 채점(probe)     : C(q)의 각 후보를 fd/fd_norm/knn_*/grad_norm 등으로 점수화
                     → discovery/audit fold 분리, audit fold는 seal(봉인)
4. 제너레이터      : npo/graddiff/rmu 등 언러닝을 실제로 돌려 checkpoint별
                     realized damage d(x) 기록  ← RQ1 ground truth
5. Table 1         : 프로브 점수 vs 실제 손상의 순위상관(Spearman ρ),
                     AUROC(top-K 손상 멤버십), Overlap@K   (analysis/prediction.py)
6. 파티션 + 복구   : profile로 보호풀 P 구성 → Stage1(제약 망각) + Stage2(guarded
                     repair) = "ours" 및 baseline 보호법 실행
7. Table 2         : reach, 손상 mean/CVaR.95, paraphrase recall  (evalx/protection.py)
```

**Table 1 성공 신호:** `fd` 행의 ρ/AUROC가 knn_*·grad_norm보다 명확히 높음.
**Table 2 성공 신호:** `ours`/`npo_transplant`가 plain `npo`보다 audit 손상 낮음.

## 4. 코드 구조 (실제 트리)

```
fdmu/  (retain-susceptibility 포크)
│
├── src/rsus/                     ── 핵심 라이브러리 (패키지명 rsus)
│   ├── blocks.py                 프로브/학습 대상 블록 B 선택 (late-layer MLP down_proj)
│   ├── losses.py                 prompt-masked per-seq / per-token NLL
│   ├── costs.py                  CostRecord + Meter (wall·peak-mem·fwd/bwd) — Table 4 텔레메트리
│   ├── partition.py              folds(discovery/audit) + 보호풀 P + remote stream
│   ├── refcache.py               Stage1 exit-gate 캐시 (seq/token ref + index map)
│   ├── sealing.py                audit-fold 점수 봉인 (append-only ledger)
│   ├── stage1.py         ★repair Stage1 — 제약 망각 → floor m (aug-Lagrangian)
│   ├── guards.py         ★repair one_sided / symmetric / sorted 가드 페널티
│   ├── stage2.py         ★repair Stage2 — projected guarded repair (eq:wpgd)
│   ├── config.py · runlog.py     YAML 해시 provenance · jsonl 로깅
│   │
│   ├── probe/                    ── susceptibility 스코어러
│   │   ├── base.py               ProbeSpec · ScoreProfile · 레지스트리
│   │   ├── finite_diff.py  ★     fd · fd_norm · fd_constrained  (loss-shake)
│   │   ├── jvp.py · graddot.py    정확 forward-mode · vmap/streaming grad-dot 비교기
│   │   ├── baselines.py           knn_{feature,embed,lexical} · grad_norm · random_*
│   │   └── fidelity.py            프로브 정합성 인증
│   │
│   ├── generators/              ── 언러닝 목적함수 (RQ1 정답 + RQ2 baseline)
│   │   ├── objectives.py         ga·graddiff·npo·simnpo·idkdpo·rmu·gru·repnoise·cb
│   │   ├── base.py               run_trajectory (checkpoint별 damage 기록)
│   │   ├── ours.py       ★repair run_ours_trajectory = Stage1 + Stage2 (제안법)
│   │   ├── repaired.py   ★repair 임의 엔진 언러닝 후 Stage2 복구
│   │   └── s2s.py                 Cheng split-aware 2-stage baseline
│   │
│   ├── evalx/
│   │   ├── protection.py         Table 2 결과 (reach·mean·CVaR·utility)
│   │   └── metrics.py            teacher-forced recall / retention
│   │
│   ├── analysis/
│   │   ├── prediction.py         Table 1 (Spearman·AUROC·Overlap@K + bootstrap)
│   │   ├── ablation.py           Table 3 matched-parent 대조
│   │   └── channels.py · mixture.py · stats.py
│   │
│   ├── data/                    ── 데이터셋 어댑터 (fail-closed 레지스트리)
│   │   ├── base.py               Request · CandidateUniverse · Example
│   │   ├── registry.py           등록: tofu · rwku · substrate
│   │   ├── tofu.py       ✅       TOFU forget10 (저자별 삭제)
│   │   ├── rwku.py       ✅       RWKU 실세계 지식 삭제 (타겟별)
│   │   ├── substrate.py  ✅       합성 substrate (ground-truth adjacency)
│   │   ├── muse.py       ✅       MUSE-News/Books knowmem (forget_qa→Df, retain_qa→C(q)) — 이 포크 추가
│   │   └── (wmdp / pistol ❌ 미구현 — 논문엔 있음)
│   │
│   └── evidence/                 raw · registry · schemas · statistics · decisions · rendering
│
├── experiments/
│   ├── gate_1p5b/                ★ gate.py — 메인 게이트 실험 + RUNBOOK
│   ├── channel_matrix/           채널 상호작용 캠페인 (objective/alpha freeze)
│   ├── cost/bench.py             Table 4 프로파일링 비용
│   ├── stability/                수치 안정성 스윕 (η·precision·block)
│   ├── diag/                     프로브 일치·채널 리포트·피규어
│   └── pilot/ · cluster/ · paper/  파일럿 · 멀티노드 큐 · 논문 산출
│
├── local_run/                   ◀ 이 포크 추가: RTX 4090 로컬 캠페인 도구
│   ├── run_one.sh                모델×데이터셋 1건 실행 + 요약
│   ├── run_queue.sh              여러 건 순차 (레인)
│   ├── summarize.py              table1/2.json → 마크다운 (1등 볼드+밑줄)
│   ├── download_{models,data,muse}.py/sh
│   └── README.md
│
├── paper/                       LaTeX 논문 (개념·수식 근거)
│   ├── main.tex
│   ├── sections/   00_abstract … 08_conclusion · 99_appendix
│   └── figures/    fig1_finite_difference · alg_repair · …
│
├── configs/                     channel_matrix/ · cluster/ · paper/  (실험 YAML)
├── docs/                        RUNBOOK · 계획 · data/ 산출 CSV·figure
├── prereg/                      constants.yaml (동결 상수) + seal_ledger.jsonl
├── tests/                       CPU 단위 배터리 24개 (195 passed / 2 skipped)
│
├── README.md                    ◀ 이 포크 추가 (프로젝트 상세 설명)
├── CLAUDE.md · DESIGN.md         클러스터 노트 · 아카이브 설계노트
└── pyproject.toml               패키지 rsus (torch · transformers · datasets)
```

**범례:** `★repair` = 보호/복구 코어(구현·테스트 완료) · `✅` 구현된 어댑터 ·
`❌` 미구현 · `◀` 이 포크에서 추가.

## 5. 구현 현황 (논문 대비) — 실제 코드 기준

| 구성요소 | 상태 |
|---|---|
| FD 프로브(fd/fd_norm/fd_constrained) + jvp/vmap/streaming 비교기 | ✅ 구현·테스트 |
| 베이스라인 스코어러(knn_*, grad_norm/cosine, last_layer, random_*) | ✅ |
| 언러닝 objectives (ga/graddiff/npo/simnpo/idkdpo/rmu/gru/circuit_breakers/repnoise) | ✅ |
| **repair (Stage1 + Stage2 guarded, "ours") / repaired / s2s** | ✅ **구현·단위테스트** (test_stages/guards/alpha) |
| Table 1 예측 분석 / Table 2 보호 / Table 3 ablation / Table 4 cost | ✅ |
| 데이터셋 **TOFU, RWKU, substrate** | ✅ 어댑터 구현 |
| 데이터셋 **MUSE (News/Books)** | ✅ **어댑터 구현 (이 포크)** — `data/muse.py` + `gate.py --dataset muse_news\|muse_books` + `test_muse.py`. (paper 캠페인 registry graduation은 roster_unit TBD라 미포함) |
| 데이터셋 **WMDP / PISTOL / KnowUnDo** | ❌ **미구현** (논문엔 있음) |

> 요약: **프로브·repair·분석·TOFU/RWKU까지는 구현 완료**. 데이터셋 확장(MUSE 등)은
> 어댑터(`src/rsus/data/<name>.py` + registry 등록 + gate.py 연동)를 추가해야 함.

## 6. 설치 & 실행

```bash
# 환경 (torch cu12x, RTX 4090 검증: 2.7.1+cu126)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install "torch==2.7.1" transformers datasets pyyaml pytest sentence-transformers accelerate
uv pip install -e .

python -m pytest                 # CPU 단위 배터리 (195 passed)

# CPU smoke (tiny 랜덤 모델로 파이프라인 검증)
python experiments/gate_1p5b/gate.py --smoke --model <로컬 모델 경로>

# 실제 게이트 (예: TOFU, 단일 GPU)
python experiments/gate_1p5b/gate.py --model <경로> --device cuda --dtype float32 \
  --dataset tofu --universe-authors 20 --trainable-scope probe_block
```

로컬 멀티모델/데이터셋 캠페인은 **`local_run/`** 참고 (경로 규약: 모델
`/rdata/models`, 데이터 `/rdata/minsoo3.kim/hf_home`, 결과
`/rdata/minsoo3.kim/results/<dataset>/`):

```bash
GPU=0 bash local_run/run_one.sh 3b Qwen2.5-3B-Instruct float32                 # TOFU
DATASET=rwku GPU=0 bash local_run/run_one.sh 3b Qwen2.5-3B-Instruct float32    # RWKU
GPU=0,1 DMAP=split:8 bash local_run/run_one.sh 7b_fp32 Qwen2.5-7B-Instruct float32  # 7B fp32 2-GPU
```

## 7. 하드웨어 노트 (RTX 4090 24GB)

- `--trainable-scope probe_block`가 1-GPU 모드. ≤4B는 fp32 단일 GPU, 7B fp32는
  `--device-map split:8`(마지막 8레이어를 cuda:1)로 2-GPU 분산.
- **bf16은 fd를 무너뜨림**(eta=3e-4 유한차분이 정밀도 바닥에 삼켜짐) → 프로브는 fp32.
- **Table 2 repair는 구현돼 있으나 24GB에서 실행 불가**: Stage1이 full-model fp32
  AdamW(1.5B도 ~25GB) → OOM. **H100급(80GB) 필요.** (프로브 Table 1은 4090에서 정상.)

## 8. 재현 산출물

`runs/<...>/`: `table1.json`, `table2.json`, `gate.log`, `run_manifest.json`,
`seal_ledger.jsonl`(순서 증명). 로컬 캠페인 요약은
`/rdata/minsoo3.kim/results/<dataset>/CAMPAIGN_REPORT.md`.

## 9. 미구현 / 앞으로 필요한 것 (Roadmap)

논문·설계에는 있으나 **아직 코드에 없거나, 있어도 현 환경에서 못 돌리는 것**과
그것을 채우기 위해 필요한 작업.

### 9.1 데이터셋 확장 (어댑터 미구현)
- **MUSE(News/Books), WMDP, PISTOL, KnowUnDo** — 논문엔 등장하나 `data/`에 어댑터
  없음(registry 미등록). `gate.py --dataset muse` 등 불가.
- 필요 작업: 데이터셋별 `src/rsus/data/<name>.py`(forget→`Df`, retain→`C(q)` 매핑 +
  fold/native-audit) → `registry.py`에 `register_adapter` → `gate.py --dataset` choices 추가.
  - MUSE는 `knowmem`이 QA 쌍(forget_qa/retain_qa)이라 TOFU 방식으로 매핑 쉬움
    (데이터는 이미 `/rdata/minsoo3.kim/hf_home`에 다운로드됨). "삭제 요청 1건"의
    단위(문서/토픽)만 동결하면 됨 — `campaign.yaml`의 `roster_unit`이 아직 TBD.
  - WMDP는 MMLU와 함께 MCQA 언러닝 세팅, PISTOL은 구조적 삭제 — 요청 의미 설계 필요.

### 9.2 repair(Stage2)를 24GB에서 실행 (구현은 됨, 메모리 이슈)
- 현재 Stage1이 `AdamW(model.parameters())` full-model fp32라 1.5B도 ~25GB → 4090 OOM.
  (H100 80GB 전제.) Table 1(프로브)은 4090 정상, **Table 2만 못 돌림**.
- 필요 작업(택1 이상): Stage1 옵티마이저를 블록 스코프로 제한 / 8-bit·paged AdamW
  (bitsandbytes) / gradient checkpointing / FSDP·ZeRO 멀티-GPU 샤딩 / bf16 마스터-가중치.
  단, 프로브의 fd는 fp32 필요 → 정밀도-메모리 트레이드오프 설계.

### 9.3 실험 커버리지 (구현됐으나 아직 안 돌린 것)
- **Table 2 reach**: single-stage(npo/npo_transplant)가 기본 gen 예산에서 forget
  게이트(0.10) 미도달 → 목적함수별 gen-lr/steps 튜닝(RUNBOOK 노브) 필요.
- **Table 3(ablation)**: `analysis/ablation.py` 있으나 로컬 캠페인 미실행 —
  partition source swap / projection·guard off 매트릭스 구성 필요.
- **Table 4(cost)**: `experiments/cost/bench.py` 있으나 fd/jvp/vmap/streaming 4종
  비용 측정 미실행 (7B/14B 타겟).
- **통계**: `analysis`가 seed 평균·hierarchical bootstrap·LOTO 지원하나 현 캠페인은
  단일 seed(2025). 다중 seed로 Table 1 ρ 노이즈 축소 필요.

### 9.4 정리·인프라
- `local_run/`은 이 포크 전용 도구 — 상류(retain-susceptibility)엔 없음.
- `prereg/constants.yaml` δ_seq/δ_tok·slack ε 기본값 동결(D4), MIA(Min-K%++) 소스(D5)는
  논문상 open decision.
