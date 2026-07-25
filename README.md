# fdmu — Update-Conditioned Retain Susceptibility (FD probe + guarded repair)

> **2026-07-25 protocol status:** `KDD_UnlearningFail.pdf` is the latest
> protocol source of truth. The active paper execution and evidence paths
> implement its metric-level v4 contract: first-reaching
> damage, Equations (7)--(8), exact `Kp`, fractional CVaR, and separate
> RQ1/RQ2/RQ3 IUTs including native-metric non-inferiority. See
> [`docs/PDF_V4_CODE_AUDIT.md`](docs/PDF_V4_CODE_AUDIT.md) and
> [`docs/NEW_MODEL_CALIBRATION_GUIDE.md`](docs/NEW_MODEL_CALIBRATION_GUIDE.md)
> before running or porting the method. This does not mean results are
> available: the 1.5B parent freeze still requires real `D_cal` outputs, model
> paths are unavailable on this host, and non-TOFU adapters/rosters remain
> incomplete. The editable `paper/sections/*.tex` tree is an older draft.
> The current config also disagrees with the PDF setting roster: the PDF uses
> TOFU primary/scale/family = 1.5B/7B/Llama and an 8-row Table 2, while the
> config currently encodes 7B-primary plus 1.5B-boundary and 14B in 9 rows.
> Neither source nor config may override the PDF until synchronized.

> LLM **언러닝(삭제)** 시 어떤 *보존 데이터*가 부수적으로 망가지는지를
> **미리 예측**하고(Finite-Difference 프로브), 그 예측을 이용해 **보호적 복구**를
> 수행하는 연구 코드. 논문 *"When Does LLM Unlearning Fail? Predicting and
> Protecting Susceptible Retained Behavior"*의 구현체.
>
> 프로토콜과 수식의 기준은 `KDD_UnlearningFail.pdf`다.
> `configs/paper/*.yaml`, `prereg/constants.yaml`, Markdown, LaTeX가 PDF와
> 충돌하면 PDF가 우선한다. `paper/sections/*.tex`와 `DESIGN.md`는 현재
> 구버전 초안이며, active config의 setting roster도 아직 PDF와 불일치한다.

## 문서 찾기

모든 Markdown 문서는 [문서 인덱스](docs/README.md)에서 용도별로 찾을 수 있다.
Table 1/2 수식과 정확한 PDF roster는
[metric guide](docs/TABLE12_METRICS.md)를 본다. 7월 23일
[campaign guide](docs/plan_table12_campaign.md)는 현재 PDF와 roster가 다른
역사 기록이므로 그대로 실행하면 안 된다.

---

## 1. 무엇을 하는가 (문제 정의)

삭제 요청 `q`는 잊어야 할 집합 `Df`(forget set)를 준다. 언러닝을 돌리면 `Df`는
잊히지만, **보존해야 할 후보 `C(q)`(retain universe) 중 일부가 함께 손상**된다.
손상량은 파라미터 업데이트 후의 손실 증가로 정의:

```
damage  d(x) = ℓ(x; θ_T) − ℓ(x; θ_0)      # 양수 = 손상
```

핵심 질문 세 가지:
- **RQ1 (예측):** 실제로 언러닝을 돌리기 *전에*, 어떤 `x∈C(q)`가 손상될지
  값싸게 예측할 수 있는가?
- **RQ2 (충실도/부가가치):** loss-shake가 정확 기준을 충분히 보존하며,
  hidden/simple control을 넘어서는가?
- **RQ3 (보호):** 그 예측(susceptibility profile)을 이용해 손상을 **미리 막는**
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

최신 PDF의 단일 순차 workflow는 다음과 같다. 세 개발 fold는 target과
분리되며, target damage와 sealed audit outcome은 어떠한 설정 선택에도
사용하지 않는다.

```text
사전 동결
  |
  v
D_cal -> D_pred -> D_prot
  |
  v
target profile 생성 및 audit score 봉인
  |
  v
변경하지 않은 parent unlearning 실행
  |
  v
direct-forgetting gate를 처음 통과한 checkpoint 선택
  |
  +----> RQ1: prospective damage prediction
  |
  +----> RQ2: loss-shake fidelity and added value
  |
  `----> 동일 checkpoint에서 고정 예산 repair arm 분기
             |
             `-> RQ3: constraint-matched protection
  |
  v
raw evidence -> IUT/eligibility 집계 -> Table 1 및 breadth readiness
```

### 3.1 사전 동결과 target-free 선택

- 요청, 후보 semantic group, discovery/audit/native-audit, 모델, parent,
  trainable block, seed, checkpoint grid, forgetting/utility 경계를 먼저 고정한다.
- `D_cal`은 parent hyperparameter를 선택한다. Loss-shake의 `R`, `eta`와
  engineering floor는 target 전에 runtime/evidence config에 동결한다.
- `D_pred`는 prediction weight `alpha_pred`와 strongest simple control을 고정한다.
- `D_prot`는 별도의 protection weight `alpha_prot`, `Kp`, repair 설정을 고정한다.
- prediction과 protection은 같은 signal family를 쓰지만 선택 fold와 weight가
  서로 다르다.

### 3.2 Target profile과 parent trajectory

언러닝 전 `theta0`에서 각 retain 후보 `x`에 대해 두 좌표를 계산한다.

```text
q_G(x): loss-shake susceptibility -- 이 loss가 얼마나 쉽게 움직이는가
q_H(x): forget-conditioned proximity -- 이 요청에 얼마나 노출됐는가
S_alpha(x) = (1-alpha) rank(q_G(x)) + alpha rank(q_H(x))
```

Audit rank는 parent trajectory 전에 봉인하고, discovery rank의 Top-`Kp`만
repair pool 후보로 사용한다. GradDiff, NPO, SimNPO, GRU, RMU, RepNoise,
Circuit Breakers parent를 변경하지 않고 실행한 뒤, direct-forgetting 기준을
처음 통과한 저장 checkpoint `theta_t_dagger`를 선택한다. 통과 checkpoint가
없으면 non-reaching으로 남기고 claim pass를 허용하지 않는다.

### 3.3 RQ1/RQ2/RQ3

- **RQ1 -- prospective risk:** 봉인된 joint rank와 이후 audit damage의
  Spearman rho, `q_G`/`q_H` 단독 대비 gain, positive-damage tail lift와
  tail coverage를 평가한다.
- **RQ2 -- fidelity and added value:** loss-shake를 동일 block의 exact
  candidate gradient energy와 비교한다. Fidelity floor는 Spearman 0.80,
  Top-`Kp` overlap 0.70이며, proximity와 frozen simple control을 넘어서는
  prediction gain도 별도로 요구한다.
- **RQ3 -- decision value:** 동일한 `theta_t_dagger`에서 joint, no-repair,
  repeated-random, `q_G`, `q_H` arm을 분기한다. 모든 active arm은 repair
  operator, neutral stream, example order, seed, token budget, projection,
  guard를 공유하고 selector만 다르다. Direct/paraphrase/extraction forgetting과
  utility 조건을 모두 만족한 공통 support에서 mean damage, fractional
  CVaR.95, dataset-native retain metric을 비교한다.

Table 1/2 각 열의 수식, 방향, bootstrap/IUT 판정과 Table 2 분모 정의는
[Table 1/2 metric guide](docs/TABLE12_METRICS.md)에 정리돼 있다.

### 3.4 실행과 산출물

논문 stage 오케스트레이터는 `experiments/paper/run_v4_stage.py`이며 정확한
`D_cal/D_pred/D_prot/target` roster와 unit command를 소비하고 원자료를
검증·봉인한다. TOFU unit producer는 `experiments/paper/tofu_v4_unit.py`,
전체 진입점은 다음과 같다.

```bash
python experiments/paper/run_tofu_table1.py --action plan
python experiments/paper/run_tofu_table1.py --action run
```

`plan`은 네 stage의 exact manifest를 만들고, `run`은 freeze 순서를 강제한 뒤
raw evidence를 집계한다. 최종 Table 1은 Panel A(RQ1/RQ2)와 Panel B(RQ3)로
`paper/sections/generated/table1.tex`에 생성된다. Breadth 판정 자료는
`evidence_readiness.json`에 기록된다. `experiments/gate_1p5b/gate.py`는 이
claim workflow의 라이브러리가 아니라 이전 단일 요청 진단 CLI다. 다만 현재
TOFU unit producer가 모델 로딩/SFT helper를 재사용하므로 파일 의존성은 남아 있다.

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
│   │   ├── protection.py         이전 gate 보호 요약 (reach·mean·CVaR·utility)
│   │   └── metrics.py            teacher-forced recall / retention
│   │
│   ├── analysis/
│   │   ├── prediction.py         예측 metric (Spearman·AUROC·Overlap@K)
│   │   ├── table1_selection.py   target-free alpha/control/Kp 선택 계약
│   │   ├── ablation.py           Table 3 matched-parent 대조
│   │   └── channels.py · mixture.py · stats.py
│   │
│   ├── data/                    ── 데이터셋 어댑터 (fail-closed 레지스트리)
│   │   ├── base.py               Request · CandidateUniverse · Example
│   │   ├── registry.py           등록: tofu · rwku · muse_news/books · substrate
│   │   ├── tofu.py       ✅       TOFU forget10 (저자별 삭제)
│   │   ├── rwku.py       ✅       RWKU 실세계 지식 삭제 (타겟별)
│   │   ├── substrate.py  ✅       합성 substrate (ground-truth adjacency)
│   │   ├── muse.py       ✅       MUSE-News/Books knowmem (forget_qa→Df, retain_qa→C(q)) — 이 포크 추가
│   │   └── wmdp.py ✅ / pistol ❌
│   │
│   └── evidence/                 v4 raw/schema/decision + Table 1 렌더링
│
├── experiments/
│   ├── gate_1p5b/                gate.py — 이전 단일 요청 진단 + 공용 helper(분리 예정)
│   ├── channel_matrix/           채널 상호작용 캠페인 (objective/alpha freeze)
│   ├── cost/bench.py             Table 4 프로파일링 비용
│   ├── stability/                수치 안정성 스윕 (η·precision·block)
│   ├── diag/                     프로브 일치·채널 리포트·피규어
│   ├── paper/
│   │   ├── run_tofu_table1.py    D_cal→D_pred→D_prot→target 전체 진입점
│   │   ├── init_v4_stage.py      exact stage manifest 생성
│   │   ├── run_v4_stage.py       unit 실행·검증·봉인
│   │   ├── select_tofu_v4.py     parent/claim freeze 생성
│   │   └── tofu_v4_unit.py       TOFU model-output unit producer
│   └── pilot/ · cluster/          파일럿 · 멀티노드 큐
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
├── tests/                       CPU 단위·계약·evidence 파이프라인 테스트
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
| **PDF v4 Eq. (7)--(8) repair + first-reaching wrapper** | ✅ **구현·단위테스트** (`repair.py`, `generators/repaired.py`) |
| **RQ1/RQ2/RQ3 raw aggregation + 4/4/12-way IUT** | ✅ schema v2, fail-closed |
| **Paper-stage 오케스트레이터** | ✅ config roster 실행·검증·봉인 구현; 최신 PDF roster와 config는 불일치 |
| **TOFU PDF-v4 model-output unit producer** | ✅ 구현·계약테스트; 실제 GPU campaign 미실행 |
| **TOFU Table 1 Panel A/B 선택·집계·LaTeX 렌더링** | ✅ 구현·합성 140-unit E2E 검증 |
| PDF Table 2 breadth/failure-boundary LaTeX 렌더링 | ⚠️ metric 렌더링 구현; PDF는 8행인데 현재 config는 9행 |
| 구버전 Table 3 ablation / Table 4 cost 도구 | ✅ 구현; 최신 PDF appendix와 재매핑 필요 |
| 데이터셋 **TOFU, RWKU, substrate** | ✅ 어댑터 구현 |
| 데이터셋 **MUSE (News/Books)** | ✅ corpus-level 어댑터와 registry 구현; 독립 target-request roster는 미지원이라 paper preflight 차단 |
| 데이터셋 **WMDP-bio/MMLU** | ✅ 어댑터·7B/14B H100 campaign·테스트 구현; PDF-v4 exact roster/producer는 미동결 |
| 데이터셋 **PISTOL** | ❌ 어댑터·exact roster 미구현 |
| **KnowUnDo / OpenUnlearning** | related work 인용이며 현재 실험 roster 아님 |

> 요약: **v4 metric, repair, evidence 집계와 renderer는 구현됨**. 다만 최신
> PDF의 1.5B-primary/7B-scale/Llama-family 및 8행 Table 2 roster가 active
> config에 반영되지 않았다. 실제 target 수치도 아직 없으므로 현재 상태는
> claim-bearing evidence가 아니다.

## 6. 설치 & 실행

```bash
# 환경 (torch cu12x, RTX 4090 검증: 2.7.1+cu126)
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install "torch==2.7.1" transformers datasets pyyaml pytest sentence-transformers accelerate
uv pip install -e .

python -m pytest                 # CPU 단위 배터리

# 이전 단일 요청 진단 CPU smoke
python experiments/gate_1p5b/gate.py --smoke --model <로컬 모델 경로>

# 이전 단일 요청 진단 GPU run
python experiments/gate_1p5b/gate.py --model <경로> --device cuda --dtype float32 \
  --dataset tofu --universe-authors 20 --trainable-scope probe_block

# 최신 PDF-v4 TOFU exact manifest 점검
python experiments/paper/run_tofu_table1.py \
  --action plan --setting tofu_qwen25_1p5b
```

### H100 Table 1/2 캠페인

현재 `enqueue_table12.sh` wave는 7B-primary/14B가 포함된 이전 9행 config용이라
최신 PDF paper evidence로 enqueue하면 안 된다. 먼저
`configs/paper/campaign.yaml`과 `configs/paper/evidence.yaml`을 PDF의
1.5B-primary/7B-scale/Llama-family 및 8행 Table 2에 맞춰야 한다. 그 전에는
아래 read-only 검사만 허용한다.

```bash
python experiments/paper/preflight.py
python experiments/cluster/next_actions.py
bash experiments/cluster/enqueue_table12.sh status
```

TOFU PDF-v4 target unit이 완료되면 sealed raw 세 종류를 하나의 ledger로
집계하고, 그 ledger에서 두 main table과 readiness를 함께 생성한다.

```bash
python experiments/paper/aggregate_raw.py \
  --plan runs/paper/tofu_table1/tofu_qwen25_1p5b/raw_plan.json \
  --prediction-raw runs/paper/tofu_table1/tofu_qwen25_1p5b/target_evaluation/sealed/prediction_raw.jsonl \
  --fidelity-raw runs/paper/tofu_table1/tofu_qwen25_1p5b/target_evaluation/sealed/fidelity_raw.jsonl \
  --protection-raw runs/paper/tofu_table1/tofu_qwen25_1p5b/target_evaluation/sealed/protection_raw.jsonl \
  --out results/paper/evidence_ledger.json

python experiments/paper/build_evidence.py \
  --ledger results/paper/evidence_ledger.json \
  --paper-root paper
```

기존 channel-matrix audit 산출물은
`experiments/paper/export_channel_matrix_raw.py`로 v4 이름의
prediction/protection shard로 변환할 수 있다. 다만 setting별 RQ2 판정에는
독립적인 per-unit `fidelity_raw.jsonl`이 필요하며, certificate summary만으로
RQ2 pass를 만들지 않는다. 정확한 PDF 계약은
[Table 1/2 metric guide](docs/TABLE12_METRICS.md)를 본다. 이전
[7월 23일 캠페인 문서](docs/plan_table12_campaign.md)는 roster가 달라
실행 지침으로 사용하지 않는다.

`gate.py`는 기본적으로 exact SFT contract별 checkpoint를
`runs/sft_cache/`에 저장하고 다음 실행에서 자동으로 불러온다. 명시적 파일은
`--sft-cache /path/theta0.pt`, 캐시 비활성화는 `--sft-cache off`를 사용한다.
PDF-v4 TOFU workflow는 `runs/paper/tofu_v4/sft_cache/`의 request/seed별
checkpoint를 모든 parent가 공유하며, 각 unit manifest의 `sft_cache.hit`로
재사용 여부를 감사할 수 있다. Forget 요청이나 seed, candidate universe,
모델 또는 SFT 설정이 다르면 같은 초기 상태가 아니므로 별도 cache를 사용한다.

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
- **RQ3 repair는 구현돼 있으나 24GB에서 실행 불가**: Stage1이 full-model fp32
  AdamW(1.5B도 ~25GB) → OOM. **H100급(80GB) 필요.** (RQ1/RQ2 프로파일링은
  4090에서 실행 가능.)

## 8. 재현 산출물

최신 TOFU workflow는 `runs/paper/tofu_table1/` 아래에 stage manifest,
unit별 `run_manifest.json`, sealed JSONL, `raw_plan.json`,
`evidence_ledger.json`, `evidence_readiness.json`을 남긴다. ledger 상태와
무관하게 main Table 1/2는 생성되며, 미완료 셀은 `\tblph`로 남는다. 완전한
ledger에서만 `--require-ready`가 성공한다. 이전 `gate.py`
캠페인의 `table1.json`, `table2.json`, `gate.log`, `seal_ledger.jsonl`과는
별도 산출물이다.

## 9. 미구현 / 앞으로 필요한 것 (Roadmap)

논문·설계에는 있으나 **아직 코드에 없거나, 있어도 현 환경에서 못 돌리는 것**과
그것을 채우기 위해 필요한 작업.

### 9.1 데이터셋 확장
- **WMDP-bio/MMLU** — adapter와 H100 channel-matrix campaign은 구현됨.
  claim-bearing PDF-v4 exact roster와 dataset unit producer는 아직 동결되지 않음.
- **PISTOL** — 최신 PDF 실험 roster지만 `data/`와 registry에 어댑터 없음.
- **KnowUnDo, OpenUnlearning** — related work 인용이며 현재 실험 구현 대상이 아님.
- **MUSE(News/Books)** — corpus-level 로더는 등록됐지만, 현재 knowmem 구조는
  독립적인 `D_cal/D_pred/D_prot/target` 삭제 요청 roster를 제공하지 않는다.
- 필요 작업: 데이터셋별 `src/rsus/data/<name>.py`(forget→`Df`, retain→`C(q)` 매핑 +
  fold/native-audit) → `registry.py` 등록 → exact stage roster와 dataset unit
  producer 추가.
  - MUSE는 요청 단위를 문서/토픽 수준으로 정의하고 결과와 무관하게 네 roster를
    동결하기 전까지 corpus 하나를 여러 요청으로 가장하면 안 된다.
  - WMDP는 MMLU와 함께 MCQA 언러닝 세팅, PISTOL은 구조적 삭제 — 요청 의미 설계 필요.

### 9.2 repair(Stage2)를 24GB에서 실행 (구현은 됨, 메모리 이슈)
- 현재 Stage1이 `AdamW(model.parameters())` full-model fp32라 1.5B도 ~25GB → 4090 OOM.
  (H100 80GB 전제.) RQ1/RQ2 프로파일링은 4090에서 가능하지만 **RQ3 repair는
  실행하지 못함**.
- 필요 작업(택1 이상): Stage1 옵티마이저를 블록 스코프로 제한 / 8-bit·paged AdamW
  (bitsandbytes) / gradient checkpointing / FSDP·ZeRO 멀티-GPU 샤딩 / bf16 마스터-가중치.
  단, 프로브의 fd는 fp32 필요 → 정밀도-메모리 트레이드오프 설계.

### 9.3 실험 커버리지 (구현됐으나 아직 안 돌린 것)
- **최신 Table 1:** 코드와 합성 evidence 검증만 완료했으며 실제 TOFU target
  campaign은 미실행이다.
- **최신 Table 2:** PDF는 8행이지만 current config/renderer denominator는
  9행이다. PDF roster 동기화 후 비-TOFU exact producer를 채워야 한다.
- **Appendix ablation/cost:** 기존 `analysis/ablation.py`와
  `experiments/cost/bench.py`를 최신 PDF appendix contract에 맞춰 재매핑해야 한다.
- **통계:** hierarchical bootstrap과 IUT는 구현됐지만 실제 다중 request/seed
  evidence로 실행하지 않았다.

### 9.4 정리·인프라
- `local_run/`은 이 포크 전용 도구 — 상류(retain-susceptibility)엔 없음.
- `prereg/constants.yaml` δ_seq/δ_tok·slack ε 기본값 동결(D4), MIA(Min-K%++) 소스(D5)는
  논문상 open decision.
