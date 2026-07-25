# Local RTX 4090 x2: TOFU 1.5B Table 2 agent guide

이 문서는 로컬 coding agent(Claude, Gemini, Codex 등)가 수행할 구현 및 실행
작업의 기준 문서다. 특정 agent 제품의 기능을 가정하지 않는다. 계획만
제안하고 끝내지 말고, 환경 설치, 코드 구현, 테스트, GPU 실행, evidence
집계와 LaTeX 생성까지 가능한 범위에서 계속 진행한다.

## 구현된 joint 개발 스윕

1.5B parent calibration, 검토 가능한 freeze proposal, 명시적 승인, 그리고
`D_prot` joint sweep까지 구현되어 있다. YAML을 직접 편집하지 않는다.

```bash
# 28개 calibration unit 실행 후 proposal 생성
GPU_IDS=0,1 bash local_run/run_tofu_1p5b_calibration.sh

# proposal의 unresolved가 0이고 수치/hash 검토가 끝난 뒤 명시적으로 승인
bash local_run/approve_tofu_1p5b_parent_freeze.sh --approve

# 24개 이하의 target-free joint sweep
GPU_IDS=0,1 bash local_run/sweep_joint_1p5b_4090x2.sh
```

설정은 `configs/local/joint_sweep_1p5b_4090x2.yaml`, controller는
`experiments/paper/run_joint_dev_sweep.py`다. joint가 feasible이고 모든
개발 셀에서 joint가 feasible이어야 한다. Comparator가 infeasible이면
제약 우선 순위로 joint가 이기고, comparator도 feasible일 때는 joint의
mean/CVaR damage가 모두 작아야 한다. 성공해도 target을 실행하지 않고 human
freeze용 draft recommendation만 만든다.

현재 탐색 예산은 target-free trial 24개다. 성공 조건을 먼저 만족하면 exit
0으로 즉시 종료한다. 24개를 모두 실행해도 조건을 만족하지 못하면
`NO_JOINT_DOMINANCE`, exit 3으로 실패 셀과 가장 가까운 후보를 로그에 남기고
종료한다.

실행 중에는 unit stdout/stderr가 `[GPU<id> <unit>]` 접두사로 콘솔에 바로
출력되고 15초마다 완료 수, 실행 중 unit, 실패 수와 경과 시간이 표시된다.
전체 launcher 출력은 다음 경로에서 동시에 확인한다.

```bash
tail -f /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/joint_sweep/launcher_logs/current.log
tail -f /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/joint_sweep/events.jsonl
```

주기를 바꾸려면 `PROGRESS_INTERVAL_SECONDS=5`처럼 지정한다. 개별 unit 원본
로그는 `trials/<trial>/logs/units/<unit>/attempt-*.log`, stage 검증 로그는
`trials/<trial>/logs/verify.log`에 보존된다.

Calibration 현황은 다음 경로에서 확인한다.

```bash
tail -f /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/parent_calibration/launcher_logs/current.log
cat /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/parent_calibration/CALIBRATION_STATUS.json
```

Calibration 결과에 unresolved parent가 하나라도 있으면 승인 명령은 실패한다.
그 경우 YAML을 손으로 채우지 않고 `PARENT_CALIBRATION_UNRESOLVED`를 유효한
개발 결과로 보고한다.

## 0. Agent 실행 원칙

1. 먼저 `AGENTS.md`, 이 문서, `configs/paper/tofu_v4.yaml`을 읽는다.
2. 기존 PDF-v4 producer와 metric을 재사용하고 legacy 결과를 섞지 않는다.
3. GPU가 비어 있고 freeze gate가 통과된 stage는 사용자에게 명령을 떠넘기지
   말고 직접 실행한다.
4. 장시간 실행은 background에 던지고 끝내지 말고 PID, log, progress를
   추적한다. 필요한 worker가 살아 있는 동안 작업 완료라고 보고하지 않는다.
5. human freeze가 필요한 지점만 멈춘다. proposal과 검토할 수치, 다음에 다시
   실행할 동일 명령을 출력한다.
6. target 결과를 본 뒤 설정을 바꾸지 않는다.
7. 기존 run, cache, freeze, seal을 삭제하거나 우회하지 않는다.

## 1. 목표

로컬 워크스테이션의 RTX 4090 24GB 두 장에서
`tofu_qwen25_1p5b` PDF-v4 실험을 실행하고, 최종적으로 다음 파일을 만드는
재실행 가능한 단일 진입점을 구현한다.

```text
paper/sections/generated/table_core_evidence.tex
paper/sections/generated/table_robustness.tex
paper/sections/generated/results_macros.tex
```

사용자가 기억할 최종 명령은 하나여야 한다.

```bash
GPU_IDS=0,1 bash local_run/run_tofu_1p5b_table2.sh
```

이 shell은 Python을 import하기 전에 `.venv` 상태부터 확인하고, 필요하면
Section 2의 설치를 수행한 다음 preflight와 state machine으로 이어져야 한다.
별도 수동 설치 명령을 먼저 실행해야만 동작하는 wrapper로 만들지 않는다.

같은 명령을 다시 실행하면 완료 unit은 검증 후 건너뛰고, 다음 허용 stage부터
계속해야 한다. 단, 논문의 prospective freeze를 보존하기 위해 human gate에서는
의도적으로 멈춘다. 사용자가 freeze를 검토하고 커밋한 뒤 같은 명령을 다시
실행하면 된다.

## 2. 이 환경의 고정 경로

이 작업에서는 루트 `CLAUDE.md`의 H100 클러스터 경로 설명보다 아래 로컬
워크스테이션 경로를 우선한다. 과학적 freeze/seal 규칙은 `AGENTS.md`를 그대로
지킨다.

```text
repo:              /home/minsoo3.kim/dev/retain-susceptibility
venv:              <repo>/.venv
model:             /rdata/models/Qwen2.5-1.5B-Instruct
HF_HOME:           /rdata/minsoo3.kim/hf_home
results root:      /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b
GPU:               physical GPU 0,1, RTX 4090 24GB each
```

### 최초 설치

원큐 shell entrypoint는 설치까지 idempotent하게 처리해야 한다. `.venv`가
있고 아래 import/CUDA 검사를 통과하면 재설치하지 않는다. 없거나 불완전할
때만 다음과 동등한 설치를 수행한다.

```bash
cd /home/minsoo3.kim/dev/retain-susceptibility
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install "torch==2.7.1"
.venv/bin/python -m pip install -e ".[dev,campaign]"
```

이 저장소의 로컬 4090 검증 기준은 Python 3.11, PyTorch
`2.7.1+cu126`이다. 기존 환경에 호환되는 cu12x PyTorch가 설치되어 있고
GPU 검사가 통과하면 정확한 local version suffix가 다르다는 이유만으로
교체하지 않는다. CPU-only torch는 통과시키지 않는다.

설치 검증:

```bash
.venv/bin/python - <<'PY'
import torch
import transformers
import datasets
import sentence_transformers
import yaml

assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 2
assert torch.version.cuda and torch.version.cuda.startswith("12.")
print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu0": torch.cuda.get_device_name(0),
    "gpu1": torch.cuda.get_device_name(1),
})
PY
```

설치 원칙:

- `sudo`, system Python 변경, global pip install을 사용하지 않는다.
- model/dataset이 없다고 다른 checkpoint나 dataset을 자동 다운로드하지
  않는다.
- Python package 설치 실패는 사용한 command와 마지막 오류를 남기고 멈춘다.
- 설치 후 `pip freeze`, Python/PyTorch/transformers/datasets 버전과 GPU
  이름을 `<results root>/environment.json`에 기록한다.

### 실행 환경

```bash
cd /home/minsoo3.kim/dev/retain-susceptibility
source .venv/bin/activate
export HF_HOME=/rdata/minsoo3.kim/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

경로가 없으면 임의로 다운로드하거나 다른 모델로 바꾸지 말고, preflight를
실패시키면서 누락 경로를 정확히 출력한다.

## 3. 이미 구현된 것

다음을 새로 만들지 말고 재사용한다.

| 기능 | 현재 위치 |
|---|---|
| 1.5B PDF-v4 runtime | `configs/paper/tofu_v4.yaml` |
| 1.5B model/campaign setting | `configs/paper/campaign.yaml` |
| Table 2의 1.5B boundary row | `configs/paper/evidence.yaml` |
| exact stage manifest 생성 | `experiments/paper/init_v4_stage.py` |
| unit 실행 및 stage seal | `experiments/paper/run_v4_stage.py` |
| TOFU GPU unit producer | `experiments/paper/tofu_v4_unit.py` |
| parent/claim 선택 | `experiments/paper/select_tofu_v4.py` |
| raw plan 및 ledger | `experiments/paper/init_raw_plan.py`, `aggregate_raw.py` |
| Table 2A/2B renderer | `experiments/paper/build_evidence.py` |
| request/seed SFT 재사용 | `runs/paper/tofu_v4/sft_cache` 계약 |
| 로컬 모델/데이터 경로 관례 | `local_run/README.md`, `local_run/run_one.sh` |

현재 빈 부분:

1. `run_v4_stage.py`는 unit을 순차 실행하며 GPU lane 지정과 resume가 없다.
2. `run_tofu_table1.py`는 마지막에 Table 1만 생성하고 `--paper-root`로
   Table 2를 만들지 않는다.
3. paper config의 모델 경로는 `/group-volume/models/...`라 로컬 `/rdata`
   경로 overlay가 필요하다.
4. `configs/paper/tofu_parent_freeze_1p5b.yaml`은 현재
   `status: draft`, 7개 parent가 모두 unresolved다.
5. global `configs/paper/selection_freeze.yaml`도 target 전에 human freeze가
   필요하다.

## 4. 사용 금지 경로

다음은 이번 Table 2 evidence에 사용하지 않는다.

- `experiments/gate_1p5b/gate.py`
- `local_run/run_one.sh`, `run_queue.sh`, `sweep_ours.sh`
- `experiments/local_pdf_v4.py`
- legacy `table1.json`, `table2.json`, `crossed.json`

이들은 진단 또는 이전 표용이며 현재 `prediction_raw.jsonl`,
`fidelity_raw.jsonl`, `protection_raw.jsonl`의 exact roster를 만들지 않는다.
`gate.py --smoke`는 tiny random CPU model이므로 실제 1.5B/4090 메모리
검증으로 인정하지 않는다.

## 5. GPU 실행 방식

1.5B official unit 하나를 GPU 두 장에 model-shard하지 않는다. 각 child
process가 모델 하나를 가지고, 서로 다른 unit 두 개를 GPU 0과 GPU 1에서
동시에 실행한다.

```text
worker A: CUDA_VISIBLE_DEVICES=0, unit process sees cuda:0
worker B: CUDA_VISIBLE_DEVICES=1, unit process sees cuda:0
```

지켜야 할 조건:

- dtype은 `float32` 그대로 유지한다.
- `trainable_scope: probe_block`, `block_last_n: 8`을 유지한다.
- `device_map=balanced` 또는 `split:*`를 official PDF-v4 unit에 넣지 않는다.
- worker 수는 GPU 수를 넘지 않는다.
- child마다 별도 `unit.log`를 남긴다.
- 한 unit 실패가 이미 완료된 다른 unit의 결과를 지우면 안 된다.
- 동시 SFT cache miss가 같은 request/seed 파일을 만들지 않도록 unit 순서를
  `(request, seed)` cache key 기준으로 배치하거나 cache-level exclusion을
  사용한다.

현재 parent trajectory optimizer는 `TrajectoryConfig.trainable_pattern`으로
마지막 8개 down-projection block만 선택한다. repair도 같은 block에서 수동
gradient/update를 수행한다. 이 경로를 유지하고 full-model AdamW로 되돌리지
않는다.

## 6. 먼저 만들 hardware preflight

원큐 runner는 실제 실험 전에 두 GPU를 모두 점검해야 한다.

필수 점검:

1. `nvidia-smi`에서 GPU 0,1이 존재하고 각각 RTX 4090, 24GB인지 확인한다.
2. 다른 process가 사용 중이면 kill하지 말고 PID와 사용량을 출력한 뒤 멈춘다.
3. `.venv`의 `torch.cuda.is_available()` 및 device count를 확인한다.
4. model, tokenizer, sentence encoder, TOFU Arrow cache를 offline으로
   열 수 있는지 확인한다.
5. 두 child를 동시에 띄워 각 GPU에서 실제 Qwen2.5-1.5B fp32를 로드한다.
6. last-8 down-projection block만 `requires_grad=True`로 설정한다.
7. 작은 real-token forward/backward 1회, block-scoped AdamW step 1회,
   repair velocity와 같은 `zeros_like` block vector 할당을 수행한다.
8. GPU별 peak allocated/reserved memory를 JSON으로 기록한다.

결과:

```text
<results root>/preflight/hardware.json
<results root>/preflight/gpu-0.log
<results root>/preflight/gpu-1.log
```

이 검사가 OOM이면 batch size나 scientific knob를 자동 변경하지 않는다.
어느 할당 단계에서 몇 GiB로 실패했는지 보고하고 종료한다.

## 7. 구현할 단일 state machine

권장 구조:

```text
local_run/run_tofu_1p5b_table2.sh
experiments/paper/run_tofu_1p5b_4090x2.py
```

기존 `run_tofu_table1.py`를 공용 함수로 정리해 재사용해도 된다. 단, 기존
CLI와 테스트를 깨지 않는다.

### State 0: local config overlay

canonical config를 직접 로컬 경로로 오염시키지 않는다. runner가 canonical
YAML을 읽고 아래 operational field만 바꾼 파생 config를 results root에
원자적으로 쓴다.

```text
models.Qwen2.5-1.5B.source
runtime.sft_cache_root
```

생성 예:

```text
<results root>/config/campaign.local.yaml
<results root>/config/tofu_v4.local.yaml
<results root>/config/overlay_diff.json
```

`overlay_diff.json`에서 위 세 경로 이외의 차이가 하나라도 있으면 실패한다.
campaign ID, roster, seeds, parent grid, thresholds, bootstrap, repair,
selection rule은 바꾸지 않는다.

### State 1: calibration

- exact planned unit 수: `7 parents x 2 D_cal requests x 2 seeds = 28`.
- `run_v4_stage.py`에 `--gpus 0,1 --resume` 또는 동등한 API를 추가한다.
- 완료 판단은 파일 존재 여부가 아니라 기존 output validator와 SHA-256
  검증을 모두 통과한 경우로 제한한다.
- stage가 끝나면 parent selection을 `--freeze` 없이 실행해 draft proposal을
  만든다.

```text
<results root>/freeze_proposals/tofu_parent_freeze_1p5b.yaml
```

여기서 종료 코드와 함께 다음을 출력한다.

```text
HUMAN_FREEZE_REQUIRED: parent
proposal: <absolute path>
unresolved: [...]
```

7개 parent 중 하나라도 unresolved면 freeze를 위조하거나 target으로
진행하지 않는다. 현재 저장소의 기존 기록상 1.5B가 모두 unresolved였으므로
이 결과가 다시 나올 가능성이 높다. 그것은 runner 실패가 아니라 유효한
development 결과다.

### Human gate A

사람이 proposal과 development diagnostics를 검토하고
`configs/paper/tofu_parent_freeze_1p5b.yaml`을 승인/커밋해야 한다. Agent가
`status: frozen`으로 바꾸거나 대신 커밋하지 않는다.

같은 원큐 명령을 다시 실행했을 때 canonical parent freeze가 `frozen`이고
source artifact hash가 현재 sealed calibration과 맞으면 State 2로 간다.

### State 2: prediction and protection development

- prediction: `7 x 4 D_pred x 2 seeds = 56` units.
- protection: `7 x 4 D_prot x 2 seeds = 56` units.
- stage 둘은 각각 exact seal을 만든다.
- 두 sealed selection input을 사용해 `--freeze` 없이 claim-selection draft를
  만든다.

```text
<results root>/freeze_proposals/selection_freeze.yaml
```

여기서 다음을 출력하고 멈춘다.

```text
HUMAN_FREEZE_REQUIRED: claims
proposal: <absolute path>
unresolved_primary: [...]
```

### Human gate B

사람이 claim proposal을 검토하고 `configs/paper/selection_freeze.yaml`로
승인/커밋한다. Agent는 target 결과를 보기 전에 frozen commit이 존재하는지
검증해야 한다.

### State 3: target evaluation

- exact planned unit 수: `7 x 10 target requests x 2 seeds = 140`.
- physical GPU 두 장에 한 unit씩 배치한다.
- prediction, fidelity, protection raw가 모두 unit validator를 통과해야
  완료로 센다.
- non-reach, infeasible arm, IUT failure는 결과이며 재튜닝 사유가 아니다.
- target 결과를 본 뒤 seed, grid, alpha, Kp, threshold를 바꾸지 않는다.

### State 4: ledger and LaTeX

반드시 세 sealed target shard를 함께 사용한다.

```bash
python experiments/paper/init_raw_plan.py \
  --campaign <local campaign> \
  --evidence configs/paper/evidence.yaml \
  --selection-freeze configs/paper/selection_freeze.yaml \
  --setting tofu_qwen25_1p5b \
  --out <results root>/raw_plan.json

python experiments/paper/aggregate_raw.py \
  --plan <results root>/raw_plan.json \
  --prediction-raw <target sealed>/prediction_raw.jsonl \
  --fidelity-raw <target sealed>/fidelity_raw.jsonl \
  --protection-raw <target sealed>/protection_raw.jsonl \
  --core-only \
  --out <results root>/evidence_ledger.json

python experiments/paper/build_evidence.py \
  --config configs/paper/evidence.yaml \
  --ledger <results root>/evidence_ledger.json \
  --readiness-out <results root>/evidence_readiness.json \
  --paper-root paper
```

기존 `run_tofu_table1.py`처럼 `--table1-out`만 호출하면 Table 2가 생성되지
않는다. 최종 단계에는 반드시 `--paper-root paper`가 있어야 한다.

## 8. Resume 및 실패 계약

원큐 명령 재실행 시:

- manifest/config hash가 같고 unit output 전체가 검증되면 `SKIP valid`.
- partial, hash mismatch, validation failure는 자동 삭제하거나 덮어쓰지 않는다.
- partial 경로와 검증 오류를 출력하고 사람이 보존/이동하도록 멈춘다.
- 실행 중 SIGINT/SIGTERM을 받으면 새 child launch를 중단하고 실행 중 child를
  정상 종료한 뒤 이미 완료된 unit은 보존한다.
- stage seal은 planned unit 전체가 valid일 때만 다시 쓴다.
- 로그 파일명은 unit identity를 포함하고 서로 공유하지 않는다.
- 각 stage 끝에 planned/skipped/completed/failed 카운트를 JSON으로 남긴다.

## 9. 테스트

GPU 없이 가능한 테스트를 먼저 추가하고 실행한다.

```bash
PYTHONPATH=src:tests pytest -q --noconftest \
  tests/test_paper_stage_executor.py \
  tests/test_tofu_v4_pipeline.py \
  tests/test_evidence_tables.py \
  tests/test_paper_evidence.py
```

새 테스트가 검증해야 할 항목:

1. GPU `0,1`에 최대 두 subprocess만 동시 실행된다.
2. 각 child는 정확히 하나의 `CUDA_VISIBLE_DEVICES` 값을 받는다.
3. valid unit만 resume skip된다.
4. corrupt/partial unit은 skip되지 않으며 자동 삭제되지 않는다.
5. 두 worker 중 하나가 실패해도 성공 unit 산출물은 보존된다.
6. parent/claim draft 상태에서 target command를 실행하지 않는다.
7. local overlay는 허용된 operational field 세 개만 바꾼다.
8. final build command에 `--paper-root paper`가 포함된다.
9. generated robustness file에 두 label이 정확히 한 번씩 있다.

```text
\label{tab:robustness}
\label{tab:robustness-funnel}
```

## 10. 완료 조건

코드 완료:

- 원큐 shell entrypoint가 환경 설치를 자동 처리하고 `--help`, `--plan`,
  `--preflight`, `--install-only`를 지원한다.
- plan은 stage별 `28 / 56 / 56 / 140` unit과 GPU lane 두 개를 출력한다.
- CPU 테스트가 모두 통과한다.
- shell/Python syntax와 `git diff --check`가 통과한다.
- README 또는 이 문서에 최종 사용자 명령과 output 위치가 맞게 남는다.

GPU 실행 완료:

- hardware preflight가 두 GPU에서 실제 1.5B fp32 backward/step을 통과한다.
- 두 human freeze가 이미 승인된 경우 target 140 units가 모두 valid다.
- ledger의 1.5B parent row 7개가 attempted이며 raw 세 종류가 같은 exact
  support를 사용한다.
- `table_robustness.tex`가 생성되고 1.5B row에 실제 funnel/decision 값이
  들어간다.

과학적으로 진행 불가한 경우:

- calibration 또는 claim selection이 unresolved면 그 상태에서 멈추는 것이
  정상 완료다.
- 이 경우 proposal, diagnostics, stage summary, 로그 경로를 보고한다.
- fallback을 valid로 바꾸거나 target을 억지로 실행해서 표를 채우지 않는다.

## 11. 최종 보고 형식

작업을 마치면 다음만 간결하게 보고한다.

```text
command:
current state:
GPU preflight:
stage counts:
cache hits/misses:
human gate:
ledger:
Table 2 LaTeX:
tests:
remaining blocker:
```
