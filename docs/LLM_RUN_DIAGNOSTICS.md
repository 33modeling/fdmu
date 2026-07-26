# Local LLM Run Diagnostics

이 문서는 로컬 LLM과 실행 에이전트가 로그를 찾아 원인을 분류하기 위한
가이드다.

## 공통 진단 순서

1. `git rev-parse --show-toplevel`, `git status --short`, `hostname`을 확인한다.
2. 아래 환경별 top-level launcher `current` 로그의 마지막 200줄을 읽는다.
3. 마지막 `stage`, 첫 traceback, 실제 process exit를 구분한다.
4. launcher가 가리키는 worker/unit 로그와 상태 JSON을 직접 읽는다.
5. `nvidia-smi`, `df -h`를 확인하고 근거가 있을 때만 OOM/디스크/NFS로 분류한다.
6. 원인, 근거 파일, 보존되는 결과, 안전한 다음 동작을 함께 보고한다.

`STALE CLAIM`은 원인이 아니라 heartbeat 중단 결과다. 정상적인 child 오류는
queue의 `failed` 또는 `pending`으로 기록된다. stale이면 owning host/PID,
worker 로그, unit 로그, OOM/NFS 흔적을 확인하기 전 원인을 단정하거나
requeue하지 않는다.

## RTX 4090 x2: 1.5B

전체 자동 실행과 재개에 사용하는 명령:

```bash
GPU_IDS=0,1 bash local_run/run_tofu_1p5b_4090x2.sh
```

기본 root:

```text
/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5B
```

실제 기본값은 소문자 `1p5b`다:

```text
/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b
```

| 목적 | 경로 |
|---|---|
| 전체 원클릭 로그 | `<RUN_ROOT>/launcher_logs/current.log` |
| 현재 최상위 단계 | `<RUN_ROOT>/CURRENT_STAGE.txt` |
| 마지막 실패 요약 | `<RUN_ROOT>/LAST_ERROR.txt` |
| calibration 로그 | `<RUN_ROOT>/parent_calibration/launcher_logs/current.log` |
| sweep 로그 | `<RUN_ROOT>/joint_sweep/launcher_logs/current.log` |
| sweep 이벤트 | `<RUN_ROOT>/joint_sweep/events.jsonl` |
| 실시간 중간 요약 | `<RUN_ROOT>/joint_sweep/live/LIVE_STATUS.md` |
| 기계 판독 중간 요약 | `<RUN_ROOT>/joint_sweep/live/LIVE_STATUS.json` |
| 중간 요약 변경 이력 | `<RUN_ROOT>/joint_sweep/live/history.jsonl` |
| 완료 trial 비교 | `<RUN_ROOT>/joint_sweep/trials/*/joint_comparison.json` |
| unit 실행 로그 | `<RUN_ROOT>/joint_sweep/trials/*/logs/units/*/attempt-*.log` |
| unit 종료 상태 | `<RUN_ROOT>/joint_sweep/trials/*/logs/units/*/attempt-*.json` |
| 완료 trial 요약 | `<RUN_ROOT>/joint_sweep/summary.csv` |
| sweep 종료 상태 | `<RUN_ROOT>/joint_sweep/SWEEP_STATUS.json` |
| 개발 winner | `<RUN_ROOT>/joint_sweep/BEST.json` |
| fidelity 로그 | `<RUN_ROOT>/fidelity/launcher_logs/current.log` |
| finalize 로그 | `<RUN_ROOT>/final/launcher_logs/current.log` |
| finalize 내부 단계 | `<RUN_ROOT>/final/FINAL_CURRENT_STAGE.json` |
| finalize 완료 마커 | `<RUN_ROOT>/final/FINALIZATION_STATUS.json` |
| strict/best-available 결론 | `<RUN_ROOT>/final/RESULT_CONCLUSION.json` |
| 최종 LaTeX | `<RUN_ROOT>/final/table1.tex` |

최상위 순서는 `environment-bootstrap` → `calibration` →
`automatic-parent-freeze` → `joint-sweep` → `declared-fidelity` →
`target-evidence-latex`의 6단계다. `CURRENT_STAGE.txt`의 `state`, `stage_index`,
`stage_total`, `elapsed_seconds`, `log`를 먼저 읽는다. `state=failed`이면
`LAST_ERROR.txt`에서 exit code, line, command를 확인한 뒤 그 파일의 `log`를
읽는다.

재개 로그 판정:

| 로그 | 의미 |
|---|---|
| `CALIBRATION SKIPPED` | terminal calibration marker가 유효하며 GPU 재학습 없음 |
| `PARENT FREEZE SKIPPED` | 승인 record와 freeze 해시가 일치하며 재계산 없음 |
| `JOINT SWEEP SKIPPED` | `BEST.json`과 terminal sweep status가 유효하며 재학습 없음 |
| `BEST_AVAILABLE_SELECTED` | strict 조건 미달이며 가장 좋은 관측 trial로 계속 진행 |
| `DECLARED FIDELITY SKIPPED` | 기존 setting-level summary와 source hash 재사용 |
| `LATEX SKIPPED` | 최종 marker의 Table 1 및 evidence artifact 해시가 모두 일치 |
| `UNIT REUSED` / `TRIAL_REUSE` | 해당 유닛 검증 완료, `retraining=0` |
| `UNIT PENDING` / `TRIAL_PENDING` | 출력된 `reason` 때문에 해당 유닛만 실행 |
| `SFT_CACHE HIT` | theta0 SFT 학습 생략 |
| `CLEANUP ... signal=TERM/KILL` | 실패 단계가 만든 process group 회수 |

`target-evidence-latex` 내부는 7단계이며
`FINAL_CURRENT_STAGE.json`에서 prediction, selection freeze, target evaluation,
raw/aggregate evidence, Table 1 생성을 구분한다. 완료된 최종화는
`FINALIZATION_STATUS.json`의 artifact SHA-256을 검증한 뒤 전체를 건너뛴다.
Strict joint dominance나 fidelity threshold가 실패해도 측정 실패를 결과로
기록하고 Table 1 생성을 계속한다.

현재 GPU 실행을 재시작하지 않고 read-only watcher만 붙일 때:

```bash
nohup bash local_run/watch_tofu_1p5b_intermediate.sh \
  >> /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/joint_sweep/live/watcher.log 2>&1 &
```

watcher는 기존 artifact를 읽고 `joint_sweep/live/`만 갱신한다. Partial cell
결과는 `descriptive_only`이며 incomplete roster로 trial을 선택하지 않는다.
다음 원클릭 sweep부터 watcher는 자동으로 시작된다.

## H100: 7B와 14B

공유 root는 `/group-volume/fdmu/runs`다.

| 목적 | 7B | 14B |
|---|---|---|
| launcher | `logs/cluster/launcher_qwen25_7b_<host>_current.out` | `logs/cluster/launcher_qwen25_14b_<host>_current.out` |
| campaign | `logs/channel_matrix/qwen25_7b_<action>_<host>_current.log` | `logs/channel_matrix/qwen25_14b_<action>_<host>_current.log` |
| queue | `cluster_queue/wave2` | `cluster_queue/wave1_14b` |
| aggregate | `channel_matrix_7b/aggregate/` | `channel_matrix_14b/aggregate/` |
| LaTeX | `channel_matrix_7b/aggregate/table1_channel_matrix_qwen25_7b.tex` | `channel_matrix_14b/aggregate/table1_channel_matrix_qwen25_14b.tex` |

### 클러스터 결과 저장 구조

로컬 LLM은 아래 두 root를 먼저 변수처럼 해석한다.

```text
RUNS=/group-volume/fdmu/runs
R7=$RUNS/channel_matrix_7b
R14=$RUNS/channel_matrix_14b
```

실제 root의 근거는 각각 `configs/channel_matrix/7b_tofu.yaml`과
`configs/channel_matrix/14b_tofu.yaml`의 `output_root`다. `runs/...` 상대 경로는
cluster에서 `$CLUSTER_RUNS_ROOT/...`로 변환된다. 분석할 때 저장소의 `runs/`나
user volume에 같은 이름의 오래된 디렉터리가 있어도 혼용하지 않는다.

| 단계/역할 | 7B | 14B |
|---|---|---|
| Campaign SFT cache | `$R7/sft_cache/qwen25_7b/` | `$R14/sft_cache/qwen25_14b/` |
| Automatic audit SFT cache | `$RUNS/sft_cache/qwen25_7b-*/` | `$RUNS/sft_cache/qwen25_14b-*/` |
| Audit cell root | `$R7/audit/qwen25_7b/` | `$R14/audit/qwen25_14b/` |
| Alpha development | `$R7/alpha_protection/development/qwen25_7b/` | `$R14/alpha_protection/development/qwen25_14b/` |
| Alpha audit | `$R7/alpha_protection/audit/qwen25_7b/` | `$R14/alpha_protection/audit/qwen25_14b/` |
| Pooled aggregate | `$R7/aggregate/` | `$R14/aggregate/` |

#### 과거 Fidelity 파일

TOFU 7B/14B 런처, audit 재개, aggregate는 fidelity certificate를 사용하지
않는다. `$R7/fidelity/` 또는 `$R14/fidelity/` 아래에 남은 CSV, JSON, `.lock`
파일은 과거 실행 artifact이며 현재 실행 상태나 Table 1/2 결과로 해석하지 않는다.

#### SFT cache

SFT cache에는 두 경로 형식이 있다. Calibration과 alpha protection이 명시적으로
지정하는 campaign cache pair:

```text
<MODEL_ROOT>/sft_cache/<model>/tofu-a<author>_seed-<seed>.pt
<MODEL_ROOT>/sft_cache/<model>/tofu-a<author>_seed-<seed>.pt.json
```

Audit gate의 기본 `--sft-cache auto`가 사용하는 exact-contract cache pair:

```text
$RUNS/sft_cache/<model>-<model-source-hash>/tofu/
  tofu-a<author>__<contract-hash>.pt
  tofu-a<author>__<contract-hash>.pt.json
```

Cluster bootstrap은 checkout의 `runs`를 `$RUNS`로 연결하므로 auto cache도 group
volume에 저장된다. 어떤 cache가 실제 사용됐는지는 각 audit
`run_manifest.json`의 `sft_cache.path`, `hit`, `contract_sha256`을 최종 기준으로
판단한다. 경로 이름만 보고 cache hit를 추정하지 않는다.

`.pt`는 재사용할 모델 state이고 `.pt.json`은 contract, `full_mean_nll`,
`reached`, 크기, SHA-256을 담은 metadata다. 분석 시 metadata의 `integrity`와
`sft_result`를 먼저 읽는다. `.guard`, `.tmp`, local staging 파일은 결과가
아니다. SFT cache는 계산 재사용 artifact이며 Table metric으로 직접 집계하지
않는다.

#### Audit 원자료

현재 frozen roster는 모델마다 author `181, 186, 191`과 seed `2025, 2026`의
6개 cell이다.

```text
<MODEL_ROOT>/audit/<model>/tofu-a<author>/seed-<seed>/
```

각 cell의 주요 파일:

| 파일 | 의미와 사용법 |
|---|---|
| `run_manifest.json` | model, request, seed, candidate roster, objective/predictor 목록, config/fidelity/freeze SHA, code commit을 담은 provenance 기준 |
| `gate.log` | SFT cache hit, predictor scoring, objective 진행 및 first-reaching step을 사람이 읽는 실행 로그 |
| `profile_artifacts/<predictor>.json` | predictor별 cost, discovery score, candidate/fold metadata. Audit score는 seal 밖 평문에 두지 않음 |
| `seals/` + `seal_ledger.jsonl` | audit-fold predictor score와 open 이력. 임의로 수정하거나 seal 파일만 단독 해석하지 않음 |
| `traj_<objective>/damage.json` | step별 forget recall, candidate NLL, damage trajectory |
| `traj_<objective>/DONE` | 해당 objective trajectory가 원자적으로 완료됐다는 marker |
| `table1.json` | 단일 request/seed의 predictor-objective 계산 결과 |
| `channel_report.csv` | 단일 cell의 predictor x objective rho, AUROC, overlap, tail-rho |
| `channel_report.json` | 단일 cell interaction과 bootstrap CI를 포함한 구조화 요약 |

`damage.json`에서는 terminal snapshot을 임의로 고르지 않는다. Aggregate 코드는
frozen `forget_recall_max`를 처음 만족한 snapshot의 damage를 사용한다. 로컬
LLM도 논문 수치 재계산 시
`experiments/channel_matrix/aggregate.py::_first_reaching_damage`와 같은 규칙을
사용해야 한다.

Audit cell 완료 조건은 다음을 모두 만족하는 것이다.

1. `run_manifest.json`의 contract SHA와 code commit이 현재 campaign과 일치한다.
2. 선언된 모든 `traj_<objective>/DONE`과 `damage.json`이 존재한다.
3. 모든 predictor의 seal ledger 상태가 `opened`다.
4. `channel_report.csv`와 `channel_report.json`이 존재한다.

`gate.log`가 끝났거나 `table1.json`만 있다고 완료로 판단하지 않는다.

#### Aggregate와 LaTeX

정상 audit 6개가 모두 끝나면 아래 파일이 생성된다.

```text
<MODEL_ROOT>/aggregate/pooled_channel_report.csv
<MODEL_ROOT>/aggregate/pooled_channel_report.json
<MODEL_ROOT>/aggregate/model_channel_report.csv
<MODEL_ROOT>/aggregate/table1_channel_matrix_<model>.tex
<MODEL_ROOT>/aggregate/table1_stress_<model>.tex
```

분석의 기준은 `pooled_channel_report.json`과 CSV다. JSON에서 `n_runs == 6`,
`requests`, `seeds`, `predictors`, `objectives`, `stress_objectives`,
`objective_status`, `roster_interaction`을 먼저 검사한다. CSV는
predictor-objective별 `rho`, CI, AUROC, overlap, tail-rho를 후속 plot/table
생성에 사용한다. `model_channel_report.csv`는 모델별 축약본이다. `.tex`는 표시용
파생 artifact이므로 원수치 분석의 입력으로 사용하지 않는다.

파일명이 `table1_channel_matrix`여도 이 H100 channel-matrix 결과는 기존 진단
campaign 산출물이다. 최신 PDF-v4의 최종 Table 1/2 evidence와 동일하다고
간주하지 않는다. 최신 paper evidence의 의미와 metric은
`docs/TABLE12_METRICS.md`, `docs/PREDICTOR_METRICS.md`,
`docs/PAPER_EVIDENCE_PIPELINE.md`를 함께 읽어 구분한다.

#### Alpha protection

Alpha cell은 아래에 저장된다.

```text
<MODEL_ROOT>/alpha_protection/<development|audit>/<model>/tofu-a<author>/seed-<seed>/
```

주요 파일은 `run_manifest.json`, `alpha_protection.log`,
`results.partial.jsonl`, `random_draws.partial.jsonl`, 최종 `results.json`,
`DONE`이다. `.partial.jsonl`은 재개용 append log이며 `DONE` 없는 cell을 최종
결과로 사용하지 않는다. Legacy diagnostic aggregate를 명시적으로 실행한 경우에만
`alpha_protection/aggregate/alpha_protection_curve.csv`,
`alpha_protection_contrasts.csv`, `alpha_protection_summary.json`이 생긴다.
이 summary는 자체적으로 `paper_evidence=false`를 선언하므로 최신 paper
evidence로 승격하지 않는다.

### 로컬 LLM 분석 순서

로컬 LLM은 사용자에게 경로 확인 명령을 요청하기 전에 filesystem을 직접 읽고
다음 순서를 따른다.

1. 해당 모델 launcher `current` 로그에서 마지막 stage와 실행 commit을 찾는다.
2. queue의 `done/claimed/pending/failed`를 세어 실행 상태와 완료 결과를 구분한다.
3. `done/<unit>.json`의 `result.log`를 따라가 unit 로그를 찾는다.
4. Fidelity JSON/CSV pair를 검증한다.
5. Audit의 6개 `run_manifest.json`과 completion marker를 roster와 대조한다.
6. 전체 roster가 완성됐으면 pooled JSON/CSV를 우선 분석한다.
7. Aggregate가 없으면 완료 cell의 `channel_report.json`만 중간 결과로 요약하고
   반드시 `partial/descriptive only`라고 표시한다.
8. 수치 해석에는 config, manifest, source artifact 경로를 함께 기록한다.

분석 보고에는 최소한 아래를 포함한다.

```text
model / campaign commit
fidelity passed 여부와 frozen metrics
queue counts와 실패 unit
완료 audit cell 수 / 기대 6개
누락 author-seed-objective
pooled n_runs와 interaction CI (aggregate가 있을 때)
사용한 JSON/CSV의 절대 경로
paper evidence인지 diagnostic artifact인지
```

Queue JSON, log, `.partial.jsonl`, forensics artifact는 실행 상태와 원인 분석에
활용할 수 있지만 최종 metric 입력으로 자동 승격하지 않는다. 분석 목적으로
`retry-failed`, `requeue-stale`, 파일 이동/삭제, seal open을 실행하지 않는다.

공통 worker/unit 로그:

```text
/group-volume/fdmu/runs/logs/cluster/worker_<host>_gpu<gpu>.out
/group-volume/fdmu/runs/logs/cluster/<unit>__<host>_gpu<gpu>__try<n>.out
```

Claim owner와 heartbeat:

```text
<queue>/claimed/<unit>.meta.json
<queue>/claimed/<unit>.hb
```

환경 setup:

```text
/group-volume/fdmu/environment/setup.lock.owner
/group-volume/fdmu/runs/logs/cluster/setup/setup_<host>_<timestamp>_<pid>.out
```

보존된 장애 artifact:

```text
/group-volume/fdmu/runs/forensics/fidelity-artifacts/
/group-volume/fdmu/runs/forensics/sft-cache-corrupt/
/group-volume/fdmu/runs/forensics/audit-partials/
```

## 원인 분류

| 증거 | 분류 |
|---|---|
| `CUDA out of memory`, kernel OOM kill | GPU/host memory |
| `No space left on device` | 해당 로그에 찍힌 filesystem 용량 |
| `Input/output error`, `Stale file handle` | 공유 볼륨/NFS |
| `CodeCommitMismatch`, dirty worktree | 실행 commit 계약 |
| `setup.lock` timeout | 공유 venv 설치/검증 경쟁 |
| `inline_container.cc`, `unexpected pos` | 손상된 PyTorch cache/container |
| `ModuleNotFoundError` | 로그에 기록된 Python executable의 환경 |
| `fidelity certificate mismatch ... /schema` | 과거 코드 로그; 현재 TOFU 7B/14B 경로에서는 발생하지 않음 |
| `STALE CLAIM`만 존재 | 원인 미확정; owner worker/unit 로그 추가 조사 |

로컬 LLM은 filesystem/terminal 접근 권한이 있으면 위 파일을 직접 읽는다.
사용자에게 `tail`, `cat`, `df`, `nvidia-smi` 실행을 요청하지 않는다. 접근할 수
없는 보안망 머신일 때만 필요한 파일 경로와 최소 출력 범위를 요청한다.
