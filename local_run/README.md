# RTX 4090 x2 runbook

로컬 GPU 실행은 아래 두 경로만 지원한다.

## 1. TOFU 1.5B 개발 파이프라인

## 지금 calibration까지 실행한 경우

아래 원클릭 명령으로 전체 파이프라인을 실행하거나 재개할 수 있다.

```bash
cd /path/to/fdmu
GPU_IDS=0,1 bash local_run/run_tofu_1p5b_4090x2.sh
```

4090 경로는 FP32 수식과 마지막 8개 down-projection 블록을 유지하되,
제약-gradient basis를 기본적으로 CPU RAM에 보관한다. Qwen2.5-1.5B의
해당 블록은 gradient 벡터 하나가 약 0.41GiB이므로 GPU에 basis를 누적하지
않는다. 로그의 `[memory-plan]`, `[repair-memory]`, `[CUDA_OOM]` 행에서
계산값과 실제 peak를 확인할 수 있다. 특별한 검증 목적이 아니면
`RSUS_REPAIR_BASIS_STORAGE`를 `model`로 덮어쓰지 않는다.

런처는 완료된 calibration unit을 검증해 재사용하고, `pending=0`이면 parent
freeze를 자동 생성·검증한 뒤 joint sweep과 최종 LaTeX까지 계속 진행한다.
오류 후 같은 명령을 실행하면 완료된 단계는 재사용하고 미완료 단계부터 이어진다.

재실행 시 terminal marker가 유효한 단계는 다음처럼 즉시 건너뛴다.

```text
[CALIBRATION SKIPPED] ... retraining=0
[PARENT FREEZE SKIPPED] ... recompute=0
[JOINT SWEEP SKIPPED] ... retraining=0
[DECLARED FIDELITY SKIPPED] ... rerun=0
[LATEX SKIPPED] ... rerun=0
```

완료 마커가 없는 partial 단계만 재개한다. 유닛마다 `[UNIT REUSED]` 또는
`[UNIT PENDING] ... reason=...`을 출력하므로 재학습 여부와 이유를 로그에서
구분할 수 있다. SFT는 `[SFT_CACHE] HIT ... training_skipped=true`일 때
다시 학습하지 않는다.

Joint sweep은 첫 실행의 campaign/evidence/runtime를
`<RUN_ROOT>/joint_sweep/FROZEN_INPUTS.json`과 `frozen_inputs/`에 고정한다.
실행 중 `git pull`로 tracked config가 바뀌어도 기존 trial의 저장 작업량과
SHA-256을 기준으로 원래 config를 복구하므로 새 fingerprint로 처음부터
재실행하지 않는다.

## 전체 흐름

실행 순서:

```text
environment bootstrap + dependency/CUDA/data 검사
  -> parent calibration
  -> resolved parent freeze 자동 검증
  -> joint development sweep
  -> strict 판정 또는 best observed 자동 선택
  -> declared setting-level fidelity
  -> D_pred -> selection freeze -> target
  -> raw evidence -> table1.tex
```

원클릭 실행기는 parent proposal을 sealed
calibration input에서 다시 계산해 일치 여부를 확인하고, `BEST.json`은
target-free terminal winner와 trial 산출물을 검증한다. 각 단계의 SHA-256
freeze 기록을 남긴 뒤 자동으로 다음 단계로 진행한다.

최종 LaTeX는 기본적으로
`/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/final/table1.tex`에
생성된다. 완료된 prediction/target unit은 manifest와 artifact SHA-256을
검증한 뒤 재사용하며, 전체 로그는 `final/launcher_logs/current.log`에서 본다.
원클릭 전체 로그는 `<RUN_ROOT>/launcher_logs/current.log`, 하위 단계 로그는
각각 `parent_calibration/launcher_logs/current.log`,
`joint_sweep/launcher_logs/current.log`,
`fidelity/launcher_logs/current.log`,
`final/launcher_logs/current.log`에 남는다.
Joint 로그는 `CURRENT_BEST`에 판정 상태와 closest trial을, `PROGRESS`에
현재 trial, 전체 완료 수, 실행 시간과 `eta_seconds`를 주기적으로 출력한다.

현재 최상위 단계와 마지막 실패는 다음 두 파일에서 즉시 확인한다.

```text
<RUN_ROOT>/CURRENT_STAGE.txt
<RUN_ROOT>/LAST_ERROR.txt
```

최상위 로그에는 30초마다 `[STAGE i/6] RUNNING` heartbeat가 찍힌다. 최종화
내부 단계는 `<RUN_ROOT>/final/FINAL_CURRENT_STAGE.json`과
`[FINALIZE STAGE i/7]` 로그로 확인한다. 오류 시 `LAST_ERROR.txt`에는 stage,
exit code, line, command, 통합 로그 경로가 기록된다.

각 최상위 단계는 별도 process group에서 실행된다. 오류, `Ctrl-C`, 종료 시
해당 단계의 자식 프로세스에 `TERM`을 보내고 제한 시간 뒤 남은 프로세스만
`KILL`한다. 한 GPU 유닛이 실패하면 같은 trial에서 실행 중인 다른 GPU 유닛도
종료하고 회수한 뒤 오류를 반환한다. 다른 실험의 프로세스는 건드리지 않는다.

현재 Parent calibration은 resolved proposal을 생성하면 exit `0`으로
종료하고 자동 freeze 검증 단계로 진행한다. 이전 checkout에서 생성된
exit `4` 경계도 재개 호환성을 위해 원큐 실행기가 정상으로 받아들인다.
Exit `3` (`PARENT_CALIBRATION_UNRESOLVED`)은 그대로 중단한다.

모든 단계는 저장소의 `.venv/bin/python`을 사용한다. 환경 bootstrap은
`torch==2.7.1`을 항상 검증하며, `.venv`가 정상이고 버전이 정확하면 재설치하지
않는다.

```bash
bash local_run/bootstrap_4090_env.sh
```

이전 `ensure_4090_yaml.sh` 진입점도 동일한 전체 bootstrap으로 위임한다. 따라서
YAML만 설치된 부분 환경이 만들어지거나 torch pin이 우회되지 않는다.

기본 경로:

```text
model:       /rdata/models/Qwen2.5-1.5B-Instruct
HF cache:    /rdata/minsoo3.kim/hf_home
RUN_ROOT:    /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b
SFT cache:   <RUN_ROOT>/sft_cache
calibration: <RUN_ROOT>/parent_calibration
parent freeze: <RUN_ROOT>/parent_calibration/freeze/tofu_parent_freeze_1p5b.yaml
joint sweep: <RUN_ROOT>/joint_sweep
fidelity:    <RUN_ROOT>/fidelity
final:       <RUN_ROOT>/final
```

전체 저장 위치를 바꿀 때는 `RUN_ROOT` 하나만 지정한다. 세부 경로가 꼭 달라야
할 때만 `CALIBRATION_ROOT`, `SFT_CACHE_ROOT`, `RESULTS_ROOT`, `JOINT_ROOT`,
`FINAL_ROOT`를 덮어쓴다. `JOINT_ROOT`의 기본값은 `RESULTS_ROOT`이므로 custom
sweep 위치를 finalize가 그대로 사용한다. `MODEL_PATH`, `HF_HOME`, `GPU_IDS`도
환경변수로 변경할 수 있다.

Declared fidelity 단계는 target 실행 전에 별도 support에서 certificate를 만들고
`<RUN_ROOT>/fidelity/fidelity_summary.json`에 저장한다. Finalize는
`support=declared_setting_fidelity`인 이 파일만 허용하며 target-support
diagnostics를 RQ2 certificate로 사용하지 않는다.

진행 확인:

```bash
tail -f /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/parent_calibration/launcher_logs/current.log
tail -f /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/joint_sweep/launcher_logs/current.log
tail -f /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/joint_sweep/events.jsonl
tail -f /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/fidelity/launcher_logs/current.log
```

진행 중 sweep의 중간 결과는 자동으로 아래 두 파일에 갱신된다.

```text
<RUN_ROOT>/joint_sweep/live/LIVE_STATUS.md
<RUN_ROOT>/joint_sweep/live/LIVE_STATUS.json
```

현재 실행을 재시작하지 않고 watcher만 붙이는 방법과 로그별 원인 분석 순서는
[`docs/LLM_RUN_DIAGNOSTICS.md`](../docs/LLM_RUN_DIAGNOSTICS.md)를 따른다.

Strict joint dominance를 만족하면 해당 trial을 선택한다. 선언한 sweep을 모두
실행해도 만족하지 않으면 가장 성능이 좋은 관측 trial을 `best_available`로
선택하고, 낮은 성능과 실패 판정을 그대로 기록한 채 LaTeX 생성까지 계속한다.
Parent calibration의 strict gate를 만족하지 못한 parent도 가장 좋은 관측
설정을 fallback으로 선택하고 이후 결과에서 infeasible로 기록한다.
Fidelity threshold 실패도 같은 방식으로 결과에 `false`로 기록되며 최종
`table1.tex`과 `final/RESULT_CONCLUSION.json` 생성을 막지 않는다.

## 2. 범용 단일-arm PDF-v4 진단

이 경로는 paper claim이 아닌 `claim_eligible: false` 로컬 진단이다.

```bash
cp configs/local/pdf_v4.example.yaml configs/local/pdf_v4.local.yaml

bash local_run.sh inspect-model configs/local/pdf_v4.local.yaml
bash local_run.sh prepare-manifest configs/local/pdf_v4.local.yaml

# D_cal/D_prot에서 정한 null 값을 채우고 status를
# frozen_for_local_diagnostic으로 변경한 뒤:
bash local_run.sh validate configs/local/pdf_v4.local.yaml
bash local_run.sh run configs/local/pdf_v4.local.yaml
```

`prepare-manifest`와 `run`은 기존 artifact를 덮어쓰지 않는다. Config hash,
block hash/count, fp32, candidate support, first-reaching 조건이 맞지 않으면
fail closed한다.

## 지원 범위

- 4090 x2 공식 자동화 경로는 Qwen2.5-1.5B 개발 sweep이다.
- 7B/14B paper campaign은 H100 runbook을 사용한다.
- 예전 `gate.py`용 하드코딩 래퍼와 외부 문장 인코더 sweep은 현재 PDF-v4
  결과와 섞일 위험이 있어 제거했다.
- 결과와 cache는 `runs/` 또는 `/rdata/.../results`에만 둔다. Markdown 결과를
  tracked source tree에 생성하지 않는다.

## 실패 시 확인 순서

1. Launcher 로그의 첫 `[ERROR]`와 stage 이름을 확인한다.
2. `.venv/bin/python -c 'import torch, transformers, datasets, yaml'`을 실행한다.
3. `nvidia-smi`에서 GPU 0,1의 기존 compute process를 확인한다.
4. 같은 명령을 다시 실행하면 검증된 SFT cache와 완료 unit을 재사용한다.
