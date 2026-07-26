# RTX 4090 x2 runbook

로컬 GPU 실행은 아래 두 경로만 지원한다.

## 1. TOFU 1.5B 개발 파이프라인

설정 확인부터 최종 LaTeX까지 원클릭 실행:

```bash
cd /path/to/fdmu
GPU_IDS=0,1 bash local_run/run_tofu_1p5b_4090x2.sh
```

실행 순서:

```text
environment bootstrap + dependency/CUDA/data 검사
  -> parent calibration
  -> resolved parent freeze 승인
  -> joint development sweep
  -> trial 검증과 종료 판정
  -> declared setting-level fidelity
  -> D_pred -> selection freeze -> target
  -> raw evidence -> table1.tex
```

원클릭 실행기는 parent proposal과 `BEST.json` 전체를 각각 출력하고 파일
SHA-256 앞 12자리에 묶인 승인 문구를 요구한다. 입력이 일치하지 않거나
터미널이 아닌 실행에서는 target으로 진행하지 않는다. 실행 전에 설정한
`APPROVE_JOINT_BEST` 값은 원클릭 런처가 제거한다.

단계별 실행과 재개도 지원한다. 먼저 calibration과 sweep까지만 실행한다.

```bash
RUN_FINALIZE=0 GPU_IDS=0,1 \
  bash local_run/run_tofu_1p5b_4090x2.sh
```

`joint_sweep/BEST.json`을 검토한 뒤 finalize만 실행한다.

```bash
bash local_run/run_tofu_1p5b_fidelity.sh

APPROVE_JOINT_BEST=1 GPU_IDS=0,1 \
  bash local_run/finalize_joint_sweep_to_latex.sh
```

최종 LaTeX는 기본적으로
`/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/final/table1.tex`에
생성된다. 완료된 prediction/target unit은 manifest와 artifact SHA-256을
검증한 뒤 재사용하며, 전체 로그는 `final/launcher_logs/current.log`에서 본다.
원클릭 전체 로그는 `<RUN_ROOT>/launcher_logs/current.log`, 하위 단계 로그는
각각 `parent_calibration/launcher_logs/current.log`,
`joint_sweep/launcher_logs/current.log`,
`fidelity/launcher_logs/current.log`,
`final/launcher_logs/current.log`에 남는다.

Parent calibration의 exit `4`는 실패가 아니라 resolved proposal에 대한
승인 경계다. 원큐 실행기는 이 상태만 정상으로 받아 다음
`parent-freeze-approval` 단계로 진행한다. Exit `3`
(`PARENT_CALIBRATION_UNRESOLVED`)은 그대로 중단한다.

세 단계는 모두 저장소의 `.venv/bin/python`을 사용한다. 환경 bootstrap은
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
환경변수로 변경할 수 있다. 저장소의 tracked config나 shell을 머신별 경로로
수정하지 않는다.

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

`PARENT_CALIBRATION_UNRESOLVED`나 `NO_JOINT_DOMINANCE`는 숨길 오류가 아니라
유효한 종료 결과다. Freeze 파일을 손으로 채우거나 target 결과를 보고 sweep을
확장하지 않는다.

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
4. Partial output은 삭제하지 말고 별도 forensics 경로로 옮긴다.
5. 같은 명령을 다시 실행해 검증된 SFT cache와 완료 unit만 재사용한다.
