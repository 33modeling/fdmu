# RTX 4090 x2 runbook

로컬 GPU 실행은 아래 두 경로만 지원한다.

## 1. TOFU 1.5B 개발 파이프라인

기본 실행:

```bash
cd /path/to/fdmu
GPU_IDS=0,1 bash local_run/run_tofu_1p5b_4090x2.sh
```

실행 순서:

```text
dependency/CUDA/data 검사
  -> parent calibration
  -> resolved parent freeze 승인
  -> joint development sweep
  -> trial 검증과 종료 판정
```

Parent calibration의 exit `4`는 실패가 아니라 resolved proposal에 대한
승인 경계다. 원큐 실행기는 이 상태만 정상으로 받아 다음
`parent-freeze-approval` 단계로 진행한다. Exit `3`
(`PARENT_CALIBRATION_UNRESOLVED`)은 그대로 중단한다.

세 단계는 모두 저장소의 `.venv/bin/python`을 사용한다. `.venv`가 정상이면
재생성하지 않는다. `yaml` import만 깨졌다면 다음 검사로 복구와 진단을 수행한다.

```bash
bash local_run/ensure_4090_yaml.sh
```

기본 경로:

```text
model:       /rdata/models/Qwen2.5-1.5B-Instruct
HF cache:    /rdata/minsoo3.kim/hf_home
results:     /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b
SFT cache:   <results>/sft_cache
calibration: <results>/parent_calibration
joint sweep: <results>/joint_sweep
```

필요하면 환경변수로 `MODEL_PATH`, `HF_HOME`, `CALIBRATION_ROOT`,
`SFT_CACHE_ROOT`, `RESULTS_ROOT`, `GPU_IDS`를 바꾼다. 저장소의 tracked
config나 shell을 머신별 경로로 수정하지 않는다.

진행 확인:

```bash
tail -f /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/parent_calibration/launcher_logs/current.log
tail -f /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/joint_sweep/launcher_logs/current.log
tail -f /rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/joint_sweep/events.jsonl
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
