# Local LLM Run Diagnostics

이 문서는 로컬 LLM과 실행 에이전트가 사용자에게 진단 명령을 떠넘기지 않고
직접 로그를 찾아 원인을 분류하기 위한 필수 가이드다. 실행 중인 GPU 작업,
queue, cache, manifest, seal은 수정하거나 삭제하지 않는다.

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
| 최종 LaTeX | `<RUN_ROOT>/final/table1.tex` |

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
| `STALE CLAIM`만 존재 | 원인 미확정; owner worker/unit 로그 추가 조사 |

로컬 LLM은 filesystem/terminal 접근 권한이 있으면 위 파일을 직접 읽는다.
사용자에게 `tail`, `cat`, `df`, `nvidia-smi` 실행을 요청하지 않는다. 접근할 수
없는 보안망 머신일 때만 필요한 파일 경로와 최소 출력 범위를 요청한다.
