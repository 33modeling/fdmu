# 멀티노드 H100 플릿 런북 (노드당 8×H100, 총 100장+)

전체 하드웨어 풀 용량과 별개로, 현재 이 저장소의 H100 실험에 할당해 사용하는
머신은 **총 4대**다. 실행 에이전트는 임의로 다섯 번째 머신을 추가하지 않는다.

`experiments/cluster/`는 스케줄러 없는 사내 클러스터에서 8-GPU 노드 여러 대를
하나의 작업 풀로 묶는 최소 오케스트레이션 계층이다. 조율 매체는 전 노드가
공유하는 `/group-volume` 레포 안의 파일 큐 하나뿐이다 — 데몬도, 추가 의존성도
없다 (표준 라이브러리 + pyyaml).

```
experiments/cluster/
  workqueue.py     공유 파일 큐 (pending → claimed → done|failed, 원자적 rename)
  worker.py        GPU 1장을 전담하는 워커 루프 (하트비트 + 로그 + 재시도)
  make_units.py    캠페인 config에서 작업 단위(JSONL) 생성/즉시 enqueue
  launch_node.sh   노드 부팅: venv 활성화 후 GPU당 워커 1개 nohup 기동
  next_actions.py  읽기 전용 오라클: 지금 enqueue 허용된 페이즈/막힌 이유를
                   동결 게이트 기준으로 보고 (자율 에이전트는 이걸 먼저 실행;
                   레포 루트 AGENTS.md가 에이전트용 운영 계약)
```

## 핵심 설계

- **공유 상태와 scratch를 분리.** queue/results는 checkout 위치와 무관하게
  `/group-volume/fdmu/runs`를 사용하되 Unix 사용자별로 자동 분리한다.
  기본 경로는 각각
  `/group-volume/fdmu/runs/cluster_queue/users/<user>/`와
  `/group-volume/fdmu/runs/users/<user>/`다. `CLUSTER_RUNS_ROOT`는
  외부 env나 unit payload가 변경할 수 없다. `HOME`, `TMPDIR`, Triton/CUDA,
  Hugging Face, RSUS, Torch, XDG, pip 캐시는
  `/group-volume/fdmu/runtime/<user>/<host>/`에 분리한다.
  기존 `/group-volume/data/hf_home`은 읽기 전용 원본 cache로 사용한다.
  user-volume, 노드 로컬 `/tmp`, 실제 `~/.cache`에는 쓰지 않는다.
  읽기 전용 HF source가 다른 경우에만 Git에서 무시되는
  `.cluster_env.local.sh`에 `CLUSTER_HF_HOME`을 설정한다. Runtime, queue,
  assignment와 storage root는 override할 수 없다. Queue assignment는
  `configs/cluster/fleet.yaml`을 커밋해 변경한다.
- **사용자별 재실험 격리.** `cluster_env.sh`가 `$USER`를 안전한
  `FDMU_RUN_USER`로 정규화한다. `make_units.py`는 사용자명과 결과 루트를 각
  queue unit에 고정하므로 다른 계정의 worker가 unit을 실행해도 요청자의
  결과 디렉터리만 사용한다. 사용자명이 다르면 unit ID가 같아도 큐와 결과가
  겹치지 않는다. `configs/cluster/legacy_run_owner.txt`의 기존 소유자는 과거
  `/group-volume/fdmu/runs/channel_matrix_*`와 큐를 그대로 사용하므로 이미
  학습한 결과를 이동하거나 다시 학습하지 않는다.
- **작업 단위 = 기존 러너가 이미 지원하는 최소 샤드.** run 디렉토리가 단위 간
  절대 겹치지 않도록 자름: calibration/audit은 `--only-authors <한 명>`,
  alpha 페이즈는 `--worker --author A --seed S`. 모든 명령에 `--resume`이
  들어가므로 재시도/스테일 회수가 안전하다.
- **실행 커밋 고정.** `make_units.py`가 enqueue 시점의 Git SHA를 각 unit에
  기록한다. Worker checkout의 SHA가 다르면 모델 명령을 실행하지 않고
  `CodeCommitMismatch`로 실패한다. 여러 노드 중 한 곳만 `git pull`한 상태로
  같은 sealed queue를 처리할 수 없다.
- **GPU당 워커 1개.** fp32 7B는 H100 한 장을 거의 다 쓰므로 겹배치 금지 규칙을
  코드로 강제 — 워커는 시작 시 `nvidia-smi`로 자기 GPU에 1GiB 이상 상주 메모리가
  있으면 기동 거부(`--allow-busy-gpu`로만 해제).
- **크래시 복구.** 워커는 60초마다 하트비트 파일을 갱신한다. 노드가 죽으면
  `workqueue.py requeue-stale`이 오래된 claim을 pending으로 되돌린다(시도 횟수
  증가, 기본 `max_attempts=2` 소진 시 failed로 이동).
- **동결 경계는 큐에 넣지 않는다.** select-freeze / select-alpha-freeze는
  사람 리뷰 단계 그대로: 한 페이즈를 다 비우고 → 셀렉터 실행 → freeze 커밋 →
  다음 페이즈 enqueue.

## 사용 순서

### 기존 checkout `runs/` 이동

이전 버전이 checkout의 user-volume에 만든 `runs/`는 worker를 모두 중지한 뒤
한 번만 이동한다.

```bash
bash experiments/cluster/migrate_runs_to_group_volume.sh
```

이 스크립트는 파일을 삭제하지 않고
`/group-volume/fdmu/runs`로 옮긴 뒤 기존 `runs`를 그 위치의
symlink로 바꾼다. 실행 중 cluster process가 있거나 일부 파일이 남으면
중단한다.

### 0. 매 세션 공통 (기존 규칙 그대로)

```bash
cd /path/to/fdmu
bash experiments/cluster/setup_group_volume.sh  # 최초 1회
source /group-volume/fdmu/.venv/bin/activate
source experiments/cluster/cluster_env.sh
printf 'user=%s\nqueue=%s\nresults=%s\n' \
  "$CLUSTER_RUN_USER" "$CLUSTER_USER_QUEUE_ROOT" "$FDMU_CAMPAIGN_RUNS_ROOT"
git pull --ff-only origin main
git status --short          # 출력이 없어야 함
git log -1 --oneline        # 이 커밋을 실행 기록에 남김
df -h /group-volume/fdmu
python -m pytest -q
```

기존 소유자는 legacy 경로가 자동 선택된다. 다른 계정에서 namespace 도입 전
공유 경로의 결과만 후처리해야 할 때에만 `FDMU_SHARED_LEGACY_RUNS=1`을
붙인다. 새 실험에는 사용하지 않는다. 기존 소유자가 의도적으로 완전히 독립된
재실험을 만들 때는 `FDMU_FORCE_USER_NAMESPACE=1`을 사용한다.

```bash
FDMU_SHARED_LEGACY_RUNS=1 \
  bash experiments/cluster/render_tofu_7b_h100.sh
```

기존 7B 결과의 LaTeX만 다시 만들 때는 전체 원클릭 런처를 실행하지 않는다.
아래 두 명령은 동일한 render-only 경로이며 queue/retry/enqueue/worker를 전혀
실행하지 않는다.

```bash
bash experiments/cluster/render_tofu_7b_h100.sh
bash experiments/cluster/run_tofu_7b_h100.sh render-only
```

7B 전체 실험을 재개하거나 다시 수행할 때만 `experiment`를 명시한다. 옵션을
생략하면 비용이 큰 GPU 작업을 실수로 시작하지 않고 사용법과 함께 종료한다.

```bash
bash experiments/cluster/run_tofu_7b_h100.sh experiment
```

### 실험 전 필수 확인

7B와 14B audit은 다음 순서를 지킨다.

1. `/group-volume/fdmu/.venv`를 활성화하고 `yaml`, `torch`, `datasets`,
   `transformers` import 및 GPU 인식을 preflight로 확인한다.
2. 커밋된 objective freeze와 clean worktree를 확인한다.
3. 모델별 원클릭 런처를 실행한다. 런처가 audit enqueue/monitor, aggregate,
   LaTeX 생성을 순서대로 수행한다.

TOFU 7B/14B 실행에는 fidelity certificate 단계나 certificate gate가 없다.
공유 볼륨에 남아 있는 과거 `fidelity/*.json`, `fidelity/*.csv`, `.lock` 파일은
실행·재개·집계에 사용하지 않는다. 기존 완료 audit 셀은 실제 실험 계약
(objective freeze, dtype, predictor/objective roster, candidate pool)이 같으면
재사용하고, 실패하거나 부분 저장된 셀만 격리 후 다시 실행한다.

실행 중 `RuntimeError`가 발생하면 해당 unit은 완료된 것이 아니다. 특히
`inline_container.cc:659 unexpected pos`는 Python 659줄이 아니라 PyTorch
ZIP checkpoint writer 내부의 저장 실패 위치다. 경고로 무시하거나 실패한
run 디렉토리를 정상 결과로 취급하지 않는다.

코드를 수정한 뒤에도 이미 실행 중인 Python 프로세스는 이전 코드를 계속
사용한다. 완료된 unit과 정상 SFT cache는 보존하고, 영향받은 모델 프로세스만
종료한 뒤 failed unit만 재시도한다.

```bash
# 현재 상태와 실패 로그 확인
python experiments/cluster/workqueue.py status --brief \
  --queue /group-volume/fdmu/runs/cluster_queue/wave1_14b

# 14B 저장 오류를 내던 이전 run_campaign 프로세스만 종료
pkill -TERM -f 'run_campaign.py.*14b_tofu.yaml' || true

# worker가 unit을 failed로 기록한 뒤 해당 큐의 실패분만 재시도
python experiments/cluster/workqueue.py retry-failed \
  --queue /group-volume/fdmu/runs/cluster_queue/wave1_14b
bash experiments/cluster/launch_node.sh \
  --dedicated-queue /group-volume/fdmu/runs/cluster_queue/wave1_14b 1
```

7B도 같은 원칙을 적용하되 큐는
`/group-volume/fdmu/runs/cluster_queue/wave2`다. `done` 디렉토리, 정상
결과, 정상 SFT cache를 삭제하거나 큐 전체를 초기화하지 않는다. 현재 SFT
cache writer는 worker별 stage에 legacy 순차 형식으로 쓴 뒤 SHA-256을
검증하고, 최종 이름을 게시하는 짧은 구간만 공유 cache를 잠근다. 24시간을
넘긴 abandoned 임시 파일만 자동 제거한다.

14B의 최종 `.pt/.json` cache pair가 불완전하거나
`inline_container.cc:659 unexpected pos` 계열로 손상된 경우에는
`runs/forensics/sft-cache-corrupt/`로 보존 이동하고 cache miss로 재학습한다.
`probe_block` 캠페인은 전체 fp32 14B state가 아니라 SFT가 수정하는 마지막
8개 down-projection만 `sft-cache-v3`에 저장한다. 새 cache는 key/shape/dtype,
크기와 SHA-256을 다음 load에서 검증한다. 이전 전체-model cache와 계약
불일치는 forensics로 보존한 뒤 새 block cache를 만든다. 일반적인 일시 I/O
오류는 숨기거나 무한 재시도하지 않고 즉시 실패시킨다.
14B 원클릭 실행기도 7B와 동일하게 빈 GPU마다 worker 하나를 시작한다. 이미
같은 사용자/queue의 worker가 실행 중이면 그대로 유지하고 나머지 빈 GPU만
추가한다. 노드 launcher는 호스트 lease 아래에서 충돌 검사와 worker 생성을 수행하고,
watcher/worker에는 lease FD를 넘기지 않는다. lease 대기는 기본 60초 후
실패한다. 3초 안에 죽은 worker의 로그 tail을 출력한 뒤 즉시 실패한다.

### 1. 작업 enqueue (아무 노드에서 1회)

```bash
# calibration 페이즈를 큐에 적재 (모델×저자 샤드)
python experiments/cluster/make_units.py \
  --config configs/channel_matrix/7b_tofu.yaml \
  --phase calibration --enqueue --queue runs/cluster_queue/calib

# 여러 페이즈를 한 큐에 순서대로 쌓을 수도 있다 (의존성 없는 것끼리만!)
python experiments/cluster/make_units.py \
  --phase fidelity --phase calibration \
  --enqueue --queue runs/cluster_queue/wave1
```

`--out units.jsonl`로 파일만 뽑아 검토 후 `workqueue.py enqueue`로 넣어도 된다.
같은 unit id는 큐 어느 상태에 있든 재적재 거부 (seal append-only 원칙과 동일).

### 2. 노드 투입 (노드마다 1줄)

```bash
# 8-GPU 노드 전체 투입 — 워커 8개가 nohup으로 뜨고 즉시 반환
bash experiments/cluster/launch_node.sh runs/cluster_queue/calib

# GPU 일부만 쓰거나, 큐가 비면 워커가 스스로 종료하게 하려면
bash experiments/cluster/launch_node.sh runs/cluster_queue/calib 4
WAIT=0 bash experiments/cluster/launch_node.sh runs/cluster_queue/calib
```

기본값 `WAIT=1`이면 큐가 비어도 워커가 30초 간격으로 폴링하며 대기하므로,
**노드를 먼저 다 띄워놓고 나중에 페이즈를 enqueue하는 운영이 가능**하다.
13개 노드에 같은 명령을 치면 워커 ~104개가 한 큐를 나눠 가진다.

### 3. 모니터링 / 정비

```bash
python experiments/cluster/workqueue.py status --queue runs/cluster_queue/calib
# → 상태별 개수 + 실행 중 unit의 host/gpu/하트비트 나이 + 실패 unit의 exit code/로그 경로

# 죽은 노드의 claim 회수 (하트비트 30분 초과 기본)
python experiments/cluster/workqueue.py requeue-stale --queue runs/cluster_queue/calib

# 원인 수정 후 failed 전체 재시도
python experiments/cluster/workqueue.py retry-failed --queue runs/cluster_queue/calib

# 노드 하나의 워커 전부 중지
pkill -f "experiments/cluster/worker.py --queue"
```

로그는 unit 단위로 `runs/logs/cluster/<unit>__<host>_gpu<g>__try<n>.out`,
워커 자체 로그는 `runs/logs/cluster/worker_<host>_gpu<g>.out` — 호스트명이
파일명에 박히므로 공유 볼륨에서 덮어쓰기 사고가 없다.

## 캠페인 페이즈 → 큐 웨이브 매핑

| 웨이브 | enqueue | 샤드 수(모델당) | 완료 후 사람 단계 |
|---|---|---|---|
| 1 | `--phase fidelity --phase calibration` | 1 + 2 | `h100_campaign.sh select-freeze` → objective_freeze 커밋 |
| 2 | `--phase audit` | 3 | `h100_campaign.sh aggregate` |
| 3 | `--phase alpha-development` | 2 (저자×시드) | `select-alpha-freeze` → alpha_protection_freeze 커밋 |
| 4 | `--phase alpha-audit` | 6 (저자 3×시드 2) | `legacy-alpha-diagnostic`, evidence 파이프라인 |

7B 단일 모델 기준 샤드 폭이 GPU 수보다 작으므로, 남는 GPU는 같은 큐에
**시드 복제 런·추가 모델(llama31_8b 프로비저닝 후)·다른 config의 unit**을 함께
적재해 채운다. `make_units.py`가 못 만드는 임의 명령도 JSONL 한 줄이면 된다:

```json
{"unit_id": "chanbal2-s2026", "cmd": ["python", "-u", "experiments/gate_1p5b/gate.py", "--seed", "2026", "..."], "gpus": 1, "max_attempts": 1}
```

## 실패 triage — 봉인 러너의 partial 디렉토리는 자동 재시도로 안 살아난다

calibration/audit 러너는 설계상 **부분 산출물이 남은 run 디렉토리를 절대
재사용하지 않는다** (forensics 보존, seal append-only). 따라서 유닛이 도중에
죽으면 자동 재시도(2번째 attempt)는 "partial or pre-existing directory"
메시지로 몇 초 만에 실패하고 failed로 떨어진다 — 이건 데이터 보호가 작동한
것이지 큐 버그가 아니다. 복구 절차:

```bash
python experiments/cluster/workqueue.py status --queue <Q>   # failed의 log 경로 확인
less <log>                                                   # 죽은 원인 파악 (OOM? 노드?)
mv runs/channel_matrix_7b/calibration/.../<부분런> \
   runs/forensics/<부분런>.$(date +%s)                        # 부분 산출물 보존·이동
python experiments/cluster/workqueue.py retry-failed --queue <Q>
```

`requeue-stale`은 **해당 host의 워커가 정말 죽었는지 확인한 뒤에만** 실행할 것
(status가 보여주는 host에 들어가 프로세스 확인). NFS 지연으로 하트비트만 늦은
살아있는 런을 requeue하면 같은 run 디렉토리에 이중 실행이 붙을 수 있다.
기본 임계 30분은 하트비트 주기(60초)의 30배라 정상 지연으로는 안 걸린다.
원클릭 monitor도 이 임계를 넘은 claim을 무한 `running`으로 표시하지 않고
종료하며, 안전 확인 전 자동 requeue는 하지 않는다.

## 병렬 폭 감각

calibration 유닛 1개 = gate 런 14개(7 objective × 2 설정) 직렬 ≈ GPU 1장을
오래 점유. 7B 단일 모델의 calibration 웨이브는 유닛 2개뿐이므로 플릿 전체가
아니라 **GPU 2장짜리 웨이브**다 — 이때 남는 GPU에 fidelity, 1.5B 시드 복제,
다른 config 유닛을 같이 적재해 채우는 게 맞다. 플릿이 진짜로 넓게 도는 건
audit(모델×저자 3)과 alpha-audit(모델×저자×시드 6) + 복수 모델부터다.

## 운영 주의

- audit 계열 unit은 러너 자체의 dirty-worktree 가드를 그대로 통과해야 하므로,
  **enqueue 전에 커밋 상태를 정리**할 것. 워커가 뜬 뒤 레포를 고치면 이후
  unit부터 바뀐 코드로 돌게 된다 — 봉인 페이즈 중 `git pull` 금지.
- run-tag를 쓰는 커스텀 unit(`gate.py` 등)은 재시도 시 같은 태그로 Exit 1이
  나므로 `max_attempts: 1`로 넣는 게 안전하다 (위 예시처럼).
- 큐 디렉토리는 `runs/` 아래라 git이 추적하지 않는다. 캠페인 산출물 회수는
  기존 방식(채팅 전달 → 세션 커밋) 유지.

## 부록: enqueue_table12.sh — Table 1/2 잔여 웨이브 원샷 적재

> **이 부록의 Table 1/2 매핑은 최신 PDF paper contract가 아니다.**
> `audit-7b`, `audit-14b`와 아래 표는 7B-primary/14B를 포함한 이전 9행
> campaign config를 설명한다. 최신 PDF는 1.5B-primary/7B-scale/Llama-family
> 및 8행 Table 2를 요구한다. Config가 PDF와 동기화되기 전에는 status 조회
> 외의 `enqueue_table12.sh` 명령을 paper evidence 목적으로 실행하지 않는다.

`experiments/cluster/enqueue_table12.sh`는 레포 루트에서 실행하는 운영자용
래퍼다. 워커를 띄우지 않고, `git pull`도 하지 않으며, 큐 상태는
`make_units.py --enqueue` 외에는 건드리지 않는다. 재실행 안전: 같은 unit id는
append-only 큐가 거부하고 스크립트는 "already enqueued"로 알려준다.

```bash
bash experiments/cluster/enqueue_table12.sh              # status (기본)
bash experiments/cluster/enqueue_table12.sh audit-7b     # 7B audit + alpha → wave2
bash experiments/cluster/enqueue_table12.sh audit-14b    # 14B audit + alpha → wave1_14b
bash experiments/cluster/enqueue_table12.sh wmdp         # WMDP fidelity+calibration → wave_wmdp
bash experiments/cluster/enqueue_table12.sh llama        # Llama-8B fidelity+calibration → wave_llama
bash experiments/cluster/enqueue_table12.sh rwku-audit   # RWKU audit → wave_rwku
```

7B와 14B TOFU는 각각 다른 H100 머신에서 별도 큐와 원클릭 실행기를 사용한다.
각 실행기는 certificate 없이 audit을 적재한다. 중복 unit id는 다시 적재되지
않으며 실패하거나 부분 저장된 해당 모델 unit만 복구한다. 빈 GPU마다 워커
하나가 시작된다.
7B 원클릭 런처는 `wave2`를 코드에서 고정한 dedicated mode이므로 실행 머신의
hostname을 `fleet.yaml`에 다시 등록할 필요가 없다. 다른 큐 worker가 같은
머신에서 실행 중이면 GPU 이중 사용을 막기 위해 계속 중단한다.

```bash
bash experiments/cluster/run_tofu_7b_h100.sh
bash experiments/cluster/run_tofu_14b_h100.sh
```

두 명령은 `setup_group_volume.sh`의 idempotent 환경 검사를 먼저 수행하고,
fidelity, enqueue, worker, monitor, aggregate, LaTeX까지 이어서 실행한다.
7B와 14B 실행기는 모두 실행한 호스트의 빈 GPU 전체에 worker를 띄운다.
H100 4장이 보이면 독립 queue unit을 최대 4개 병렬 처리한다. 두 원클릭 런처 모두
hostname/fleet assignment에 의존하지 않으며 monitor가 failed unit의 로그를
launcher 터미널에 표시한다. 각 14B unit 자체는 worker가 배정한
`CUDA_VISIBLE_DEVICES=<gpu>` 한 장을 사용하며 내부 multi-GPU sharding은
사용하지 않는다.

원클릭 launcher 로그:

```text
/group-volume/fdmu/runs/users/<user>/logs/cluster/launcher_qwen25_7b_<host>_current.out
/group-volume/fdmu/runs/users/<user>/logs/cluster/launcher_qwen25_14b_<host>_current.out
```

단계별 `h100_campaign.sh` 로그는
`/group-volume/fdmu/runs/users/<user>/logs/channel_matrix/`에 남는다. 7B 원클릭 런처는
aggregate 뒤에 CPU-only PDF-v4 backfill을 실행하며, 논문에 넣을 최신 형식은
다음 두 파일이다.

```text
/group-volume/fdmu/runs/users/<user>/channel_matrix_7b/aggregate/paper_v4/table1_core_evidence_qwen25_7b.tex
/group-volume/fdmu/runs/users/<user>/channel_matrix_7b/aggregate/paper_v4/table2_robustness_qwen25_7b.tex
```

기존
`aggregate/table1_channel_matrix_qwen25_7b.tex`와 14B의 동명 파일은 진단용
구형 channel matrix다. 논문 최종 Table 1로 사용하지 않는다. 7B channel
matrix가 이미 생성됐다면 GPU 실험이나 기존 aggregate를 다시 돌리지 말고 다음
CPU-only 후처리만 실행한다.

```bash
CONFIG=configs/channel_matrix/7b_tofu.yaml MODEL_ID=qwen25_7b \
  bash experiments/channel_matrix/h100_campaign.sh paper-v4
```

완료된 RQ1 값은 최신 Table 1에 보존된다. 아직 alpha-audit 결과가 없는 RQ3와
setting-level fidelity가 없는 RQ2 셀은 오류로 중단하지 않고 `--`로 표시된다.
`aggregate/paper_v4/FINALIZATION_STATUS.json`에 사용한 parent 수, raw record 수,
보호 결과 완성 여부와 최종 파일 경로가 기록된다.

각 setting의 ledger는 해당 campaign 아래에 그대로 보존되고, 공용 결과는 다음
위치에 setting/parent 키로 병합된다.

```text
/group-volume/fdmu/runs/paper_v4/evidence_ledger.json
/group-volume/fdmu/runs/paper_v4/table1.tex
/group-volume/fdmu/runs/paper_v4/table2.tex
/group-volume/fdmu/runs/paper_v4/PUBLISH_STATUS.json
```

사용자별 raw campaign과 per-setting ledger는 서로 격리되지만, 위 최종 논문
ledger는 전역 `.publish.lock` 아래 병합한다. 이후 다른 사용자가 다른
model/dataset setting을 publish하면 기존 setting 행은 유지되고 새 행만
추가된다. 같은 setting/parent를 재실행한 경우에만 그 행이 갱신된다. publish 마지막에는 위
`table1.tex`과 `table2.tex` 전체가 launcher 로그에 한 번 출력된다.

- `status`: wave2 / wave1_14b / wave_wmdp / wave_llama / wave3_alpha /
  wave4_alpha 큐별 `workqueue.py status --brief` 요약 + `fleet_status.py` 안내.
- `audit-*`: enqueue 전에 (1) 해당 config의 objective_freeze가
  `status: frozen`인지 grep으로 확인(run_campaign.py의 게이트와 동일 기준),
  (2) worktree가 clean한지 확인(audit 러너가 dirty tree를 거부). alpha freeze가
  frozen이면 alpha-audit, 아직 draft면 alpha-development를 같은 큐에 적재.
- `llama`: 모델 경로(`/group-volume/models/Llama-3.1-8B-Instruct`) 부재 시
  `provision_llama.sh` 안내 후 중단. `rwku-audit`: fidelity 인증서 JSON이
  `runs/channel_matrix_rwku7b/fidelity/`에 없으면
  `experiments/diag/fd_fidelity.py --dataset rwku` 안내 후 중단.
- 적재 후 노드 투입은 언제나 수동:
  `bash experiments/cluster/launch_node.sh <큐>` (노드당 워커 8개, GPU 0-7).
  make_units가 만든 unit의 `max_attempts`는 그대로 둘 것.

웨이브 → 큐 → 논문 테이블 매핑:

| 테이블 | 큐 | 내용 |
|---|---|---|
| Table 1 | `wave2` | 7B TOFU audit (+ alpha-development) |
| Table 1 | `wave4_alpha` | 7B alpha-audit (alpha freeze 커밋 후) |
| Table 2 (14B 행) | `wave1_14b` | 14B TOFU audit |
| Table 2 (RWKU 행) | `wave_rwku` | RWKU 7B audit |
| Table 2 (WMDP 행) | `wave_wmdp` | WMDP 7B fidelity+calibration |
| Table 2 (Llama 행) | `wave_llama` | Llama-3.1-8B fidelity+calibration |
