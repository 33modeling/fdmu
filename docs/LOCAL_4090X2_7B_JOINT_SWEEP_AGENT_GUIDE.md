# RTX 4090 x2: Qwen2.5-7B joint sweep agent guide

이 문서는 로컬 coding agent가 Qwen2.5-7B TOFU PDF-v4 개발 스윕을
RTX 4090 24GB 두 장에서 준비하고 실행할 때 따를 계약이다. `joint`는 이
프로젝트가 제안하는 방법이며, target 결과를 보면서 설정을 바꾸는 작업은
금지한다.

## 1. 완료 조건

개발 데이터 `D_prot`의 같은 parent first-reaching checkpoint와 같은 candidate
support에서 다음 다섯 arm을 비교한다.

```text
joint, s0, s1, no_repair, repeated_random x 5
```

한 parent가 통과하려면 모든 `D_prot request x seed` 셀에서 다음을 만족해야
한다.

1. `joint.metrics.feasible == true`
2. comparator가 infeasible이면 제약 우선 순위로 joint 승리
3. comparator도 feasible이면
   `joint.mean_damage < comparator.mean_damage`
4. comparator도 feasible이면
   `joint.cvar95_damage < comparator.cvar95_damage`
5. 2--4를 `s0`, `s1`, `no_repair`, 반복 난수의 모든 draw에 각각 적용

스윕 전체 종료 조건은 다음 두 그룹에서 각각 한 parent 이상이 모든 셀을
통과하는 것이다.

```text
output-readout:         graddiff, npo, simnpo, gru
representation-readout: rmu, repnoise, circuit_breakers
```

동률은 승리가 아니다. margin을 설정했다면 joint 값에 margin을 더한 뒤에도
작아야 한다. 판정에는 `D_prot`만 사용하고 `target`과 audit 결과는 읽지 않는다.

## 2. 현재 상태와 중요한 차이

현재 저장소에 이미 있는 것:

| 항목 | 위치 |
|---|---|
| 7B paper setting | `configs/paper/tofu_v4.yaml` |
| 7B parent grid | `configs/channel_matrix/7b_tofu.yaml` |
| frozen parent 설정 | `configs/channel_matrix/objective_freeze.yaml` |
| protection 5-arm producer | `experiments/paper/tofu_v4_unit.py` |
| 개발 스윕 controller | `experiments/paper/run_joint_dev_sweep.py` |
| 1.5B reference spec | `configs/local/joint_sweep_1p5b_4090x2.yaml` |

현재 없는 것:

- paper producer의 검증된 2-GPU model sharding
- 두 GPU를 한 unit에 할당하는 sweep lane
- 7B 전용 local spec과 hardware preflight

`local_run/run_one.sh`의 `DMAP=split:8`은 legacy gate에서만 동작한다.
그 결과를 Table 1/2 또는 이 joint 종료 판정에 섞으면 안 된다.

## 3. 메모리 계약

7B fp32 가중치만 약 28GB이므로 24GB GPU 한 장에는 들어가지 않는다. 7B
trial 하나가 GPU 0과 1을 모두 사용하며 동시에 실행할 수 있는 unit은 하나다.

```text
CUDA_VISIBLE_DEVICES=0,1
unit process: model shard across cuda:0 and cuda:1
parallel units: 1
```

다음 우회는 금지한다.

- bf16/fp16로 바꾸기
- CPU offload 또는 NVMe offload를 검증 없이 켜기
- 1.5B처럼 GPU마다 독립 unit 하나씩 실행하기
- OOM이 날 때 batch, `R`, `eta`, `block_last_n`을 몰래 바꾸기

loss-shake는 실제 파라미터 좌표를 작은 반경으로 교란하므로 claim-bearing
block은 fp32를 유지해야 한다.

## 4. 에이전트가 먼저 구현할 것

### 4.1 Paper runtime의 명시적 sharding

`tofu_v4_unit.py`의 model loader가 local runtime에서만 다음 필드를 받게 한다.

```yaml
runtime:
  device: cuda
  dtype: float32
  device_map: split:8
  max_memory_gib: [22, 22]
```

canonical `configs/paper/tofu_v4.yaml`에 workstation 경로를 직접 넣지 않는다.
resolved local runtime에만 위 값을 넣는다. `split:8`의 의미는 마지막 8개
transformer layer와 claim-bearing block을 `cuda:1`에 두고 나머지를
`cuda:0`에 두는 명시적 map이어야 한다. `auto`나 `balanced`는 허용하지 않는다.

입력은 embedding device로 이동하고, block 연산과 repair tensor는 block
device에 남아야 한다. 기존 `batch_to_model_device()`가 첫 parameter의
device만 가정하는 부분과 Accelerate hook의 이동을 전부 점검한다.

### 4.2 다중 GPU unit 검증

실제 7B checkpoint로 아래 테스트를 통과하기 전에는 스윕을 시작하지 않는다.

1. tokenizer/model offline load
2. layer별 device map과 last-8 block device 기록
3. claim block의 모든 parameter가 fp32이며 한 device에 있는지 확인
4. real TOFU token으로 forward/backward 1회
5. block-scoped optimizer step 1회
6. loss-shake `+eta`, `-eta` 왕복 후 parameter hash 복원 확인
7. parent checkpoint save/load와 fresh model 복원 확인
8. PDF repair accepted step와 rollback을 각각 한 번 검증
9. 두 GPU peak allocated/reserved memory 기록

수치 회귀는 같은 작은 fixture를 단일 H100 fp32로 실행한 결과와 비교한다.
score 순위, forgetting recall, mean damage, CVaR damage가 사전 선언한 tolerance
안에 들어야 한다.

### 4.3 Sweep scheduler 변경

1.5B controller의 `gpus: [0, 1]`은 GPU당 child 한 개를 뜻하므로 그대로 쓰면
안 된다. 7B spec에는 하나의 GPU set을 표현하는 구조를 추가한다.

```yaml
gpu_sets:
  - [0, 1]
```

child에는 `CUDA_VISIBLE_DEVICES=0,1`을 전달한다. `gpu_sets` 수만큼만 unit을
병렬 실행하고, 같은 request/seed SFT cache key는 직렬화한다. controller의
개발 전용 판정 함수와 append-only 로그 형식은 그대로 재사용한다.

## 5. 7B local spec

1.5B spec을 복사하되 아래 항목만 바꾼다.

```yaml
setting: tofu_qwen25_7b
paths:
  model_source: /rdata/models/Qwen2.5-7B-Instruct
  sentence_encoder: /rdata/models/all-MiniLM-L6-v2
  sft_cache_root: /rdata/minsoo3.kim/results/paper/tofu_qwen25_7b/sft_cache
  output_root: /rdata/minsoo3.kim/results/paper/tofu_qwen25_7b/joint_sweep
gpu_sets:
  - [0, 1]
```

`stop` 블록은 변경하지 않는다. 각 trial은 전역 alpha 하나와 Kp 하나를
사용해야 한다. repair knob를 바꾸는 새 trial은 기존 목록 뒤에만 추가한다.

## 6. 실행 순서

에이전트는 아래 순서를 끝까지 수행한다.

1. `AGENTS.md`, 이 문서, PDF-v4 metric 문서를 읽는다.
2. dirty worktree와 현재 commit을 기록한다.
3. Python 3.11 venv와 CUDA-enabled PyTorch를 검사한다.
4. model, encoder, TOFU cache를 offline으로 연다.
5. GPU가 사용 중이면 PID/메모리를 출력하고 멈춘다. 프로세스를 kill하지 않는다.
6. Section 4의 sharding 구현과 CPU/unit test를 완료한다.
7. 실제 7B hardware preflight와 H100 수치 회귀를 완료한다.
8. development-only protection sweep을 실행한다.
9. 실패 unit은 로그를 보존하고 infrastructure 오류부터 수정한다.
10. joint 종료 조건을 만족하면 recommendation을 만들고 멈춘다.

target 실행은 자동으로 이어가지 않는다.

## 7. 산출물

```text
<output_root>/
  environment.json
  sweep_manifest.json
  specs/<sha256>.yaml
  events.jsonl
  summary.csv
  trials/<id>--<hash>/
    config/
    manifest.yaml
    logs/units/<unit>/attempt-*.log
    units/<unit>/protection_diagnostics.json
    stage/stage_manifest.json
    joint_comparison.json
  BEST.json
  recommendation.yaml
```

`BEST.json`은 개발 추천일 뿐 freeze가 아니다. 사용자가 수치와 config hash를
검토하고 prospective freeze를 커밋한 뒤에만 target evaluation을 실행한다.

## 8. 실패 시 행동

- `HUMAN_FREEZE_REQUIRED`: parent freeze를 검토할 때까지 정지
- OOM: 마지막 성공 allocation과 peak memory를 기록하고 정지
- unit failure: 해당 attempt log를 보존하고 다음 trial로 넘어가지 않음
- `NO_JOINT_DOMINANCE` (exit 3): 모든 선언 trial이 끝났지만 성공 조건이
  없으므로 failure report를 남기고 종료
- `PAUSED_BUDGET_LIMIT` (exit 5): `--max-trials` 제한으로 일부만 실행한
  상태이며 결론이 아님
- target 파일이 controller 입력에 등장함: 즉시 실패

스윕이 끝났다는 보고에는 passing parent, 모든 셀 수, trial hash,
`target_used: false`, freeze 필요 여부를 반드시 포함한다.
