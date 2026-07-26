# FDMU: prospective retain-risk prediction and protection

FDMU는 machine unlearning 전에 retain candidate의 손상 위험을 예측하고,
동일한 first-reaching parent checkpoint에서 고정 예산 repair를 비교하는
실험 코드다. 현재 논문 계약은 `D_cal -> D_pred -> D_prot -> target` 순서와
target 이전 freeze를 강제한다.

## 현재 상태

- Loss-shake, exact-gradient fidelity, proximity, joint predictor 구현
- GradDiff, NPO, SimNPO, GRU, RMU, RepNoise, Circuit Breakers parent 구현
- PDF-v4 repair, comparator arms, raw evidence, IUT, LaTeX renderer 구현
- TOFU 1.5B paper unit producer와 4090 x2 개발 스윕 구현
- H100 7B/14B channel-matrix 실행기와 공유 파일 큐 구현
- 최신 PDF 전체 Table 2 roster는 아직 완성되지 않음
- 실제 target campaign 결과가 없으면 generated table의 placeholder가 정상

코드가 구현됐다는 것과 논문 결과가 완성됐다는 것은 다르다. 현재 차단 조건은
다음 명령으로 확인한다.

```bash
python experiments/paper/preflight.py
python experiments/cluster/next_actions.py
```

## 먼저 읽을 문서

문서의 단일 인덱스는 [docs/README.md](docs/README.md)다.

| 목적 | 문서 |
|---|---|
| 로컬 LLM/실행 에이전트 규칙 | [AGENTS.md](AGENTS.md) |
| 4090 x2 실행 | [local_run/README.md](local_run/README.md) |
| H100 플릿 실행 | [docs/CLUSTER_FLEET_RUNBOOK.md](docs/CLUSTER_FLEET_RUNBOOK.md) |
| 최종 결과와 LaTeX | [docs/FINAL_RESULTS_RUNBOOK.md](docs/FINAL_RESULTS_RUNBOOK.md) |
| Table 1/2 수식 | [docs/TABLE12_METRICS.md](docs/TABLE12_METRICS.md) |
| 최신 PDF 대비 구현 상태 | [docs/PDF_V4_CODE_AUDIT.md](docs/PDF_V4_CODE_AUDIT.md) |

## 설치와 CPU 검사

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,campaign]"
.venv/bin/python -m pytest -q
```

GPU 머신에서는 기존 검증된 `.venv`가 있으면 다시 만들지 않는다.

## 실행 진입점

### RTX 4090 x2: TOFU 1.5B 개발 파이프라인

```bash
GPU_IDS=0,1 bash local_run/run_tofu_1p5b_4090x2.sh
```

Calibration, parent-freeze 승인, joint 개발 스윕을 순서대로 실행한다. 완료
unit과 검증된 SFT cache는 재사용한다. 결과와 로그는 기본적으로
`/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/`에 쓴다.

`joint_sweep/BEST.json`을 검토한 뒤 prediction부터 target 평가와 LaTeX
생성까지 이어서 실행한다.

```bash
APPROVE_JOINT_BEST=1 GPU_IDS=0,1 \
  bash local_run/finalize_joint_sweep_to_latex.sh
```

### 범용 로컬 PDF-v4 진단

```bash
cp configs/local/pdf_v4.example.yaml configs/local/pdf_v4.local.yaml
bash local_run.sh inspect-model
bash local_run.sh prepare-manifest
bash local_run.sh validate
bash local_run.sh run
```

이 경로는 단일 로컬 diagnostic이며 paper claim을 만들지 않는다. 자세한 config
절차는 [local_run/README.md](local_run/README.md)를 따른다.

### H100 7B와 14B

현재 H100 실험에 사용하는 머신은 **총 4대**다.
서로 다른 머신에서 각각 한 명령을 실행한다.

실험 전에 각 머신에서 다음 조건을 먼저 확인한다.

```bash
git pull --ff-only origin main
git status --short                         # 출력이 없어야 함
git log -1 --oneline                       # 실행 커밋 기록
bash experiments/cluster/setup_group_volume.sh
source /group-volume/fdmu/.venv/bin/activate
source experiments/cluster/cluster_env.sh
df -h /group-volume/fdmu
```

Audit에는 모델별 fidelity certificate가 필수다. 인증서 누락을 무시하거나
audit gate를 우회하지 않는다. 아래 원클릭 실행기는 시작 단계에서 해당 모델의
preflight와 fidelity를 실행하고, 통과한 인증서를 확인한 뒤에만 audit을
적재한다.

```bash
bash experiments/cluster/run_tofu_7b_h100.sh
bash experiments/cluster/run_tofu_14b_h100.sh
```

수동 실행이 필요한 경우에는 audit보다 먼저 다음 명령으로 인증서를 생성한다.

```bash
GPU=0 CONFIG=configs/channel_matrix/7b_tofu.yaml MODEL_ID=qwen25_7b \
  bash experiments/channel_matrix/h100_campaign.sh fidelity

GPU=0 CONFIG=configs/channel_matrix/14b_tofu.yaml MODEL_ID=qwen25_14b \
  bash experiments/channel_matrix/h100_campaign.sh fidelity
```

`RuntimeError`, 특히 `inline_container.cc:659 unexpected pos`는 무시 가능한
경고가 아니라 PyTorch checkpoint 저장 실패다. 전체 결과를 삭제할 필요는
없지만, 실행 중이던 이전 코드의 프로세스를 종료하고 현재 `main`에서 해당
큐의 failed unit만 재시도해야 한다. 상세 절차는
[클러스터 런북의 실험 전 필수 확인](docs/CLUSTER_FLEET_RUNBOOK.md#실험-전-필수-확인)을
따른다.

이 두 명령은 기존 channel-matrix campaign용이다. 최신 PDF-v4 전체 Table 1/2
실행으로 오인하면 안 된다. 공유 큐, 로그, 복구 절차는 cluster runbook에 있다.

### 최신 PDF-v4 TOFU Table 1

```bash
python experiments/paper/run_tofu_table1.py \
  --action plan --setting tofu_qwen25_1p5b
```

`plan` 결과와 freeze 상태를 검토한 뒤에만 `--action run`을 사용한다.

## 핵심 구조

```text
src/rsus/                 probe, parent, repair, evidence 핵심 라이브러리
experiments/paper/        PDF-v4 stage, raw evidence, table 생성
experiments/cluster/      H100 공유 큐와 노드 실행기
experiments/channel_matrix/ 이전 7B/14B 진단 campaign
local_run/                4090 x2와 범용 로컬 실행기
configs/paper/            논문 roster와 evidence 계약
configs/local/            로컬 diagnostic template
prereg/                   동결 상수와 amendment
tests/                    CPU 계약 및 회귀 테스트
```

## 과학적 불변 조건

1. Parent는 direct-forgetting gate를 처음 통과한 저장 checkpoint다.
2. Predictor/protection 설정은 target 결과를 보기 전에 동결한다.
3. Joint, component, random, no-repair arm은 같은 parent와 예산을 사용한다.
4. Non-reaching, infeasible, incomplete row도 분모에 남긴다.
5. Freeze, seal, manifest, 기존 run artifact를 성공시키기 위해 수정하지 않는다.
6. `all_tables_ready`가 아니면 최종 paper result라고 보고하지 않는다.

과거 계획서와 구버전 실행 가이드는 저장소에서 제거했다. 필요한 변경 이력은
Git history에서 확인한다.
