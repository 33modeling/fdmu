# When Does LLM Unlearning Fail?

이 저장소는 machine unlearning 전에 retain candidate의 손상 위험을 예측하고,
동일한 first-reaching parent checkpoint에서 고정 예산 repair를 비교하는
논문 코드다. 현재 논문 제목은 *When Does LLM Unlearning Fail? Predicting
and Protecting Susceptible Retained Behavior*이며, 실행 계약은
`D_cal -> D_pred -> D_prot -> target` 순서와 target 이전 freeze를 강제한다.

## 현재 상태

- 현재 코드와 `paper/`의 primary setting은 `TOFU / Qwen2.5-7B`다.
  Qwen2.5-1.5B는 scale boundary, Qwen2.5-14B는 model-scale setting이다.
- 논문 분모는 9개 setting과 7개 parent(GradDiff, NPO, SimNPO, GRU, RMU,
  RepNoise, Circuit Breakers)로 고정되어 있다.
- Loss Susceptibility, exact-gradient fidelity, Representation Proximity,
  joint predictor, repair arms, raw evidence, IUT 판정, 최신 LaTeX renderer가
  구현되어 있다.
- 7B 기존 결과는 GPU 실행 없이 현재 paper 형식으로 다시 렌더링할 수 있다.
  14B 경로는 현재 CSV/JSON diagnostic aggregate까지만 생성한다.
- 최종 LaTeX 두 파일에는 core 5개 표와 robustness/funnel 2개 표가 들어간다.
  미완료 setting이나 evidence는 고정 분모에서 삭제하지 않고 placeholder로
  남긴다.

체크인된 `KDD_UnlearningFail.pdf`는 현재 `paper/` 소스보다 오래된
스냅샷이다. 코드와 표 생성 계약은 `paper/`, `configs/paper/evidence.yaml`,
`src/rsus/evidence/tables.py`를 기준으로 한다. 코드 구현 여부와 실제 실험
완료 여부는 다르며, 현재 결과 완성도는 생성된
`evidence_readiness.json`과 `FINALIZATION_STATUS.json`으로 확인한다.

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
| 현재 paper/code 계약 | [configs/paper/evidence.yaml](configs/paper/evidence.yaml) |

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

Calibration까지 끝났거나 중간 오류로 종료된 경우 위 명령을 다시 실행하면
완료 unit과 SFT cache를 검증·재사용하고 다음 단계부터 이어서 진행한다.

이전 실패에서 남은 4090 compute 프로세스까지 정리하고 같은 결과를 재개할
때는 아래 복구 원클릭 명령을 사용한다. 선택한 GPU의 기존 compute
프로세스를 종료하므로 선택한 GPU의 작업을 확인한 뒤 실행한다.

```bash
bash local_run/recover_and_run_tofu_1p5b_4090x2.sh
```

환경 bootstrap, calibration, parent-freeze 자동 검증, joint 개발 스윕, declared
fidelity, target, evidence, `table1.tex` 생성을 순서대로 실행한다. 원클릭
실행은 사람의 입력을 기다리지 않는다. target-free 산출물을 재검증하고
SHA-256이 포함된 freeze 기록을 남긴 뒤 자동으로 다음 단계로 진행한다.
완료 unit과 검증된 SFT cache는 재사용한다. 결과와 로그는 기본적으로
`/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/`에 쓴다. 전체 로그는
`launcher_logs/current.log`, 최종 표는 `final/table1.tex`이다.

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

아래 원클릭 실행기는 공유 볼륨 환경 설정, preflight, audit
enqueue/monitor, aggregate, LaTeX 생성을 수행한다. Fidelity certificate는
7B/14B 실행 경로에서 사용하지 않는다. 7B/14B 런처는 커밋된
freeze 설정을 검증해 사용하며 실행 중 키보드 승인 입력을 요구하지 않는다.

```bash
bash experiments/cluster/run_tofu_7b_h100.sh experiment
bash experiments/cluster/run_tofu_14b_h100.sh
```

사용자별 launcher 로그는
`/group-volume/fdmu/runs/users/<user>/logs/cluster/`에 생성된다. 7B와 14B
런처는 실행한 호스트에서 비어 있는 GPU마다 독립 worker를 띄운다. 한 unit은
GPU 한 장을 사용하며, 서로 다른 호스트의 GPU는 각 호스트에서 런처를 실행해야
활성화된다.

`RuntimeError`, 특히 `inline_container.cc:659 unexpected pos`는 무시 가능한
경고가 아니라 PyTorch checkpoint 저장 실패다. 전체 결과를 삭제할 필요는
없지만, 실행 중이던 이전 코드의 프로세스를 종료하고 현재 `main`에서 해당
큐의 failed unit만 재시도해야 한다. 상세 절차는
[클러스터 런북의 실험 전 필수 확인](docs/CLUSTER_FLEET_RUNBOOK.md#실험-전-필수-확인)을
따른다.

7B `experiment`는 aggregate 뒤 최신 논문 표 publish까지 수행한다. 14B
실행기는 현재 experiment-only이며 CSV/JSON 진단 결과만 생성한다. 공유 큐,
로그, 복구 절차는 cluster runbook에 있다.

### 데이터셋 확장: 5개 데이터셋 x 3개 모델

WMDP-bio/MMLU, MUSE-News, RWKU, MUSE-Books, PISTOL을 Qwen2.5
1.5B/7B/14B에서 실행하는 명시적 wrapper 15개가 있다. 각 wrapper는
`all`, `preflight`, `plan`, `calibration`, `freeze`, `audit`, `aggregate`,
`render`, `status` 중 하나를 반드시 받는다. GPU 실험과 기존 결과 렌더를
옵션 생략으로 혼동하지 않는다.

```bash
# 4090 x2 예시
bash local_run/run_wmdp_bio_1p5b_4090x2.sh all

# H100 예시
bash experiments/cluster/run_wmdp_bio_7b_h100.sh all
bash experiments/cluster/run_wmdp_bio_14b_h100.sh all
```

`all`은 preflight, calibration, development-only 자동 freeze, sealed audit,
aggregate, LaTeX를 순서대로 수행한다. 완료 unit과 SFT cache는 검증 후
재사용하고, 중단된 단일 run만 forensics로 이동해 다시 실행한다. 외부 문장
인코더와 fidelity certificate를 사용하지 않는다. 최종 파일은 각
`RUN_ROOT/aggregate/`의 `pooled_channel_report.json`과
`table_channel_matrix.tex`이다. 이 출력은 데이터셋 확장 진단이며 PDF-v4
claim 표로 자동 승격되지 않는다. 전체 명령과 저장 위치는
[로컬 런북](local_run/README.md)과
[클러스터 런북](docs/CLUSTER_FLEET_RUNBOOK.md)에 있다.

### 최종 논문 Table 1/2

이미 완료된 7B 결과로 표만 다시 만들 때는 다음 CPU-only 명령 하나를 실행한다.
queue, worker, 학습은 시작하지 않는다.

```bash
bash experiments/cluster/run_tofu_7b_h100.sh render-only
```

최종 LaTeX는 아래 두 파일만 사용한다.

```text
/group-volume/fdmu/runs/paper_v4/table_core_evidence.tex
/group-volume/fdmu/runs/paper_v4/table_robustness.tex
```

- `table_core_evidence.tex`: prediction rank, Loss Susceptibility fidelity,
  harmful-tail recovery, repair effects, repair contract의 5개 core 표다.
  GradDiff, NPO, SimNPO, GRU, RMU, RepNoise, CB를 항상 모두 출력한다.
- `table_robustness.tex`: 9개 setting의 robustness와 evidence funnel을
  각각 출력한다.
- 완료된 evidence는 숫자로 채우고, 미완료·비적격 evidence는 행을 없애지 않고
  `\tblph` 또는 `n/--`로 표시한다. 값을 임의로 만들지 않는다.
- forgetting 기준에 도달하지 못한 parent는 마지막 완료 checkpoint의 관측값을
  descriptive 수치로 표시하고 `n/--`로 claim-ineligible임을 명시한다.
- 모델별 `aggregate/paper_v4/`에는 ledger, readiness, status와 실제 일반
  파일인 `table_core_evidence.tex`, `table_robustness.tex`을 둔다. 구형 이름의
  중복 LaTeX는 재렌더 시 삭제한다.
- channel-matrix 진단값은 `aggregate/pooled_channel_report.csv`와
  `pooled_channel_report.json`에서 확인한다. 별도 진단 `.tex`는 만들지 않는다.

`E/P`는 `eligible/pass`다. `Rank E/P`는 prospective rank 조건만 판정하고,
최종 `RQ1 E/P`는 rank 조건에 harmful-tail 조건을 추가한다.

| 표기 | 의미 |
|---|---|
| `y/y` | 판정 자격이 있고 해당 조건 통과 |
| `y/n` | 판정 자격은 있지만 해당 조건 실패 |
| `n/--` | 자격 조건이 부족해 pass를 판정하지 않음 |

RQ1 성공을 주장하려면 `Rank E/P`와 최종 `RQ1 E/P`가 모두 `y/y`여야 한다.
Rank가 `y/y`이고 RQ1이 `y/n`이면 rank 조건은 통과했지만 harmful-tail
조건이 실패한 것이다.

최신 코드를 받은 뒤 기존 결과를 재렌더하는 전체 명령은 다음과 같다.

```bash
git pull --ff-only origin main
bash experiments/cluster/run_tofu_7b_h100.sh render-only
sed -n '1,320p' /group-volume/fdmu/runs/paper_v4/table_core_evidence.tex
sed -n '1,320p' /group-volume/fdmu/runs/paper_v4/table_robustness.tex
```

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
5. Freeze, seal, manifest의 SHA-256과 provenance를 결과에 함께 기록한다.
6. `all_tables_ready` 상태를 최종 결과와 함께 보고한다.

과거 계획서와 구버전 실행 가이드는 저장소에서 제거했다. 필요한 변경 이력은
Git history에서 확인한다.
