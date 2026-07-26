# Final results and LaTeX runbook

이 문서는 현재 코드 기준으로 기존 결과를 최종 paper LaTeX로 만드는 절차다.
현재 계약의 primary setting은 `TOFU / Qwen2.5-7B`이며, 7개 parent와 9개
setting을 고정 분모로 사용한다.

기준 파일:

```text
paper/
configs/paper/evidence.yaml
src/rsus/evidence/tables.py
```

루트의 `KDD_UnlearningFail.pdf`는 현재 `paper/` 소스보다 오래된
스냅샷이다.

## 1. 기존 7B 결과로 LaTeX만 생성

다음 명령은 CPU-only다. Queue를 변경하지 않고 worker, GPU 학습, SFT,
audit을 시작하지 않는다.

```bash
git pull --ff-only origin main
bash experiments/cluster/run_tofu_7b_h100.sh render-only
```

최종 공유 파일:

```text
/group-volume/fdmu/runs/paper_v4/table_core_evidence.tex
/group-volume/fdmu/runs/paper_v4/table_robustness.tex
```

같은 일반 파일이 선택된 7B run의 다음 폴더에도 복사된다.

```text
<CAMPAIGN_ROOT>/aggregate/paper_v4/table_core_evidence.tex
<CAMPAIGN_ROOT>/aggregate/paper_v4/table_robustness.tex
```

`table_core_evidence.tex`에는 다음 5개 표가 들어간다.

1. Prospective rank prediction
2. Loss Susceptibility fidelity and cost
3. Harmful-tail recovery
4. Fixed-budget repair effects
5. Repair feasibility/native-retention contract

`table_robustness.tex`에는 9개 setting의 claim breadth와 evidence funnel이
들어간다.

터미널에서 바로 확인:

```bash
sed -n '1,360p' /group-volume/fdmu/runs/paper_v4/table_core_evidence.tex
sed -n '1,220p' /group-volume/fdmu/runs/paper_v4/table_robustness.tex
```

렌더 상태와 판정 원본:

```text
/group-volume/fdmu/runs/paper_v4/PUBLISH_STATUS.json
/group-volume/fdmu/runs/paper_v4/evidence_readiness.json
/group-volume/fdmu/runs/paper_v4/evidence_ledger.json
```

## 2. 표 읽는 법

Core 표는 다음 parent를 항상 같은 순서와 분모로 출력한다.

```text
GradDiff, NPO, SimNPO, GRU, RMU, RepNoise, Circuit Breakers
```

`E/P`는 `eligible/pass`다.

| 표기 | 의미 |
|---|---|
| `y/y` | 판정 자격이 있고 해당 조건 통과 |
| `y/n` | 판정 자격은 있지만 해당 조건 실패 |
| `n/--` | 자격 조건이 부족해 pass를 판정하지 않음 |

`Rank E/P`는 `rho(S,d)`, `g_G`, `g_H`, `g_ctl`의 rank condition이다.
최종 `RQ1 E/P`는 이 rank condition에 harmful-tail 조건을 추가한다.
RQ1 성공에는 둘 다 `y/y`가 필요하다.

Forgetting gate에 도달하지 못한 parent는 마지막 완료 checkpoint의 관측값을
descriptive value로 표시할 수 있지만 claim에는 사용할 수 없으므로 `n/--`로
남는다. 미완료 evidence는 `\tblph`로 표시하며 행 자체를 삭제하지 않는다.

## 3. 전체 7B 실험

GPU campaign을 실제로 실행할 때만 `experiment` 모드를 사용한다.

```bash
bash experiments/cluster/run_tofu_7b_h100.sh experiment
```

이 경로는 failed audit 복구, enqueue, 현재 호스트의 빈 GPU별 worker 실행,
monitor, aggregate, 최종 LaTeX publish를 수행한다. `render-only`와
`experiment`는 명시적으로 구분되며 인자 없이 실행하면 종료 코드 2로
중단된다.

Launcher 로그:

```text
/group-volume/fdmu/runs/users/<user>/logs/cluster/
```

Campaign 로그:

```text
<USER_RUN_ROOT>/logs/channel_matrix/
```

## 4. 14B와 1.5B

14B H100 launcher는 현재 experiment-only이며 aggregate 결과는 CSV/JSON
diagnostic이다.

```bash
bash experiments/cluster/run_tofu_14b_h100.sh
```

14B diagnostic:

```text
<USER_RUN_ROOT>/channel_matrix_14b/aggregate/pooled_channel_report.json
<USER_RUN_ROOT>/channel_matrix_14b/aggregate/pooled_channel_report.csv
```

RTX 4090 x2의 1.5B scale-boundary pipeline:

```bash
GPU_IDS=0,1 bash local_run/run_tofu_1p5b_4090x2.sh
```

이 경로는 완료 unit과 검증된 SFT cache를 재사용한다. 1.5B 결과가 존재한다는
사실이 현재 primary를 1.5B로 바꾸지는 않는다.

## 5. 여러 실행 결과 병합

`publish_evidence.py`는 공유 `.publish.lock`을 잡은 뒤 incoming ledger를
`(setting, parent)` 키로 병합한다. 새 setting 결과를 publish해도 다른
setting의 기존 row를 삭제하지 않는다. 같은 키만 새 결과로 교체하고,
병합된 ledger에서 두 최종 LaTeX 파일을 원자적으로 다시 생성한다.

직접 publish가 필요한 경우:

```bash
/group-volume/fdmu/.venv/bin/python experiments/paper/publish_evidence.py \
  --ledger <SETTING_EVIDENCE_LEDGER.json> \
  --combined-root /group-volume/fdmu/runs/paper_v4 \
  --evidence-config configs/paper/evidence.yaml
```

일반적으로 7B에서는 전용 `render-only` 명령을 사용한다.

## 6. 완료 판정

LaTeX 파일이 존재한다는 것과 모든 claim이 성공했다는 것은 다르다.

```bash
jq '.settings.tofu_qwen25_7b' \
  /group-volume/fdmu/runs/paper_v4/evidence_readiness.json
```

각 parent의 `eligible`, `claim_pass`, `reasons`와 setting의 `chain`을 함께
확인한다. 실패하거나 비적격인 결과도 LaTeX는 끝까지 생성되어야 한다.
