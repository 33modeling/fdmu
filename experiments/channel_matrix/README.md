# Channel-matrix diagnostic campaign

이 디렉터리는 7B/14B TOFU의 이전 channel-matrix 진단을 보존한다. 최신
PDF-v4의 `D_cal/D_pred/D_prot/target` roster와 Table 1/2 claim workflow가
아니므로 결과를 paper target evidence로 재명명하지 않는다.

## 구현

```text
run_campaign.py                 fidelity/calibration/audit runner
alpha_protection.py             development/audit protection arms
select_freeze.py                objective freeze proposal
select_prediction_freeze.py     prediction freeze proposal
select_alpha_freeze.py          protection alpha freeze proposal
aggregate.py                    diagnostic aggregation
h100_campaign.sh                단일 H100 phase wrapper
```

Scientific roster와 runtime 값은 Markdown이 아니라 다음 tracked config가
기준이다.

```text
configs/channel_matrix/7b_tofu.yaml
configs/channel_matrix/14b_tofu.yaml
configs/channel_matrix/objective_freeze*.yaml
configs/channel_matrix/prediction_alpha_freeze*.yaml
configs/channel_matrix/alpha_protection_freeze*.yaml
```

## 직접 진단

```bash
GPU=0 MODEL_ID=qwen25_7b \
  bash experiments/channel_matrix/h100_campaign.sh preflight

GPU=0 MODEL_ID=qwen25_7b \
  bash experiments/channel_matrix/h100_campaign.sh fidelity
```

Calibration/audit/alpha phase는 직접 여러 개 띄우지 말고 cluster queue를
사용한다.

```bash
bash experiments/cluster/run_tofu_7b_h100.sh
bash experiments/cluster/run_tofu_14b_h100.sh
```

서로 다른 모델 실행기는 서로 다른 머신에서 실행한다. Queue와 결과는
`runs/` 아래에 생성되고 Git이 추적하지 않는다.

## 불변 조건

- Objective와 alpha는 development 결과로만 제안하고 사람이 commit한 freeze를
  audit 전에 확인한다.
- Audit outcome으로 설정을 다시 고르지 않는다.
- Partial run, seal, manifest를 덮어쓰거나 삭제하지 않는다.
- Fidelity certificate summary는 RQ2의 per-unit `fidelity_raw.jsonl`을
  대신하지 않는다.
- PDF-v4 evidence가 필요하면 `experiments/paper/` producer와
  [paper evidence pipeline](../../docs/PAPER_EVIDENCE_PIPELINE.md)을 사용한다.
