# Documentation index

문서는 이 파일에서 찾는다. 최신 PDF-v4 실행에 필요한 문서와 과거 기록을
분리했다.

## 먼저 볼 문서

| 목적 | 문서 |
|---|---|
| 프로젝트 개요와 현재 구현 상태 | [루트 README](../README.md) |
| 최신 PDF-v4와 코드의 일치 여부 | [PDF v4 code audit](PDF_V4_CODE_AUDIT.md) |
| Table 1/2 메트릭 수식과 해석 | [Table 1/2 metric guide](TABLE12_METRICS.md) |
| Predictor 메트릭과 좋은 값의 기준 | [Predictor metric guide](PREDICTOR_METRICS.md) |
| 최종 결과 실행과 확인 | [Final results runbook](FINAL_RESULTS_RUNBOOK.md) |
| 전체 paper evidence 생성 경로 | [Paper evidence pipeline](PAPER_EVIDENCE_PIPELINE.md) |
| 캠페인 실행 전 차단 조건 | [Paper campaign preflight](PAPER_CAMPAIGN_PREFLIGHT.md) |

## 실험 실행

| 작업 | 문서 |
|---|---|
| H100 플릿 운영 | [Cluster fleet runbook](CLUSTER_FLEET_RUNBOOK.md) |
| 새 모델 추가와 캘리브레이션 | [New model calibration](NEW_MODEL_CALIBRATION_GUIDE.md) |
| 1.5B RTX 4090 x2 Table 2 | [1.5B local agent guide](LOCAL_4090X2_TABLE2_AGENT_GUIDE.md) |
| 7B RTX 4090 x2 joint sweep | [7B joint sweep agent guide](LOCAL_4090X2_7B_JOINT_SWEEP_AGENT_GUIDE.md) |
| 14B RTX 4090 x2 하드웨어 판정 | [14B 4090 x2 decision guide](LOCAL_4090X2_14B_HARDWARE_GUIDE.md) |
| 14B H100 joint sweep | [14B joint sweep agent guide](H100_14B_JOINT_SWEEP_AGENT_GUIDE.md) |
| 채널 방향과 score 해석 | [Channel direction](channel_direction.md) |
| 로컬 GPU 실행 | [Local run](../local_run/README.md) |
| 이전 1.5B gate 진단 | [Gate 1.5B runbook](../experiments/gate_1p5b/RUNBOOK.md) |
| 이전 channel-matrix 실험 | [Channel-matrix README](../experiments/channel_matrix/README.md) |

## 결과와 근거 자료

| 자료 | 문서 |
|---|---|
| 7B alpha development | [Alpha development record](data/alpha_dev_7b/README.md) |
| 2026-07-23 calibration | [Calibration record](data/calibration_2026-07-23/README.md) |
| Channel-balance 결과 | [Chanbal2 record](data/chanbal2/README.md) |
| Fidelity 결과 | [Fidelity record](data/fidelity/README.md) |
| CVaR bound 변경 기록 | [CVaR amendment](../prereg/AMENDMENT-2026-07-23-cvar-bound.md) |
| Primary setting 변경 기록 | [Primary-setting amendment](../prereg/AMENDMENT-2026-07-23-primary-setting.md) |

## 과거 계획과 참고 문서

아래 문서는 실행의 현재 기준이 아니라 당시 계획과 설계 기록이다.

| 문서 | 상태 |
|---|---|
| [Paper revision plan](paper_revision_plan.md) | 이전 논문 개정 계획 |
| [2026-07-23 fleet plan](plan_2026-07-23_fleet.md) | 날짜 고정 플릿 계획 |
| [2026-07-23 GPU plan](plan_2026-07-23_gpu.md) | 날짜 고정 GPU 작업 기록 |
| [2026-07-23 Table 1/2 campaign](plan_table12_campaign.md) | 최신 PDF와 roster가 달라 실행 금지 |
| [DESIGN.md](../DESIGN.md) | 구버전 설계 참고 |
| [Channel-matrix design](../experiments/channel_matrix/DESIGN.md) | 이전 캠페인 설계 |
| [Local hyperparameters](../local_run/HYPERPARAMS.md) | 로컬 진단 설정 |
| [Local ours sweep](../local_run/OURS_SWEEP.md) | 로컬 sweep 기록 |

`AGENTS.md`와 `CLAUDE.md`는 사용자 문서가 아니라 실행 에이전트와 클러스터
운영 메모이므로 루트에 유지한다.
