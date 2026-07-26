# Documentation

이 파일이 문서의 단일 인덱스다. 날짜별 작업 계획과 이미 구현이 끝난 agent
prompt는 제거했으며, 과거 내용은 Git history에서만 확인한다.

## 실행

| 환경/목적 | 문서 |
|---|---|
| 모든 로컬 LLM과 실행 에이전트 | [AGENTS.md](../AGENTS.md) |
| RTX 4090 x2 | [Local run](../local_run/README.md) |
| H100 멀티노드 | [Cluster fleet runbook](CLUSTER_FLEET_RUNBOOK.md) |
| 최종 결과와 LaTeX 생성 | [Final results runbook](FINAL_RESULTS_RUNBOOK.md) |
| 새 모델 캘리브레이션 | [New model calibration](NEW_MODEL_CALIBRATION_GUIDE.md) |
| 실행 전 fail-closed 검사 | [Paper campaign preflight](PAPER_CAMPAIGN_PREFLIGHT.md) |
| 로컬 LLM 로그·클러스터 결과·원인 분석 | [Run diagnostics](LLM_RUN_DIAGNOSTICS.md) |

## 논문 계약과 메트릭

| 주제 | 문서 |
|---|---|
| 최신 PDF와 구현 차이 | [PDF-v4 code audit](PDF_V4_CODE_AUDIT.md) |
| Table 1/2 수식과 판정 | [Table 1/2 metrics](TABLE12_METRICS.md) |
| Predictor 값 해석 | [Predictor metrics](PREDICTOR_METRICS.md) |
| Raw evidence에서 표까지 | [Paper evidence pipeline](PAPER_EVIDENCE_PIPELINE.md) |

## 진단 실험

| 주제 | 문서 |
|---|---|
| 구버전 7B/14B channel-matrix | [Channel-matrix README](../experiments/channel_matrix/README.md) |

진단 실험은 최신 PDF paper target evidence가 아니다.

## 보존 기록

`docs/data/*/README.md`와 `prereg/AMENDMENT-*.md`는 중복 가이드가 아니라
실험 provenance와 사전등록 변경 기록이므로 유지한다.

새 문서를 만들기 전에 이 인덱스의 기존 문서를 갱신한다. 날짜별 계획,
세션 메모, 완료된 구현 prompt를 새 Markdown 파일로 추가하지 않는다.
