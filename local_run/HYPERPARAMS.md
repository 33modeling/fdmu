# 실험 하이퍼파라미터 정리 (로컬 RTX 4090 캠페인)

`experiments/gate_1p5b/gate.py` 게이트 실험의 모델별·방법별 설정. 값 출처: 7B는
`configs/channel_matrix/objective_freeze.yaml`(동결됨), 1.5B는 그 값을 **레퍼런스로
전용**(1.5B freeze는 미보정=draft/null). repair 튜닝값 근거는 CLAUDE.md 캠페인 메모.

## 1. 모델별 하드웨어 / 정밀도

| 모델 | 경로(`/rdata/models/`) | Table 1(프로브) | Table 2(repair) |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct | `Qwen2.5-1.5B-Instruct` | fp32, 1-GPU | **fp32, 2-GPU (device_map balanced)** |
| Qwen2.5-3B-Instruct | `Qwen2.5-3B-Instruct` | fp32, 1-GPU | fp32 2-GPU (경계) / bf16 2-GPU |
| Qwen3-4B-Instruct-2507 | `Qwen3-4B-Instruct-2507` | fp32, 1-GPU | bf16, 2-GPU |
| Qwen2.5-7B-Instruct | `Qwen2.5-7B-Instruct` | bf16 1-GPU / **fp32 2-GPU (`--device-map split:8`)** | ❌ 불가 (H100 필요) |
| DeepSeek-R1-Distill-Qwen-1.5B | `DeepSeek-R1-Distill-Qwen-1.5B` | fp32, 1-GPU | fp32, 2-GPU |

> **정밀도 주의:** `fd` 프로브(eta=3e-4 유한차분)는 **fp32 필수** — bf16은 정밀도 바닥에
> 삼켜져 무너짐(7B bf16 fd AUROC 0.47 vs fp32 0.59). repair(stage1 full-model AdamW)는
> ≤4B만 2-GPU에 적재 가능, 7B는 H100(80GB) 필요.

## 2. 게이트 공통 설정

| 항목 | 값 |
|---|---|
| trainable-scope | Table 1: `probe_block` / **repair: `full`** (도달 위해 full-model 학습) |
| universe-authors (TOFU) | 20 · pool-size 16 · batch-size 4 · seed 2025 |
| RWKU | `--author 0 --candidate-authors 100-119` |
| MUSE | `--dataset muse_news\|muse_books` (knowmem forget_qa/retain_qa) |
| predictors | fd, fd_norm, knn_feature, knn_embed, knn_lexical, grad_norm, random_rank |
| gen-steps (repair) | 240 · beta 0.1 |

## 3. 방법별 목적함수 하이퍼파라미터 (7B 동결 → 1.5B 전용)

| objective | lr | steps | beta | 기타 | 채널 |
|---|---|---|---|---|---|
| ga | 1e-6 | 60 | — | — | loss_gradient |
| graddiff | 8e-7 | 240 | — | fw 1.0 rw 1.0 | loss_gradient |
| npo | **1.6e-5** | **240** | **0.1** | fw 1.0 rw 1.0 | loss_gradient |
| simnpo | 1.6e-5 | 240 | 4.5 | fw 0.125 rw 1.0 | loss_gradient |
| idkdpo | 4e-6 | 120 | 0.1 | — | loss_gradient |
| rmu | 3.2e-5 | 240 | — | rmu_alpha 10, rmu_c 3 | representation |
| gru | 2e-6 | 120 | — | fw 1.0 rw 1.0 | representation |
| repnoise | 1e-6 | 120 | — | rmu_alpha 1, rmu_c 3 | representation |
| circuit_breakers | 1e-6 | 240 | — | rmu_alpha 100 | representation |

> gate.py 적용: 생성기(Table 1)는 `--gen-lr-per`/`--gen-beta-per`/`--gen-steps-per`,
> T2(Table 2)는 `--t2-lr-per`(+전역 `--gen-steps`/`--beta`). **gate.py T2는 per-method
> LR만 지원**하고 beta/steps는 전역이라, simnpo(beta 4.5)는 근사(beta 0.1)로 실행됨.

## 4. Repair (Stage1 + Stage2 guarded) 튜닝

| knob | 기본값 | 사용값 | 근거 |
|---|---|---|---|
| `--s2-eta2` (Stage2 repair step) | **5e-3 (토이값→발산)** | **3e-5** | CLAUDE.md: "5e-3/1e-4 발산, 3e-5 최적". 5e-3 쓰면 ours dNLL 62 폭발 |
| `--s1-recall-gate` | 0.0 (floor-only) | **0.10** | T2 도달 기준(recall<0.10)까지 stage1이 잊게. floor-only면 ours reached=False |
| `--s1-lr` / `--s1-max-steps` | 1e-5 / 600 | 튜닝 중 | s1-recall-gate 0.10이 과붕괴 유발 시(min_forget≫floor) 낮춰 안정화 |
| `--s2-steps` | 80 | 튜닝 중 | 늘리면 복구↑ → collateral↓ |

## 5. Alpha 채널 혼합 (파티션)

`s_alpha = (1−α)·midrank(gradient) + α·midrank(proximity)` (`analysis/mixture.py`)
- gradient = `fd`, proximity = `knn_embed`
- α=0 gradient(=fd) / α=1 representation(RMU 정렬) / α=0.5 hybrid
- `--partition-alpha select`: **discovery(dev) fold에서 damage 예측 최적 α 선택**
  (grid 0/0.25/0.5/0.75/1.0), audit는 안 봄 (anti-post-hoc)
- 이전엔 `--partition-predictor fd` = α=0 고정이었음

## 6. 재현 명령 (TOFU 1.5B repair, 확정 설정)

```bash
python experiments/gate_1p5b/gate.py \
  --model /rdata/models/Qwen2.5-1.5B-Instruct --model-id 1p5b_repair \
  --device cuda --dtype float32 --device-map balanced \
  --trainable-scope full --dataset tofu --universe-authors 20 --pool-size 16 --batch-size 4 --seed 2025 \
  --predictors fd,fd_norm,knn_feature,knn_embed,knn_lexical,grad_norm,random_rank \
  --sentence-encoder /rdata/models/all-MiniLM-L6-v2 \
  --generators npo,graddiff,rmu --gen-steps 240 --beta 0.1 \
  --gen-lr-per "npo=1.6e-5,graddiff=8e-7,rmu=3.2e-5" --gen-beta-per "npo=0.1" \
  --t2-roster "ga,graddiff,npo,simnpo,idkdpo,rmu,gru,s2s,npo_transplant,ours" \
  --t2-lr-per "ga=1e-6,graddiff=8e-7,npo=1.6e-5,simnpo=1.6e-5,idkdpo=4e-6,rmu=3.2e-5,gru=2e-6" \
  --s2-eta2 3e-5 --s1-recall-gate 0.10 \
  --partition-predictor fd --partition-proximity knn_embed --partition-alpha select
```

## 7. 미해결 / 튜닝 진행 중
- 1.5B objective freeze 미보정 → 7B 값 전용 중 (스케일 전이 근사). 제대로 하려면
  dev-pool 보정(`select_freeze.py`).
- npo 계열 도달: beta 0.1 + lr 1.6e-5 + steps 240으로 개선 중이나 gen-lr/steps 추가 조정 여지.
- ours collateral 최소화: s1-lr / s1-max-steps / s2-steps / s2-eta2 5개 값 sweep 예정.

## 8. 튜닝 변경 이력 (왜 바꿨나 → 효과)

날짜 2026-07-24, TOFU 1.5B repair 디버깅 순서:

| # | 파라미터 | 변경 | 이유 / 증상 | 효과 |
|---|---|---|---|---|
| 1 | `--s2-eta2` | 5e-3(기본) → **3e-5** | ours가 dNLL **62 폭발**(Stage2 발산); 기본 5e-3은 토이 이식값 | 발산 해소 |
| 2 | `--trainable-scope` | probe_block → **full** | probe_block은 언러닝이 약해 NPO 계열 전부 미도달(recall 0.6~0.7) | graddiff/rmu 도달 |
| 3 | `--beta` | 1.0(기본) → **0.1** | beta 1.0이 NPO 그래디언트 소멸(코드 주석 명시) | simnpo 도달, npo 0.754→0.41 |
| 4 | gru 코드 | projection coef를 host float로 | 멀티-GPU에서 cuda:0/cuda:1 텐서 혼합 RuntimeError | gru 실행됨 |
| 5 | gen/t2 lr | 단일 1e-5 → **per-method**(7B 동결값) + `--gen-steps 240` | 단일 lr은 ga/gru 폭발(81/14.8)시키면서 npo는 미도달 | 폭발 억제 + npo steps 확보 |
| 6 | 파티션 | `fd`(α=0 고정) → **`--partition-alpha select`** | 채널 혼합 없이 gradient 끝점 고정이었음 | fd+knn_embed 혼합, dev로 α 선택 |
| 7 | `--s1-recall-gate` | 0.0(floor-only) → **0.10** | ours/s2s가 floor까지만 잊어 T2 기준(recall<0.10) 미도달 | ours/s2s 도달 (단 과붕괴 주의) |

> 다음 값이 확정되면 이 표에 계속 append. 최종 확정 시 §9로 승격.

## 9. 모델별 최종 확정값 (FROZEN)

각 모델을 튜닝해 확정되면 여기에 기록. `status`: draft(튜닝중) → frozen(확정).
7B는 objective_freeze.yaml에서 동결됨; 나머지는 로컬 보정 필요.

| 모델 | 데이터셋 | scope | dtype/GPU | s2-eta2 | s1-recall-gate | α(파티션) | status | 비고 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | TOFU | full | fp32/2-GPU | 3e-5 | 0.10 | select(진행중) | **draft** | 5개 값 sweep 예정 |
| Qwen2.5-1.5B | RWKU | — | fp32/2-GPU | — | — | — | pending | |
| Qwen2.5-1.5B | MUSE-books/news | — | fp32/2-GPU | — | — | — | pending | |
| Qwen2.5-3B | TOFU/RWKU/MUSE | — | bf16/2-GPU | — | — | — | pending | fp32 2-GPU는 경계 |
| Qwen3-4B | TOFU/RWKU/MUSE | — | bf16/2-GPU | — | — | — | pending | |
| DeepSeek-R1-1.5B | TOFU/RWKU/MUSE | — | fp32/2-GPU | — | — | — | pending | |
| Qwen2.5-7B | 전부 | — | — | — | — | — | ❌ repair 불가 | H100 필요 |

> 방법별 lr/beta/steps는 §3 표 참조 (모델별로 재보정 시 이 섹션 아래에 모델별 블록 추가).
