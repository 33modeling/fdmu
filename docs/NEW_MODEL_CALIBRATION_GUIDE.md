# New model calibration

새 LLM을 추가할 때는 기존 모델의 숫자를 복사하지 않고 architecture, dtype,
data roster, parent, predictor, repair를 target-free fold에서 다시 검증한다.
Machine-readable handoff는
`configs/paper/new_model_calibration.template.yaml`에 기록한다.

## 실행 순서

```text
environment/model contract
  -> architecture/block validation
  -> dataset roster and manifests
  -> D_cal: loss-shake + parent calibration
  -> D_pred: alpha_pred
  -> D_prot: alpha_prot, Kp, repair
  -> human-reviewed immutable freeze
  -> target execution
  -> evidence ledger and LaTeX
```

앞 단계가 unresolved이면 target 값으로 대체하지 않는다.

## 1. 환경 기록

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,campaign]"
.venv/bin/python -m pytest -q
```

다음을 run manifest에 남긴다.

- Git commit과 clean-worktree 상태
- Python, PyTorch, Transformers, CUDA, driver, GPU
- Model/tokenizer source revision 또는 content hash
- Dtype, attention implementation, deterministic settings, device map
- Dataset revision/cache fingerprint와 모든 manifest hash

Confirmatory profiling 기본 dtype은 fp32다. 메모리 때문에 bf16으로 바꾸는
것은 별도 fidelity를 통과하기 전까지 diagnostic이다.

## 2. Architecture와 block

현재 helper는 Qwen/Llama/Mistral 계열의 다음 파라미터 형태를 지원한다.

```text
*.layers.<index>.mlp.down_proj.weight
```

새 구조는 model-specific block selector와 테스트를 먼저 추가한다.

1. 선택된 parameter name/count가 의도한 block과 일치한다.
2. `+eta*v -> -eta*v -> restore` 뒤 parameter가 bit-exact하다.
3. `v`는 fp32 unit norm이고 block 밖 좌표는 변하지 않는다.
4. Central difference와 exact directional derivative가 작은 subset에서 맞는다.
5. Exact energy와 loss-shake가 loss, mask, batch, dropout, block을 공유한다.
6. Hidden-state pooling과 answer-token 위치가 tokenizer 계약과 맞는다.

이 검사가 끝나기 전에 sweep을 시작하지 않는다.

## 3. Dataset 계약

각 request마다 score 계산 전에 immutable candidate manifest를 만든다.

- `Df`, candidate universe, semantic group, discovery/audit IDs
- repair eligibility와 exclusion reason
- neutral, guard, utility, native-audit IDs
- repeated-random draw IDs와 seed
- tokenizer/truncation 결과와 answer-token 유효성

Request roster는 서로 겹치지 않는 네 집합이어야 한다.

| Fold | 선택 가능한 값 |
|---|---|
| `D_cal` | parent, loss-shake `R/eta`, numerical integrity |
| `D_pred` | `alpha_pred`, simple control |
| `D_prot` | `alpha_prot`, `Kp`, repair/guard budget |
| target | 선택 없음, frozen contract 실행만 허용 |

Dataset adapter 추가 절차와 fail-closed 조건은
[paper preflight](PAPER_CAMPAIGN_PREFLIGHT.md)를 따른다.

## 4. D_cal

### Loss-shake

각 `(R, eta, block)` 후보에서 다음을 저장한다.

- Effective displacement와 changed-coordinate fraction
- Signed central response와 squared energy estimate
- Exact gradient energy 대비 Spearman과 Top-`Kp` overlap
- Direction-split stability, wall time, peak memory

논문 fidelity 기준:

```text
LB(f_rho - 0.80) > 0
LB(f_K   - 0.70) > 0
```

과거 모델의 `R`, `eta`, block depth는 새 모델의 기본값이 아니다.

### Parent

각 parent는 profile score를 보지 않고 direct-forgetting과 ordinary utility로
calibrate한다. Objective, trainable support, optimizer, learning rate,
processed-token budget, seed, checkpoint grid를 동결한다. Target에서 도달하지
않는 parent는 non-reaching으로 기록하며 budget을 늘리지 않는다.

## 5. D_pred와 D_prot

`D_pred`에서는 다음 결합 score의 `alpha_pred`와 simple control만 선택한다.

```text
S_alpha = (1 - alpha) * rank(loss-shake)
          + alpha * rank(request-proximity)
```

`D_prot`에서는 별도로 `alpha_prot`, `Kp`, repair와 guard 설정을 선택한다.
모든 arm은 같은 first-reaching parent, support, neutral/guard stream,
processed-token budget, seed, checkpoint schedule을 사용한다.

Repair objective와 projection은 최신 PDF Eq. (7)--(8) 구현인
`src/rsus/repair.py`와 `src/rsus/generators/repaired.py`를 사용한다.
`src/rsus/stage2.py`는 구버전 diagnostic이며 v4 parameter를 번역해 넣지 않는다.

## 6. Freeze와 preflight

Freeze에는 environment, model, tokenizer, theta0, roster, manifest, block,
probe, parent, repair, metric, bootstrap/IUT contract의 hash와 resolved value가
모두 있어야 한다. Agent는 proposal을 만들 수 있지만 `status: frozen` 승인은
사람이 target 전에 검토하고 commit한다.

```bash
python experiments/paper/preflight.py
```

- Exit `0`: 현재 frozen contract 실행 가능
- Exit `2`: unresolved blocker가 있어 실행 금지
- Exit `1`: malformed contract

## 7. Target과 evidence

Target unit은 다음 순서를 지킨다.

1. `theta0`에서 profile을 만들고 audit score를 seal한다.
2. Parent의 first-reaching checkpoint를 선택한다.
3. 모든 repair/comparator arm을 같은 checkpoint에서 분기한다.
4. Hard margin, retry, rollback, processed-token budget을 기록한다.
5. Full feasibility를 만족하는 마지막 저장 checkpoint만 선택한다.
6. 그 뒤에만 sealed outcome을 열고 RQ1/RQ2/RQ3를 집계한다.

메트릭 수식과 통과 기준은 [Table 1/2 metrics](TABLE12_METRICS.md), 원자료에서
LaTeX까지의 경로는 [paper evidence pipeline](PAPER_EVIDENCE_PIPELINE.md)을
따른다.

최종 검증:

```bash
python experiments/paper/build_evidence.py \
  --config configs/paper/evidence.yaml \
  --ledger results/paper/evidence_ledger.json \
  --paper-root paper \
  --require-ready
```

## 완료 체크리스트

- Block selector와 tokenizer masking 테스트가 통과했는가?
- `theta0` construction이 명시적이고 sealed outcome과 독립적인가?
- 네 request roster와 내부 semantic split이 유효한가?
- Loss-shake가 이 model/block/dtype에서 fidelity를 통과했는가?
- Parent, `alpha_pred`, `alpha_prot`, repair가 각자 허용된 fold에서 동결됐는가?
- 모든 target row가 하나의 immutable manifest와 clean commit으로 재현되는가?
- Ledger가 세 RQ를 따로 판정하고 missing support에서 fail closed하는가?

하나라도 아니면 해당 모델 결과는 diagnostic으로 남기고 paper table에 합치지
않는다.
