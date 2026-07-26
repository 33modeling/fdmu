# Current paper evidence-table metric guide

## 0. 기준 문서

현재 기준은 추적 중인 `paper/` 소스, `configs/paper/evidence.yaml`, 그리고
`src/rsus/evidence/tables.py`다. 루트의 `KDD_UnlearningFail.pdf`는 현재
paper 소스보다 오래된 스냅샷이므로 roster나 표 배치의 기준으로 사용하지
않는다.

현재 core 표 구조는 다음과 같다.

- **Table I:** prospective rank correlation, endpoint/control gains, `Rank E/P`
- **Table II:** Loss Susceptibility fidelity/cost와 no-refit swap diagnostic
- **Table III:** harmful-tail recovery와 최종 `RQ1 E/P`
- **Table IV:** fixed-budget repair damage effects와 `Effect E/P`
- **Table V:** feasibility/native-retention contract와 최종 `RQ3 E/P`
- **Robustness:** claim breadth와 evidence funnel 두 표

Core table의 parent roster는 output-readout
`{GradDiff, NPO, SimNPO, GRU}`와 representation-readout
`{RMU, RepNoise, CB}`다.

Robustness setting roster는 정확히 9행이다.

| Axis | Current setting |
|---|---|
| Request | held-out TOFU requests (Qwen2.5-7B primary) |
| Dataset | WMDP-bio/MMLU |
| Dataset | MUSE-News |
| Dataset | RWKU |
| Dataset | MUSE-Books (stress) |
| Dataset | PISTOL (stress) |
| Model | Qwen2.5-1.5B (boundary) |
| Model | Qwen2.5-14B |
| Model | Llama-3.1-8B |

현재 primary는 `TOFU / Qwen2.5-7B`, 1.5B는 scale boundary, 14B는
model-scale, Llama-3.1-8B는 model-family setting이다.

계산과 렌더링이 구현되어 있다는 것과 실제 target campaign이 완료되어 숫자가
채워졌다는 것은 다르다. 원자료가 없거나 eligibility가 불완전한 셀은
placeholder로 남는다.

## 1. 공통 표기

한 실험 unit은 `(setting, parent, request, seed)`이다. 후보 행동을 `x`,
초기 SFT checkpoint를 `theta_0`, arm `a`의 평가 checkpoint를 `theta_a`라
하면 audit damage는 다음과 같다.

```text
d_a(x) = NLL(x; theta_a) - NLL(x; theta_0)
```

`d_a(x) > 0`은 해당 retain 행동의 NLL이 증가해 손상이 생겼다는 뜻이다.
따라서 damage의 mean과 CVaR는 작을수록 좋다.

두 기본 score의 정규화 rank와 결합 score는 다음과 같다.

```text
S0(x) = q_G_tilde(x)  # normalized loss-shake susceptibility rank
S1(x) = q_H_tilde(x)  # normalized request-proximity rank
S_alpha(x) = (1 - alpha) S0(x) + alpha S1(x)
```

`alpha_pred`와 `alpha_prot`는 서로 다른 development fold에서 target 결과를
보지 않고 동결한다. RQ1/RQ2는 `S_(alpha_pred)`, RQ3는 별도로 동결한
`S_(alpha_prot)`를 사용한다.

| arm | 의미 |
|---|---|
| `joint` | 동결된 `S_(alpha_prot)`로 보호 대상을 선택해 repair |
| `no_repair` | 동일한 first-reaching parent checkpoint, repair 없음 |
| `repeated_random` | 동결된 random draw roster로 보호 대상을 반복 선택 |
| `s0` | `q_G`만 사용해 repair |
| `s1` | `q_H`만 사용해 repair |

## 2. 집계와 불확실성

점 추정치는 unit 안에서 candidate-level metric을 계산한 뒤 같은 request의
seed들을 동일 가중치로 평균하고, 마지막으로 request들을 동일 가중치로
평균한다.

신뢰 한계는 request, seed, semantic candidate group 순서의 hierarchical
bootstrap으로 구한다. 기본 반복 수는 `B=2000`, 유의수준은 `alpha=0.05`다.
동점 Spearman rank는 midrank를 사용하고, Top-K 동점은 `candidate_id`로
결정적으로 푼다.

```text
LB_0.95(z)  = bootstrap 5th percentile
UCB_0.95(z) = bootstrap 95th percentile

p_positive = (1 + count(z_boot <= 0)) / (B + 1)
p_negative = (1 + count(z_boot >= 0)) / (B + 1)
p_IUT      = max(p_1, ..., p_m)
```

큰 값이 좋은 효과에는 one-sided lower bound, 작은 값이 좋은 damage
차이에는 one-sided upper bound를 사용한다. IUT(intersection-union test)는
모든 구성 효과가 동시에 통과해야 한다.

## 3. Core Tables I--III: RQ1/RQ2

Predictor 메트릭만 빠르게 확인하려면
[`PREDICTOR_METRICS.md`](PREDICTOR_METRICS.md)를 본다. 각 값의 좋은 방향,
lower-bound 통과 기준과 `E/P` 조합을 함께 정리했다.

### Joint rho [LB]

결합 score와 이후 발생한 audit damage의 Spearman 상관이다.

```text
rho_S = Spearman(S_(alpha_pred)(x), d_parent(x))
```

범위는 `[-1, 1]`이다. 양수이면 높은 사전 score가 실제 고손상 후보를 먼저
배치했다는 뜻이다. RQ1에서는 `LB > 0`을 요구한다.

### min(g_G, g_H) [min LB]

결합 score가 두 단독 score보다 얼마나 나은지 측정한다.

```text
g_G = rho_S - Spearman(S0, damage)
g_H = rho_S - Spearman(S1, damage)
```

표에는 두 점 추정치의 최솟값과 두 lower bound의 최솟값을 표시한다. 둘 다
양수여야 결합 score가 `q_G`, `q_H` 각각보다 낫다고 말할 수 있다.

### f_rho / f_K [LB]

Loss-shake가 동일 parameter block의 exact per-candidate gradient energy를
얼마나 잘 재현하는지 측정한다.

```text
f_rho = Spearman(q_G_hat, q_G_exact)
f_K   = |TopK(loss_shake) intersect TopK(exact)| / K

margin_rho = f_rho - 0.80
margin_K   = f_K - 0.70
```

`K`는 동결된 `Kp`다. PDF 표 머리글은 `[LB]`로 축약하지만 Section 4.7의
판정 대상은 `LB(f_rho-0.80)`과 `LB(f_K-0.70)`이다. 즉 단순히
`LB(f_rho)>0`, `LB(f_K)>0`를 보는 것이 아니다.

### g_ctl [LB]

결합 score가 `D_pred`에서 미리 선택한 strongest simple control보다 얻는
Spearman 이득이다.

```text
g_ctl = rho_S - Spearman(b_star, damage)
```

`b_star`는 development fold에서 고른 strongest predeclared simple control이다.
`LB > 0`이면 결합 score가 이 control보다 유의하게 낫다.

### L_tail [LB]; eligible n/N

전체 후보 수를 `N`, 미리 동결한 tail 크기를 `M`이라 한다. 양의 damage
후보가 최소 `M`개인 unit에서만 다음 값을 계산한다.

```text
H_M = positive-damage candidates 중 damage 상위 M개
P_M = S_(alpha_pred) 상위 M개
Recall_M = |H_M intersect P_M| / M
q = M / N
L_tail = Recall_M / q - 1
```

무작위 순위의 기대값은 `L_tail=0`이다. `eligible n/N`은 tail metric을
계산할 수 있었던 unit 수와 reached+valid unit 수다. RQ1 eligibility에는
이 비율이 최소 `0.80`이어야 한다.

### RQ1 E/P

`E/P`는 eligible/pass다.

- `E=y`: prediction selection, profile integrity, external gate reach,
  shared support가 유효하고 tail coverage가 `0.80` 이상이다.
- `Rank E/P`의 `P=y`: `rho_S`, `g_G`, `g_H`, `g_ctl`의 네 lower
  bound가 모두 `0`보다 크고 four-way IUT가 `alpha=0.05`를 통과한다.
- 최종 `RQ1 E/P`의 `P=y`: rank 조건과 `L_tail` lower bound가 모두
  양수이고 five-way IUT를 통과한다.
- 모든 planned trajectory가 완료되지 않으면 pass가 차단된다.

RQ1 성공에는 `Rank E/P=y/y`와 `RQ1 E/P=y/y`가 모두 필요하다.
`Rank=y/y`, `RQ1=y/n`은 harmful-tail 조건 실패를 뜻한다.

### RQ2 E/P

- `E=y`: paired fidelity/common-control support가 2 unit 이상이고,
  perturbation, exact-reference, common-control validity가 모두 참이다.
- `P=y`: `f_rho-0.80`, `f_K-0.70`, `g_H`, `g_ctl`의 네 lower bound가
  모두 `0`보다 크고 four-way IUT를 통과한다.

`g_H`는 RQ1에서는 두 endpoint 중 하나이고, RQ2에서는 proximity만으로는
설명되지 않는 loss-shake의 added value를 검사한다.

## 4. Core Tables IV--V: RQ3

### Profile mean; CVaR

`joint` repair arm의 sealed-audit damage 평균과 upper-tail CVaR다.

```text
mean(d) = sum_i d_i / N
```

CVaR.95는 가장 큰 damage 5%의 fractional empirical mean이다. Damage를
내림차순 `d_(1) >= ... >= d_(N)`, `m=0.05N`, `k=floor(m)`,
`gamma=m-k`라 하면:

```text
CVaR.95 = (sum_{i=1..k} d_(i) + gamma * d_(k+1)) / m
```

`m < 1`이면 표본 최댓값을 사용하고 ledger의 `low_tail_support`를 켠다.
Mean과 CVaR 모두 작을수록 좋다.

### No-repair mean; CVaR

동일한 first-reaching parent checkpoint에서 repair하지 않은 arm의 절대
mean/CVaR다. Profile arm의 효과를 읽기 위한 기준값이며 RQ3의 네 comparator
중 하나이기도 하다.

### max_(a,k) Delta_(a,k) [UCB]

Comparator `a`와 outcome `k`에 대한 paired damage 차이다.

```text
a in {no_repair, repeated_random, s0, s1}
k in {mean, CVaR.95}
Delta_(a,k) = T_k(joint) - T_k(a)
```

음수이면 joint가 comparator보다 damage가 작다. 표에는 8개 차이 중 가장 큰
점 추정치와 가장 큰 upper bound를 표시한다. RQ3 superiority에는 8개 UCB가
모두 `0`보다 작아야 한다.

### min_a h_a [LB]

Dataset-native retain metric의 comparator별 non-inferiority margin이다.
PDF는 native metric `M_a`를 미리 favorable orientation으로 바꾼 뒤,
동결된 허용 열화량 `delta_nat >= 0`을 더한다.

```text
h_a = M_profile - M_a + delta_nat
```

표에는 네 comparator 중 가장 작은 점 추정치와 lower bound를 표시한다.
모든 `LB(h_a) > 0`이면 joint가 각 comparator보다 허용 margin을 넘게
열등하지 않다.

### min F/U slack

Forgetting audit은 낮을수록 좋고 utility는 높을수록 좋다.

```text
slack_direct  = threshold_direct - recall_direct
slack_para    = threshold_para - recall_para
slack_extract = threshold_extract - recall_extract
slack_utility = utility - required_utility_floor

F = min(slack_direct, slack_para, slack_extract)
U = slack_utility
```

표의 `min F/U`는 모든 claim arm, repeated-random draw, request, seed에서
각각 가장 작은 slack이다. 하나라도 음수이면 RQ3 eligibility가 차단된다.

### updates/rollback

Joint repair arm에서 accept된 update 수와 guard 위반으로 되돌린 update 수의
request-balanced 평균이다. 최적화 진단이며 RQ3 IUT의 구성 효과는 아니다.

### RQ3 E/P

- `E=y`: protection selection이 valid/non-fallback이고, 다섯 arm과 모든
  random draw가 완료됐으며, 동일 candidate support를 공유하고, 모든 arm의
  forgetting/utility margin이 0 이상이고, support unit이 2개 이상이다.
- `P=y`: 8개 damage UCB가 모두 `< 0`, 4개 native non-inferiority LB가
  모두 `> 0`, twelve-way IUT가 `alpha=0.05`를 통과한다.
- 모든 planned trajectory 완료도 별도로 필요하다.

## 5. Robustness와 failure boundary

Table 2는 새로운 candidate metric을 계산하지 않는다. Table 1과 같은
decision을 위에 열거한 PDF의 8개 predeclared setting에 대해 요약하며,
미실행 setting도 분모에서 제거하지 않는다.

### Claim breadth

| 열 | 의미 |
|---|---|
| `Axis` | request, model, dataset 중 변화 축 |
| `Setting` | dataset/model setting; stress와 boundary는 표시 |
| `Plan/done` | 계획된 parent row 수 / 완전히 완료된 parent row 수 |
| `RQ1 E/P` | 해당 setting에서 RQ1 eligible parent 수 / passed parent 수 |
| `RQ2 E/P` | 해당 setting에서 RQ2 eligible parent 수 / passed parent 수 |
| `RQ3 E/P` | 해당 setting에서 RQ3 eligible parent 수 / passed parent 수 |
| `Chain` | output-readout와 representation-readout group에서 각각 최소 1개 parent가 within-readout Bonferroni 보정 후 RQ1/RQ2/RQ3를 모두 통과했는지 (`y/n`) |

아직 시도하지 않은 setting의 `Chain`은 실패를 뜻하는 `n`이 아니라
`\tblph`로 남긴다. Stress setting은 primary 실패를 구제하지 않는다.

### Evidence funnel

| 열 | 의미 |
|---|---|
| `Axis` | request, model, dataset 중 변화 축 |
| `Setting` | dataset/model setting; stress와 boundary는 표시 |
| `Profiles valid` | sealed profile valid 수 / planned profile 수 |
| `Gate reached` | forgetting gate 도달 trajectory 수 / planned trajectory 수 |
| `Common n` | prediction에서 complete common support를 가진 audit unit 수 |
| `Tail-elig. n` | RQ1 harmful-tail 계산 자격을 갖춘 unit 수 |
| `All-arm feas.` | 다섯 repair arm이 모두 feasible한 unit 수 / valid profile로 gate에 도달한 unit 수 |
| `Worst RQ1/RQ2/RQ3` | RQ1 LB 최솟값 / RQ2 LB 최솟값 / RQ3 damage UCB 최댓값 |
| `Failure modes` | non-reach, invalid profile, common-support, constraint/IUT 실패 요약 |

`worst` 열에서 RQ1·RQ2는 큰 값이 유리하므로 lower bound의 최솟값을,
RQ3 damage는 작은 값이 유리하므로 upper bound의 최댓값을 쓴다. Native
non-inferiority bound는 damage와 척도가 달라 Table 1의 `min_a h_a`에만
표시한다. 이 worst 값은 기술적 요약이며 누락 row를 대신하거나 pass를
허가하지 않는다.

Setting-level/transfer claim은 row별 IUT보다 더 엄격하다. `Chain`은 각
readout parent group 안에서 Bonferroni 보정한 alpha를 적용해
RQ1/RQ2/RQ3를 모두 통과한 parent가 있는지를 표시한다. Primary,
model-transfer, dataset-replication group의 사전 동결 규칙까지 만족해야
최종 transfer 문구가 licensed된다.

## 6. 구현 위치와 현재 상태

| 단계 | 구현 |
|---|---|
| Damage/fidelity/protection raw 생성 | `experiments/paper/tofu_v4_unit.py` |
| Spearman, tail lift, CVaR, effect 집계 | `src/rsus/evidence/raw.py`, `src/rsus/analysis/prediction.py` |
| Bootstrap bound와 p-value | `src/rsus/evidence/statistics.py` |
| RQ1/RQ2/RQ3 IUT와 eligibility | `src/rsus/evidence/pdf_v4.py`, `src/rsus/evidence/decisions.py` |
| 현재 paper 표 렌더링 | `src/rsus/evidence/tables.py`, `src/rsus/evidence/table1.py` |
| 현재 동결 config | `configs/paper/evidence.yaml`, `configs/paper/campaign.yaml` |

현재 renderer와 `paper/main.tex`은 같은 generated core/robustness 파일을
사용한다. 7B 완료 결과는 `render-only`로 현재 형식에 재집계할 수 있다.
나머지 setting은 실제 evidence가 없으면 고정된 9행 분모를 유지하면서
placeholder로 남는다. 결과 완성 여부는 `evidence_readiness.json`의
eligibility/pass와 `FINALIZATION_STATUS.json`을 기준으로 판단한다.
