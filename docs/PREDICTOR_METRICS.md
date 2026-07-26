# Predictor metric guide

이 문서는 현재 `paper/`의 prospective-rank, fidelity, harmful-tail 표에서
predictor를 평가하는 메트릭만 따로 설명한다. 전체 evidence-table 정의는
[`TABLE12_METRICS.md`](TABLE12_METRICS.md)를 따른다.

## 1. Predictor가 맞혀야 하는 것

한 candidate retain behavior를 `x`, SFT가 끝난 초기 checkpoint를
`theta_0`, parent unlearning의 first-reaching checkpoint를 `theta_p`라 한다.
실제 damage는 다음과 같다.

```text
d_p(x) = NLL(x; theta_p) - NLL(x; theta_0)
```

- `d_p(x) > 0`: unlearning 이후 NLL이 증가해 retain behavior가 손상됨
- `d_p(x) = 0`: 변화 없음
- `d_p(x) < 0`: NLL 기준으로는 오히려 개선됨

Predictor는 unlearning을 실행하기 전에 각 candidate의 향후 damage 순위를
맞히는 score다.

```text
S0(x) = normalized loss-shake susceptibility rank
S1(x) = normalized request-proximity rank
S_alpha(x) = (1 - alpha_pred) S0(x) + alpha_pred S1(x)
```

`alpha_pred`는 development fold에서 미리 선택해 동결한다. Target 결과를 본
뒤 alpha를 다시 고르면 predictor 검증으로 인정하지 않는다.

## 2. 무엇이 좋은 결과인가

요약하면 **점 추정치보다 lower bound가 기준을 넘는지가 중요하다.**

| 메트릭 | 좋은 방향 | claim 통과 기준 | 의미 |
|---|---:|---:|---|
| `Joint rho` | 클수록 좋음 | `LB(rho_S) > 0` | 결합 score가 실제 damage 순위를 예측 |
| `g_G` | 클수록 좋음 | `LB(g_G) > 0` | 결합 score가 loss-shake 단독보다 우수 |
| `g_H` | 클수록 좋음 | `LB(g_H) > 0` | 결합 score가 proximity 단독보다 우수 |
| `f_rho` | 1에 가까울수록 좋음 | `LB(f_rho - 0.80) > 0` | loss-shake와 exact gradient energy의 순위 일치 |
| `f_K` | 1에 가까울수록 좋음 | `LB(f_K - 0.70) > 0` | 두 방식의 Top-K 선택 일치 |
| `g_ctl` | 클수록 좋음 | `LB(g_ctl) > 0` | 결합 score가 strongest simple control보다 우수 |
| `L_tail` | 클수록 좋음 | `LB(L_tail) > 0` | 고손상 tail을 무작위보다 잘 찾음 |
| `eligible n/N` | 1에 가까울수록 좋음 | `n/N >= 0.80` | tail metric을 계산할 수 있는 unit coverage |

`LB`는 one-sided 95% bootstrap lower bound다. 예를 들어
`rho_S=0.30 [LB=-0.02]`는 점 추정치는 양수지만 불확실성을 고려하면
`0`보다 크다고 결론 낼 수 없으므로 pass가 아니다.

## 3. RQ1: 실제 damage 예측력

### Joint rho

```text
rho_S = Spearman(S_alpha(x), d_p(x))
```

범위는 `[-1, 1]`이다.

- `rho_S > 0`: score가 높은 candidate일수록 실제 damage도 큰 경향
- `rho_S = 0`: 순위 예측력이 없음
- `rho_S < 0`: 위험 순위를 반대로 예측

큰 양수가 좋지만 논문 claim의 명시적 기준은 효과 크기 자체가 아니라
`LB(rho_S) > 0`이다.

### g_G와 g_H

```text
g_G = rho_S - Spearman(S0, damage)
g_H = rho_S - Spearman(S1, damage)
```

두 값은 결합 predictor가 각 단독 component보다 얻는 Spearman 이득이다.

- `g_G > 0`: 결합 score가 loss-shake 단독보다 좋음
- `g_H > 0`: 결합 score가 proximity 단독보다 좋음
- 둘 중 하나가 `<= 0`: 결합이 두 component 모두를 개선했다는 주장은 불가

Table 1의 `min(g_G, g_H) [min LB]`는 두 값 중 약한 쪽을 표시한다.

```text
display estimate = min(estimate(g_G), estimate(g_H))
display LB       = min(LB(g_G), LB(g_H))
```

따라서 이 열의 lower bound가 양수면 두 gain lower bound가 모두 양수다.

### L_tail

전체 candidate 수를 `N_c`, 동결한 tail 크기를 `M`이라 한다.

```text
H_M = positive-damage candidate 중 실제 damage 상위 M개
P_M = predictor score 상위 M개
Recall_M = |H_M intersect P_M| / M
q = M / N_c
L_tail = Recall_M / q - 1
```

- `L_tail = 0`: 무작위 순위와 같은 기대 성능
- `L_tail > 0`: 무작위보다 고손상 tail을 잘 찾음
- `L_tail < 0`: 무작위보다 나쁨
- 최댓값: `1/q - 1`

`q`가 다르면 `L_tail`의 척도도 달라지므로 `M`과 candidate universe를
사전에 동결해야 한다. RQ1에서는 `LB(L_tail) > 0`을 요구한다.

### eligible n/N

```text
n = L_tail을 계산할 수 있었던 unit 수
N = reached + valid unit 수
coverage = n / N
```

한 unit에 positive-damage candidate가 최소 `M`개 있어야 `L_tail`을 계산할
수 있다. `eligible n/N`은 예측 성능이 아니라 **tail evidence coverage**다.

- `10/10`: 모든 reached+valid unit에서 tail 계산 가능
- `8/10`: coverage `0.80`, RQ1 최소 기준 충족
- `7/10`: coverage `0.70`, 다른 metric이 좋아도 RQ1 ineligible

수식의 candidate 수와 혼동을 피하려고 여기서는 candidate 수를 `N_c`,
coverage 분모를 `N`으로 구분했다.

## 4. RQ2: loss-shake 충실도와 추가 가치

### f_rho

```text
f_rho = Spearman(q_G_loss-shake, q_G_exact)
```

Loss-shake susceptibility rank가 동일 parameter block에서 계산한 exact
per-candidate gradient-energy rank와 얼마나 일치하는지 측정한다.

```text
통과: LB(f_rho - 0.80) > 0
동치: absolute LB(f_rho) > 0.80
```

### f_K

```text
f_K = |TopK(loss-shake) intersect TopK(exact)| / K
```

범위는 `[0, 1]`이며 실제 보호 대상으로 쓰는 상위 `K`개가 얼마나 겹치는지
측정한다.

```text
통과: LB(f_K - 0.70) > 0
동치: absolute LB(f_K) > 0.70
```

Table 1의 점 추정치는 absolute `f_rho/f_K`이고, 현재 evidence renderer의
대괄호는 각 threshold를 뺀 margin lower bound다. 예를 들어:

```text
0.91/0.78 [+0.05/+0.03]
```

이는 `f_rho=0.91`, `f_K=0.78`, absolute lower bound가 각각 `0.85`,
`0.73`이라는 뜻이다.

### g_ctl

```text
g_ctl = rho_S - Spearman(b_star, damage)
```

`b_star`는 development fold에서 선택하고 동결한 strongest predeclared
simple control이다.

- `LB(g_ctl) > 0`: 결합 predictor가 가장 강한 단순 control보다도 나음
- `LB(g_ctl) <= 0`: predictor의 이득을 단순 heuristic으로 설명할 가능성을
  배제하지 못함

RQ2는 `g_H`도 다시 사용한다. 이는 fidelity가 높은 loss-shake가 단순히
request proximity와 같은 정보가 아니라 추가 predictive value를 주는지
검사하기 위해서다.

## 5. 최종 판정 읽는 법

### Rank E/P

`Rank E/P`는 순위 번호가 아니라 rank-condition의 `eligible/pass`다.

```text
Rank pass =
    eligible
    and LB(rho_S) > 0
    and LB(g_G) > 0
    and LB(g_H) > 0
    and LB(g_ctl) > 0
    and four-way IUT p <= 0.05
```

### RQ1 E/P

`E/P`는 `eligible/pass`다.

```text
RQ1 pass =
    Rank pass
    and LB(L_tail) > 0
    and five-way IUT p <= 0.05
```

Eligibility에는 valid selection/profile, common support, external gate reach,
`eligible n/N >= 0.80`이 포함된다.

따라서 RQ1 성공을 주장하려면 `Rank E/P`와 `RQ1 E/P`가 모두 `y/y`여야
한다. Rank `y/y`, RQ1 `y/n`은 순위 조건은 통과했지만 harmful-tail 조건이
실패했다는 뜻이다.

### RQ2 E/P

```text
RQ2 pass =
    eligible
    and LB(f_rho - 0.80) > 0
    and LB(f_K - 0.70) > 0
    and LB(g_H) > 0
    and LB(g_ctl) > 0
    and four-way IUT p <= 0.05
```

Eligibility에는 paired fidelity/common-control support와 perturbation,
exact-reference, control validity가 포함된다.

| 표기 | 해석 |
|---|---|
| `y/y` | evidence가 유효하고 해당 통계 기준도 통과 |
| `y/n` | 계산 자격은 있지만 하나 이상의 bound 또는 IUT가 실패 |
| `n/--` | coverage, validity, support 또는 완료 조건이 부족해 판정하지 않음 |

좋은 predictor의 최종 형태는 Table 1 Panel A에서 **RQ1 `Y/Y`와 RQ2
`Y/Y`를 동시에 얻는 것**이다. `Joint rho` 하나만 높거나 fidelity만 높은
것으로는 충분하지 않다.

## 6. 구현 위치

| 역할 | 코드 |
|---|---|
| Spearman, gain, tail 계산 | `src/rsus/evidence/raw.py::_prediction_metrics` |
| RQ2 fidelity/add-value 구성 | `src/rsus/evidence/raw.py::_rq2_metrics` |
| RQ1/RQ2 eligibility와 IUT | `src/rsus/evidence/pdf_v4.py` |
| Table 1 Panel A 렌더링 | `src/rsus/evidence/tables.py` |

내부적으로 `top_q_recall`도 계산하지만 최신 PDF의 RQ1/RQ2 IUT 구성
메트릭은 아니다. Claim 판정은 위에 명시한 RQ1 네 효과와 RQ2 네 효과만
사용한다.
