# Latest PDF Table 1/2 metric guide

## 0. 기준 문서

이 문서의 유일한 기준은 저장소 루트의 `KDD_UnlearningFail.pdf`다.

```text
PDF creation: 2026-07-24 19:48 KST
Pages: 13
SHA-256: b0ea5de888e4e5b3e429ce57f5bebd6a6cb18f36306422bcbafb963d02d93209
```

PDF와 Markdown, LaTeX, YAML, 코드가 충돌하면 PDF 정의를 따른다. 특히
`paper/sections/05_experiments.tex`의 predictor-by-objective channel matrix는
이전 초안의 Table 1이며 최신 PDF Table 1이 아니다.

최신 PDF의 표 구조는 다음과 같다.

- **Table 1, Panel A:** `Joint rho [LB]`,
  `min(g_G,g_H) [min LB]`, `f_rho/f_K [LB]`, `g_ctl [LB]`,
  `L_tail [LB]; eligible n/N`, `RQ1 E/P`, `RQ2 E/P`
- **Table 1, Panel B:** `Profile mean; CVaR`,
  `No-repair mean; CVaR`, `max_(a,k) Delta_(a,k) [UCB]`,
  `min_a h_a [LB]`, `min F/U slack`, `updates/rollback`, `RQ3 E/P`
- **Table 2:** `Axis`, `Setting`, `Plan/done`, `RQ1 E/P`, `RQ2 E/P`,
  `RQ3 E/P`, `valid/reach`, `tail/common n`, `all-arm feasible`,
  `worst RQ1/RQ2/RQ3 bounds`, `Failure modes`

Table 1의 parent roster는 output-readout
`{GradDiff, NPO, SimNPO, GRU}`와 representation-readout
`{RMU, RepNoise, CB}`다.

PDF Table 2의 setting roster는 정확히 8행이다.

| Axis | PDF setting |
|---|---|
| Request | held-out TOFU requests |
| Dataset | WMDP-bio/MMLU |
| Dataset | MUSE-News |
| Dataset | RWKU |
| Dataset | MUSE-Books (stress) |
| Dataset | PISTOL (stress) |
| Model | Qwen2.5-7B |
| Model | Llama-3.1-8B |

PDF Table 3은 TOFU primary를 `Qwen2.5-1.5B / fp32`, scale을
`Qwen2.5-7B / fp32`, family를 `Llama-3.1-8B / fp32`로 정의한다.
현재 코드/config의 7B-primary, 1.5B-boundary, 14B 추가 roster는 이 PDF와
일치하지 않으며 별도 동기화 전에는 paper contract로 취급하면 안 된다.

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

## 3. Table 1 Panel A: RQ1/RQ2

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
- `P=y`: `rho_S`, `g_G`, `g_H`, `L_tail`의 네 lower bound가 모두
  `0`보다 크고 four-way IUT가 `alpha=0.05`를 통과한다.
- 모든 planned trajectory가 완료되지 않으면 pass가 차단된다.

### RQ2 E/P

- `E=y`: paired fidelity/common-control support가 2 unit 이상이고,
  perturbation, exact-reference, common-control validity가 모두 참이다.
- `P=y`: `f_rho-0.80`, `f_K-0.70`, `g_H`, `g_ctl`의 네 lower bound가
  모두 `0`보다 크고 four-way IUT를 통과한다.

`g_H`는 RQ1에서는 두 endpoint 중 하나이고, RQ2에서는 proximity만으로는
설명되지 않는 loss-shake의 added value를 검사한다.

## 4. Table 1 Panel B: RQ3

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

## 5. Table 2: robustness와 failure boundary

Table 2는 새로운 candidate metric을 계산하지 않는다. Table 1과 같은
decision을 위에 열거한 PDF의 8개 predeclared setting에 대해 요약하며,
미실행 setting도 분모에서 제거하지 않는다.

| 열 | 의미 |
|---|---|
| `Axis` | request, model, dataset 중 변화 축 |
| `Setting` | dataset/model setting; stress와 boundary는 표시 |
| `Plan/done` | 계획된 parent row 수 / 완전히 완료된 parent row 수 |
| `RQ1 E/P` | 해당 setting에서 RQ1 eligible parent 수 / passed parent 수 |
| `RQ2 E/P` | 해당 setting에서 RQ2 eligible parent 수 / passed parent 수 |
| `RQ3 E/P` | 해당 setting에서 RQ3 eligible parent 수 / passed parent 수 |
| `valid/reach` | valid profile/planned profile; reached trajectory/planned trajectory |
| `tail/common n` | RQ1 tail-eligible unit 수 / prediction common-support unit 수 |
| `all-arm feasible` | 다섯 arm이 모두 feasible한 unit 수 / reached+valid unit 수 |
| `worst RQ1/RQ2/RQ3 bounds` | RQ1 LB 최솟값 / RQ2 LB 최솟값 / RQ3 damage UCB 최댓값 |
| `Failure modes` | non-reach, invalid profile, common-support, constraint/IUT 실패 요약 |

`worst` 열에서 RQ1·RQ2는 큰 값이 유리하므로 lower bound의 최솟값을,
RQ3 damage는 작은 값이 유리하므로 upper bound의 최댓값을 쓴다. Native
non-inferiority bound는 damage와 척도가 달라 Table 1의 `min_a h_a`에만
표시한다. 이 worst 값은 기술적 요약이며 누락 row를 대신하거나 pass를
허가하지 않는다.

Setting-level/transfer claim은 row별 IUT보다 더 엄격하다. 각 readout parent
group 안에서 Bonferroni 보정한 alpha를 적용해 RQ1/RQ2/RQ3를 모두 통과한
parent가 필요하다. Primary, model-transfer, dataset-replication group의
사전 동결 규칙까지 만족해야 최종 transfer 문구가 licensed된다. Stress
setting은 primary 실패를 구제할 수 없다.

## 6. 구현 위치와 PDF 일치 상태

| 단계 | 구현 |
|---|---|
| Damage/fidelity/protection raw 생성 | `experiments/paper/tofu_v4_unit.py` |
| Spearman, tail lift, CVaR, effect 집계 | `src/rsus/evidence/raw.py`, `src/rsus/analysis/prediction.py` |
| Bootstrap bound와 p-value | `src/rsus/evidence/statistics.py` |
| RQ1/RQ2/RQ3 IUT와 eligibility | `src/rsus/evidence/pdf_v4.py`, `src/rsus/evidence/decisions.py` |
| Table 1/2 렌더링 | `src/rsus/evidence/tables.py`, `src/rsus/evidence/table1.py` |
| 현재 동결 config | `configs/paper/evidence.yaml`, `configs/paper/campaign.yaml` |

Metric 계산과 RQ1/RQ2/RQ3 IUT의 부호는 PDF와 대응한다. 그러나 저장소 전체가
최신 PDF와 동기화된 것은 아니다.

| 항목 | 상태 |
|---|---|
| Table 1 metric 열과 4/4/12-way IUT | PDF와 대응 |
| Fractional CVaR.95와 low-tail-support 규칙 | PDF와 대응 |
| `paper/main.tex`이 컴파일하는 Table 1 | 불일치: 구버전 channel matrix |
| TOFU primary/scale/family roster | 불일치: PDF는 1.5B/7B/Llama |
| Table 2 denominator | 불일치: PDF 8행, 현재 config/renderer 9행 |
| 실제 target evidence | 미생성 |

따라서 현재 생성 renderer의 숫자는 metric-level 진단에는 사용할 수 있지만,
roster를 PDF와 동기화하기 전에는 최신 PDF Table 1/2를 채운 paper evidence로
간주할 수 없다.
