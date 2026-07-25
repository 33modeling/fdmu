# Table 1/2 metric guide

이 문서는 최신 PDF-v4 증거 파이프라인이 생성하는 다음 두 표의 열을
설명한다.

- Table 1: `tab:core-evidence`
  (`paper/sections/generated/table_core_evidence.tex`)
- Table 2: `tab:robustness`
  (`paper/sections/generated/table_robustness.tex`)

`paper/sections/05_experiments.tex`에 직접 작성된 이전 channel-matrix 표와
`local_run/table1.json`, `local_run/table2.json`은 이 문서의 범위가 아니다.
계산과 렌더링이 구현되어 있다는 것과 실제 target campaign이 완료되어 숫자가
채워졌다는 것은 다르다. 원자료가 없거나 eligibility가 불완전한 셀은
`\tblph`로 남는다.

## 1. 공통 표기

한 실험 unit은 `(setting, parent, request, seed)`이다. 후보 행동을 `x`,
초기 SFT checkpoint를 `theta_0`, arm `a`의 평가 checkpoint를 `theta_a`라
하면 audit damage는 다음과 같다.

```text
d_a(x) = NLL(x; theta_a) - NLL(x; theta_0)
```

`d_a(x) > 0`은 해당 retain 행동의 NLL이 증가해 손상이 생겼다는 뜻이다.
따라서 damage의 mean과 CVaR는 작을수록 좋다.

두 기본 score와 결합 score는 다음과 같다.

```text
S0(x) = q_G(x)  # loss-shake susceptibility
S1(x) = q_H(x)  # forget-conditioned proximity
S_joint(x) = (1 - alpha) rank(S0(x)) + alpha rank(S1(x))
```

`alpha_pred`와 `alpha_prot`는 서로 다른 development fold에서 target 결과를
보지 않고 동결한다.

| arm | 의미 |
|---|---|
| `joint` | 동결된 `S_joint`로 보호 대상을 선택해 repair |
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
rho_joint = Spearman(S_joint(x), d_parent(x))
```

범위는 `[-1, 1]`이다. 양수이면 높은 사전 score가 실제 고손상 후보를 먼저
배치했다는 뜻이다. RQ1에서는 `LB > 0`을 요구한다.

### min(g_G, g_H) [min LB]

결합 score가 두 단독 score보다 얼마나 나은지 측정한다.

```text
g_G = rho_joint - Spearman(S0, damage)
g_H = rho_joint - Spearman(S1, damage)
```

표에는 두 점 추정치의 최솟값과 두 lower bound의 최솟값을 표시한다. 둘 다
양수여야 결합 score가 `q_G`, `q_H` 각각보다 낫다고 말할 수 있다.

### f_rho / f_K [margin LB]

Loss-shake가 동일 parameter block의 exact per-candidate gradient energy를
얼마나 잘 재현하는지 측정한다.

```text
f_rho = Spearman(loss_shake_score, exact_gradient_energy)
f_K   = |TopK(loss_shake) intersect TopK(exact)| / K

margin_rho = f_rho - 0.80
margin_K   = f_K - 0.70
```

`K`는 동결된 `Kp`다. 표의 앞 숫자는 실제 `f_rho/f_K`, 대괄호 안 숫자는
floor 대비 lower-bound margin이다. Setting-level fidelity certificate는
진단용 fallback일 뿐이며, per-unit `fidelity_raw.jsonl` 없이 RQ2 pass를
만들 수 없다.

### g_ctl [LB]

결합 score가 `D_pred`에서 미리 선택한 strongest simple control보다 얻는
Spearman 이득이다.

```text
g_ctl = rho_joint - Spearman(S_control, damage)
```

`LB > 0`이면 결합 score가 선택된 최강 단순 control보다 유의하게 낫다.

### L_tail [LB]; eligible n/N

전체 후보 수를 `N`, 미리 동결한 tail 크기를 `M`이라 한다. 양의 damage
후보가 최소 `M`개인 unit에서만 다음 값을 계산한다.

```text
H_M = positive-damage candidates 중 damage 상위 M개
P_M = S_joint 상위 M개
Recall_M = |H_M intersect P_M| / M
q = M / N
L_tail = Recall_M / q - 1
```

무작위 순위의 기대값은 `L_tail=0`이다. `eligible n/N`은 tail metric을
계산할 수 있었던 unit 수와 reached+valid unit 수다. RQ1 eligibility에는
이 비율이 최소 `0.80`이어야 한다.

### RQ1 E/P

`E/P`는 eligible/pass다.

- `E=y`: prediction selection이 valid/non-fallback이고, profile과 exact
  common support가 유효하며, support unit이 2개 이상이고, tail coverage가
  `0.80` 이상이다.
- `P=y`: `rho_joint`, `g_G`, `g_H`, `L_tail`의 네 lower bound가 모두
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
동결된 허용 열화량을 `delta_NI >= 0`이라 하면:

```text
h_a = native(joint) - native(a) + delta_NI  # higher is better
h_a = native(a) - native(joint) + delta_NI  # lower is better
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
ledger/decision을 모든 predeclared setting에 대해 요약하며, 미실행 setting도
분모에서 제거하지 않는다.

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

## 6. 구현 위치와 점검 결과

| 단계 | 구현 |
|---|---|
| Damage/fidelity/protection raw 생성 | `experiments/paper/tofu_v4_unit.py` |
| Spearman, tail lift, CVaR, effect 집계 | `src/rsus/evidence/raw.py`, `src/rsus/analysis/prediction.py` |
| Bootstrap bound와 p-value | `src/rsus/evidence/statistics.py` |
| RQ1/RQ2/RQ3 IUT와 eligibility | `src/rsus/evidence/pdf_v4.py`, `src/rsus/evidence/decisions.py` |
| Table 1/2 렌더링 | `src/rsus/evidence/tables.py`, `src/rsus/evidence/table1.py` |
| 동결된 threshold/roster | `configs/paper/evidence.yaml`, `configs/paper/campaign.yaml` |

수식 점검 기준으로 Table 1/2에 표시되는 metric은 모두
producer -> raw validation -> aggregation -> decision -> renderer 경로가
연결되어 있다. 현재 남은 `\tblph`는 metric 함수 미구현이 아니라 실제
target raw evidence 또는 비-TOFU exact producer/roster가 아직 완료되지 않은
경우다.
