# Final results runbook

이 문서는 최신 PDF 기준 TOFU Table 1 결과를 만들고 확인하는 실행 절차다.
Predictor 값의 해석은 [`PREDICTOR_METRICS.md`](PREDICTOR_METRICS.md)를 본다.

## 0. 현재 범위

최신 `KDD_UnlearningFail.pdf`에서 TOFU primary는
`Qwen2.5-1.5B / fp32`다. 따라서 이 문서의 기본 setting은 다음과 같다.

```text
tofu_qwen25_1p5b
```

현재 `configs/paper/evidence.yaml`에는 이전 7B-primary, 14B 포함 9-setting
roster가 남아 있다. 아래 전용 runner는 **TOFU 1.5B의 per-setting Table 1**을
만들 수 있지만, 현재 config 그대로는 최신 PDF의 전체 8-setting Table 2와
paper-wide `all_tables_ready`를 만들었다고 해석하면 안 된다.

또한 이전 9-setting용 `experiments/cluster/enqueue_table12.sh`를 최신 PDF
paper evidence 실행에 사용하면 안 된다.

## 1. 실행 환경

모델 실행은 fp32와 RQ3 repair 때문에 H100급 GPU 환경을 전제로 한다.
저장소 루트에서 실행한다.

```bash
cd /path/to/fdmu
source /group-volume/fdmu/.venv/bin/activate

python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

두 번째 출력이 `True`여야 한다. 기본 모델 경로는 campaign config의
다음 위치다.

```text
/group-volume/models/Qwen2.5-1.5B-Instruct
```

현재 `/home/kms`의 기본 Python에는 `torch`가 없으므로 그 환경에서는 실제
GPU workflow를 실행할 수 없다.

## 2. 먼저 실행 계획만 생성

아래 명령은 GPU 학습을 하지 않고 정확한 stage manifest 네 개를 만든다.

```bash
python experiments/paper/run_tofu_table1.py \
  --action plan \
  --setting tofu_qwen25_1p5b
```

생성 위치:

```text
runs/paper/tofu_table1/manifests/
  tofu_qwen25_1p5b__calibration.yaml
  tofu_qwen25_1p5b__prediction.yaml
  tofu_qwen25_1p5b__protection.yaml
  tofu_qwen25_1p5b__target_evaluation.yaml
```

이 단계에서 manifest roster와 명령을 검토한 뒤 실제 실행으로 넘어간다.
현재 동결 roster의 planned unit 수는 다음과 같다.

```text
calibration:        28
prediction:         56
protection:         56
target_evaluation: 140
total:             280
```

## 3. 전체 TOFU 결과 실행

4090 x2에서 환경 설정부터 calibration, joint sweep, target, evidence,
`table1.tex`까지 원클릭으로 실행한다.

```bash
GPU_IDS=0,1 bash local_run/run_tofu_1p5b_4090x2.sh
```

통합 로그는 `<RUN_ROOT>/launcher_logs/current.log`에 남는다. Parent proposal과
joint `BEST.json`은 target 전에 화면에 출력되며, 각 파일 해시에 묶인 승인
문구를 직접 입력해야 한다. Declared fidelity 결과는
`<RUN_ROOT>/fidelity/fidelity_summary.json`에 생성된다.

4090 x2 joint sweep이 이미 `BEST.json`을 만들었다면 calibration과 D_prot를
다시 실행하지 않고 다음 두 단계를 순서대로 사용한다.

```bash
bash local_run/run_tofu_1p5b_fidelity.sh

APPROVE_JOINT_BEST=1 GPU_IDS=0,1 \
  bash local_run/finalize_joint_sweep_to_latex.sh
```

이 명령은 winning D_prot를 재사용해 D_pred, prospective selection freeze,
target evaluation, raw aggregation, evidence 판정, `final/table1.tex` 생성을
순서대로 수행한다. `APPROVE_JOINT_BEST=1`은 target을 보기 전에
고정된 development-only BEST 선택을 사용한다는 명시적 사전 승인이다.

이미 동결된 stage manifest에서 범용 Table 1 runner만 실행할 때는 다음 명령을
사용한다.

```bash
python experiments/paper/run_tofu_table1.py \
  --action run \
  --setting tofu_qwen25_1p5b
```

Runner는 다음 순서를 강제한다.

```text
D_cal 실행
  -> parent 설정 동결
  -> D_pred 실행
  -> D_prot 실행
  -> alpha_pred, alpha_prot, control, Kp 동결
  -> target_evaluation 실행
  -> raw evidence 집계
  -> RQ1/RQ2/RQ3 판정
  -> Table 1 생성
```

현재 top-level runner는 unit을 순차 실행하며 `--resume` 옵션이 없다. 중간에
중단된 뒤 같은 `--action run`을 다시 실행하면 완료 unit도 다시 호출한다.
따라서 장시간 실행은 유지되는 H100 세션에서 수행해야 한다.

## 4. SFT 재사용

각 request/seed의 SFT checkpoint는 다음 위치에 저장된다.

```text
runs/paper/tofu_v4/sft_cache/
  <setting>/<request>__seed-<seed>__<contract-digest>.pt
  <setting>/<request>__seed-<seed>__<contract-digest>.pt.json
```

동일한 model, request, seed, candidate universe, trainable block, SFT
hyperparameter contract가 일치하면 checkpoint를 불러오고 SFT를 다시 하지
않는다. 로그에는 다음과 같이 표시된다.

```text
loaded validated development SFT cache ...
```

각 unit의 `run_manifest.json`에는 아래 값이 기록된다.

```json
{
  "sft_cache": {
    "hit": true
  }
}
```

Contract가 다른 cache를 같은 checkpoint로 조용히 재사용하지 않는다.
이전 고정 이름 `<request>__seed-<seed>.pt` 캐시는 metadata contract가 현재
실행과 정확히 일치할 때만 계속 사용한다. 불일치하거나 불완전한 이전 캐시는
삭제하지 않고 보존하며, 현재 contract digest 경로에 새 checkpoint를 만든다.

## 5. 최종 결과 파일

전체 runner가 성공하면 가장 먼저 볼 파일은 다음 두 개다.

```text
paper/sections/generated/table1.tex
runs/paper/tofu_table1/tofu_qwen25_1p5b/evidence_readiness.json
```

주요 산출물 전체:

| 파일 | 의미 |
|---|---|
| `paper/sections/generated/table1.tex` | TOFU 1.5B의 최종 2-panel Table 1 |
| `runs/paper/tofu_table1/tofu_qwen25_1p5b/evidence_readiness.json` | parent별 RQ1/RQ2/RQ3 eligibility, pass, 실패 이유 |
| `runs/paper/tofu_table1/tofu_qwen25_1p5b/evidence_ledger.json` | bootstrap effect와 funnel을 담은 정규화 evidence |
| `runs/paper/tofu_table1/tofu_qwen25_1p5b/raw_plan.json` | 집계에 사용한 동결 plan |
| `runs/paper/tofu_table1/tofu_qwen25_1p5b/*/sealed/*.jsonl` | stage별 검증·봉인 raw evidence |
| `runs/paper/tofu_table1/tofu_qwen25_1p5b/*/sealed/stage_manifest.json` | stage 산출물 개수와 SHA-256 |

Table을 터미널에서 바로 확인:

```bash
sed -n '1,260p' paper/sections/generated/table1.tex
```

Parent별 최종 판정 확인:

```bash
jq '{
  setting: .settings.tofu_qwen25_1p5b,
  rows: [
    .rows[]
    | select(.setting == "tofu_qwen25_1p5b")
    | {
        parent,
        completed,
        rq1: (.rq1 | {eligible, claim_pass, reasons}),
        rq2: (.rq2 | {eligible, claim_pass, reasons}),
        rq3: (.rq3 | {eligible, claim_pass, reasons})
      }
  ]
}' runs/paper/tofu_table1/tofu_qwen25_1p5b/evidence_readiness.json
```

행별로 좋은 결과는 각 parent의 `RQ1`, `RQ2`, `RQ3`에서
`eligible=true`, `claim_pass=true`인 것이다. Table의 표기는 `Y/Y`다.

Setting 전체의 최종 판정은 모든 parent가 통과해야 하는 방식이 아니다.
Bonferroni 보정 후 output-readout parent group과 representation-readout
parent group에서 각각 최소 한 parent가 RQ1, RQ2, RQ3를 모두 통과해야 한다.
아래 값이 최종 setting chain 판정이다.

```bash
jq '.settings.tofu_qwen25_1p5b.chain' \
  runs/paper/tofu_table1/tofu_qwen25_1p5b/evidence_readiness.json
```

최종 성공 값:

```json
{
  "pass": true
}
```

## 6. 이미 실험 결과가 있을 때 표만 재생성

`evidence_ledger.json`이 이미 있으면 GPU 모델 실행이나 SFT 없이 CPU에서
Table과 readiness만 다시 만들 수 있다.

```bash
python experiments/paper/build_evidence.py \
  --config configs/paper/evidence.yaml \
  --ledger runs/paper/tofu_table1/tofu_qwen25_1p5b/evidence_ledger.json \
  --readiness-out runs/paper/tofu_table1/tofu_qwen25_1p5b/evidence_readiness.json \
  --table1-setting tofu_qwen25_1p5b \
  --table1-out paper/sections/generated/table1.tex
```

이 명령은 저장된 ledger를 다시 판정하고 LaTeX를 렌더링할 뿐 학습을 실행하지
않는다.

## 7. 전체 paper table 생성

모든 setting의 ledger가 완성되고 `configs/paper/evidence.yaml`을 최신 PDF의
8-setting roster와 동기화한 뒤에는 다음 명령을 사용한다.

```bash
python experiments/paper/build_evidence.py \
  --config configs/paper/evidence.yaml \
  --ledger results/paper/evidence_ledger.json \
  --paper-root paper \
  --require-ready
```

성공 시 생성되는 main table:

```text
paper/sections/generated/table_core_evidence.tex
paper/sections/generated/table_robustness.tex  # Table 2A breadth + 2B funnel
paper/sections/generated/results_macros.tex
```

`--require-ready`는 등록된 row나 artifact가 하나라도 불완전하면 exit code
`2`로 실패한다. Placeholder가 있는 상태를 최종 완료로 오인하지 않기 위한
검사다.

현재 `paper/main.tex`과 `paper/sections/05_experiments.tex`에는 위 generated
Table 파일을 자동으로 `\input`하는 구문이 없다. 따라서 생성된 `.tex` 파일
자체는 확인할 수 있지만, 현재 manuscript PDF를 다시 컴파일해도 새 Table이
자동 삽입되지는 않는다. Manuscript 반영은 generated Table include 위치를
확정한 뒤 별도로 연결해야 한다.

## 8. 빠른 선택

```text
실행 계획만 확인:
  run_tofu_table1.py --action plan

TOFU GPU 실험부터 최종 Table 1까지:
  run_tofu_table1.py --action run

기존 ledger에서 Table만 다시 생성:
  build_evidence.py --ledger ... --table1-out ...

전체 paper readiness를 강제:
  build_evidence.py --paper-root paper --require-ready
```
