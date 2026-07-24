# local_run — PDF v4 로컬 실행기와 legacy 게이트 도구

## 현재 PDF v4 실행 경로

루트의 `local_run.sh` 또는 `local_run/local_run.sh`가
`experiments/local_pdf_v4.py`를 호출한다. 이 경로만 July-24 PDF의
loss-shake/hidden 혼합, exact Top-Kp, parent first-reaching, Equation (7)--(8),
hard token/example guard, rollback/retry, processed-token budget을 연결한다.

```bash
cp configs/local/pdf_v4.example.yaml configs/local/pdf_v4.local.yaml

# 1) 모델 block의 실제 이름/hash/count를 확인해 local config에 기록
bash local_run.sh inspect-model configs/local/pdf_v4.local.yaml

# 2) score를 계산하기 전에 discovery/audit/neutral/utility/eligibility 동결
bash local_run.sh prepare-manifest configs/local/pdf_v4.local.yaml

# 3) D_cal/D_prot에서 정한 모든 null 값을 채우고 status를
#    frozen_for_local_diagnostic으로 바꾼 뒤 검증
bash local_run.sh validate configs/local/pdf_v4.local.yaml

# 4) 실행. 결과는 config의 output.directory에 기록
bash local_run.sh run configs/local/pdf_v4.local.yaml
```

`prepare-manifest`는 기존 manifest를 덮어쓰지 않는다. 다시 만들려면 Python
entrypoint에 `--force`를 명시해야 한다. `run`은 null/TBD, manifest hash 불일치,
block hash/count 불일치, fp32가 아닌 block, `k>|Df|`, 미도달 parent를 묵인하지
않는다. 출력 디렉터리도 기본적으로 덮어쓰지 않는다. semantic duplicate/
paraphrase/template review를 하지 않은 manifest에는
`eligibility_status: provisional_local_diagnostic`이 남는다. 이 단일 로컬 arm은
calibration/comparator/CI 전체를 대신하지 않으므로
결과에는 항상 `claim_eligible: false`가 기록된다.

## Legacy 게이트 (PDF v4 실행에 사용 금지)

`experiments/gate_1p5b/gate.py`(retain-susceptibility 프로브 게이트)를 사내 로컬
머신(RTX 4090 ×2, 24GB)에서 여러 모델·데이터셋에 대해 돌리고 결과를 정리하는 래퍼.
아래 `run_one.sh`, `run_queue.sh`, `sweep_ours.sh`는 이전 Stage1/Stage2 진단을
재현하기 위해 남겨 둔 것이며 PDF v4 repair를 실행하지 않는다.

## 경로 규약
- 모델: `/rdata/models/` (공유 zoo)
- 데이터(HF_HOME): `/rdata/minsoo3.kim/hf_home` (TOFU · RWKU · MUSE 캐시)
- 결과: `/rdata/minsoo3.kim/results/<dataset>/<label>/`

## 스크립트
| 파일 | 용도 |
|---|---|
| `download_models.sh` | 누락 모델(1.5B/4B 등)을 `/rdata/models`로 다운로드 (Xet 비활성) |
| `download_data.py` | TOFU + RWKU + MiniLM 인코더 다운로드 |
| `download_muse.py` | MUSE-News / MUSE-Books 다운로드 |
| `local_run.sh` | 새 PDF v4 config/manifest 검증 및 실행 |
| `run_one.sh` | legacy 모델 1개 × 데이터셋 1개 게이트 실행 + 요약 |
| `run_queue.sh` | run_one 여러 개를 순차 실행 (레인) |
| `summarize.py` | table1/2.json → 마크다운 (열 1등 볼드+밑줄) |

## Legacy 실행 예시
```bash
# 단일 (TOFU, GPU0)
GPU=0 bash local_run/run_one.sh 3b Qwen2.5-3B-Instruct float32

# RWKU (author 0, remote pool 100-119)
DATASET=rwku GPU=0 bash local_run/run_one.sh 3b Qwen2.5-3B-Instruct float32

# 7B fp32는 2-GPU 분산 (마지막 8레이어를 cuda:1에)
GPU=0,1 DMAP=split:8 bash local_run/run_one.sh 7b_fp32 Qwen2.5-7B-Instruct float32
```

## 4090(24GB) 메모리 노트
- `--trainable-scope probe_block` (1-GPU 모드), `--pool-size 16 --batch-size 4`.
- `≤4B` = fp32(단일 GPU), `7B fp32` = `--device-map split:8`로 2-GPU 분산
  (auto/balanced는 학습 레이어 GPU를 과적재해 OOM → 수동 split 필요).
- `bf16`는 `fd`(eta=3e-4 유한차분)를 정밀도 바닥으로 무너뜨림 → 프로브는 fp32 권장.
- Table 2 two-stage(ours/s2s)는 stage1이 full-model fp32 AdamW라 24GB에서 불가(H100 필요).

## gate.py 확장 (이 포크에서 추가)
- `--device-map` : `auto`|`balanced`|`split:K` (K = cuda:1에 올릴 마지막 레이어 수)
- `--max-memory-gib` : device_map 샤딩 시 GPU별 가중치 상한(정수 GiB, 콤마 리스트 가능)
