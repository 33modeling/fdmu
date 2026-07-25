# H100: Qwen2.5-14B joint sweep agent guide

이 문서는 coding agent가 Qwen2.5-14B TOFU PDF-v4 개발 스윕을 H100에서
설치부터 결과 보존까지 수행하는 기준이다. RTX 4090 24GB 두 장은 14B fp32
공식 실행 환경이 아니다.

## 1. 하드웨어 중단 조건

14B fp32 가중치만 약 56GB이다. activation, last-8 trainable block,
optimizer state, loss-shake/repair vector까지 더하면 2x4090 48GB aggregate로
실행할 수 없다.

다음 환경이면 실험을 시작하지 않는다.

```text
GPU: RTX 4090 x2 24GB
dtype: float32
result: HARDWARE_UNSUPPORTED
```

bf16, CPU offload, ZeRO/FSDP로 자동 전환하지 않는다. 그런 경로는 새로운
수치/실행 계약이며 loss-shake fidelity와 repair rollback을 별도로 검증해야
한다.

기본 지원 환경:

```text
GPU: H100 80GB
allocation: one complete unit per H100
dtype: float32
parallel units: number of free H100s
```

한 H100에서 preflight가 OOM이면 batch size나 과학 파라미터를 자동 변경하지
말고 allocation 단계와 peak memory를 기록한다.

## 2. 완료 조건

`joint`가 이 프로젝트의 방법이다. 같은 parent first-reaching checkpoint와
같은 candidate support에서 다음 arm을 비교한다.

```text
joint, s0, s1, no_repair, repeated_random x 5
```

모든 `D_prot request x seed` 셀에서 joint가 feasible이어야 한다. Infeasible
comparator는 제약 우선 순위에서 탈락하고, comparator도 feasible일 때는
joint의 mean damage와 CVaR damage가 모두 엄격히 낮아야 해당 parent가
통과한다.
최종 종료에는 아래 두 그룹에서 각각 하나 이상의 passing parent가 필요하다.

```text
output-readout:         graddiff, npo, simnpo, gru
representation-readout: rmu, repnoise, circuit_breakers
```

판정에는 `D_prot`만 사용한다. target/audit outcome을 읽은 뒤 trial을 추가하면
안 된다.

## 3. 저장소 기준

| 항목 | 위치 |
|---|---|
| 14B paper model | `configs/paper/campaign.yaml` |
| 14B evidence setting | `configs/paper/evidence.yaml` |
| 14B paper runtime | `configs/paper/tofu_v4.yaml` |
| 14B parent grid | `configs/channel_matrix/14b_tofu.yaml` |
| frozen parent 설정 | `configs/channel_matrix/objective_freeze_14b.yaml` |
| protection producer | `experiments/paper/tofu_v4_unit.py` |
| sweep controller | `experiments/paper/run_joint_dev_sweep.py` |
| reference trial grid | `configs/local/joint_sweep_1p5b_4090x2.yaml` |

14B runtime은 다음 계약을 사용한다.

```yaml
setting: tofu_qwen25_14b
channel_model_id: qwen25_14b
SFT: lr=1e-5, steps=800, target_loss=0.8, eval_every=100
parent freeze: configs/channel_matrix/objective_freeze_14b.yaml
```

## 4. 설치와 offline preflight

클러스터의 repository/model/results 경로는 환경 변수로 넘긴다. canonical
config에 host별 절대 경로를 커밋하지 않는다.

```bash
export REPO=/path/to/retain-susceptibility
export MODEL_PATH=/group-volume/models/Qwen2.5-14B-Instruct
export ENCODER_PATH=/group-volume/models/all-MiniLM-L6-v2
export RESULTS_ROOT=/group-volume/results/paper/tofu_qwen25_14b/joint_sweep
export SFT_CACHE_ROOT=/group-volume/results/paper/tofu_qwen25_14b/sft_cache
export HF_HOME=/group-volume/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

에이전트가 확인할 항목:

1. Python 3.11 venv와 repository editable install
2. CUDA-enabled PyTorch와 H100 compute capability
3. 각 H100의 총/가용 메모리와 active process
4. model/tokenizer/encoder/TOFU dataset의 offline load
5. Qwen2.5-14B fp32 weight load
6. last-8 down-projection block만 trainable인지 확인
7. real-token forward/backward와 block optimizer step
8. loss-shake parameter 왕복 복원
9. repair vector 할당과 rollback
10. GPU별 peak allocated/reserved memory 저장

사용 중인 GPU의 process는 kill하지 않는다. free H100만 worker lane으로 넣는다.

## 5. 14B spec 만들기

1.5B reference spec을 별도 local file로 복사하고 다음만 바꾼다.

```yaml
setting: tofu_qwen25_14b
paths:
  model_source: /group-volume/models/Qwen2.5-14B-Instruct
  sentence_encoder: /group-volume/models/all-MiniLM-L6-v2
  sft_cache_root: /group-volume/results/paper/tofu_qwen25_14b/sft_cache
  output_root: /group-volume/results/paper/tofu_qwen25_14b/joint_sweep
gpus: [0, 1]  # 실제로 할당받은 free H100 id만 기록
```

controller는 GPU마다 독립 child를 실행하고 child는 `cuda:0`만 보게 한다.
즉 두 H100이면 unit 두 개를 병렬 실행한다. 14B에서는 model sharding이나
`device_map`을 넣지 않는다.

trial의 alpha와 Kp는 각각 singleton이어야 한다. 이렇게 해야 모든 개발 셀이
같은 전역 설정을 사용하고 comparator arm과의 비교가 일치한다. repair trial
목록은 순서와 기존 항목을 바꾸지 말고 뒤에만 추가한다.

## 6. 실행과 재개

controller CLI의 표준 호출:

```bash
.venv/bin/python -u experiments/paper/run_joint_dev_sweep.py \
  --spec configs/local/joint_sweep_14b_h100.local.yaml \
  --gpus 0,1 \
  --model-source "$MODEL_PATH" \
  --sentence-encoder "$ENCODER_PATH" \
  --sft-cache-root "$SFT_CACHE_ROOT" \
  --output-root "$RESULTS_ROOT"
```

같은 명령을 다시 실행하면 config hash와 output hash가 유효한 unit만
건너뛴다. 단순한 `DONE` 파일이나 `.lock` 존재만으로 완료 처리하지 않는다.
SFT cache hit는 request/seed contract hash와 state hash가 모두 맞아야 한다.

unit 오류가 발생하면 해당 trial에서 멈춘다. infrastructure 오류를 나쁜
hyperparameter로 취급해 다음 trial로 넘어가면 안 된다.

## 7. 종료 후 산출물

```text
<output_root>/
  environment.json
  sweep_manifest.json
  specs/<sha256>.yaml
  events.jsonl
  summary.csv
  trials/<id>--<hash>/
    config/campaign.local.yaml
    config/tofu_v4.local.yaml
    manifest.yaml
    logs/units/<unit>/attempt-*.log
    units/<unit>/protection_diagnostics.json
    stage/stage_manifest.json
    joint_comparison.json
  BEST.json
  recommendation.yaml
```

`BEST.json`이 만들어져도 target을 자동 실행하지 않는다. 이 파일은
development recommendation이며 `status: draft`,
`human_review_required: true`, `target_used: false`여야 한다.

사용자가 passing parent, 모든 셀, resolved runtime hash를 검토한 뒤
prospective selection freeze를 커밋한다. 그 다음 별도 명령으로 target stage와
Table 2/LaTeX 생성을 실행한다.

## 8. NO_JOINT_DOMINANCE 처리

사전 선언한 trial이 모두 실패하면 controller는 `NO_JOINT_DOMINANCE`,
exit code 3으로 로그와 failure report를 남기고 종료한다. 이 결과도 유효한
부정적 실험 결과이며 무한히 trial을 생성하지 않는다.
에이전트는 다음 절차만 허용된다.

1. `joint_comparison.json`에서 D_prot 실패 셀과 comparator를 요약
2. target 파일을 읽지 않았음을 확인
3. repair 안정성 범위 안의 새 trial을 spec 마지막에 추가
4. 변경 이유와 spec hash를 commit
5. 동일 output root에서 재실행

기존 trial 디렉터리, 실패 로그, manifest를 삭제하거나 덮어쓰지 않는다.
