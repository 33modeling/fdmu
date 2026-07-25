# RTX 4090 x2: Qwen2.5-14B hardware decision guide

이 문서는 Qwen2.5-14B TOFU PDF-v4 실험을 RTX 4090 24GB 두 장에서
요청받은 coding agent가 잘못된 저정밀도 실험이나 반복 OOM을 만들지 않도록
하는 중단 계약이다.

## 결론

현재 claim-bearing 계약으로는 실행하지 않는다.

```text
model: Qwen2.5-14B
dtype: float32
GPU: RTX 4090 24GB x2
decision: HARDWARE_UNSUPPORTED
```

14B fp32 가중치만 약 56GB다. 두 GPU의 총 VRAM 48GB보다 크고, 실제 실행에는
activation, last-8 trainable block, optimizer state, finite-difference
direction, repair velocity/reference tensor도 추가된다. 따라서 단순
`device_map=split:*`로도 들어가지 않는다.

## 에이전트가 해야 할 preflight

실험 프로세스를 띄우기 전에 아래 정보만 기록한다.

```bash
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.free \
  --format=csv,noheader
```

두 장 모두 24GB 4090이면 `<results_root>/preflight/hardware_unsupported.json`을
다음 필드로 만든 뒤 종료한다.

```json
{
  "model": "Qwen2.5-14B",
  "required_dtype": "float32",
  "hardware": "RTX 4090 24GB x2",
  "supported": false,
  "reason": "fp32 weights exceed aggregate VRAM before runtime allocations",
  "next_guide": "docs/H100_14B_JOINT_SWEEP_AGENT_GUIDE.md"
}
```

이 판정에는 model load OOM을 실제로 발생시킬 필요가 없다. 다른 GPU
프로세스를 kill하거나 기존 결과를 삭제하지 않는다.

## 자동 우회 금지

다음 변경은 논문과 다른 새로운 실행 계약이므로 agent가 임의로 적용하면 안
된다.

- bf16, fp16, fp8 또는 양자화
- CPU/NVMe offload
- ZeRO-3 또는 FSDP parameter sharding
- `R`, `eta`, `block_last_n`, batch size 변경
- 7B나 1.5B checkpoint를 14B 행 이름으로 실행

특히 loss-shake는 실제 parameter coordinate를 작은 반경으로 교란한다.
저정밀도로 바꾸면 반경이 coordinate ULP 아래로 내려가 predictor와 fidelity가
달라질 수 있다.

## 4090에서 14B를 새로 지원하려면

사용자가 별도 연구 작업으로 승인한 경우에만 다음 계약을 먼저 구현한다.

1. 최소 4x4090 또는 CPU offload/FSDP 중 하나를 명시적으로 선택
2. fp32 claim block과 perturbation vector 보존
3. cross-device forward/backward, fresh-model restore, repair rollback 구현
4. 같은 fixture를 H100 단일 GPU fp32 기준과 비교
5. finite-difference score 순위와 fidelity threshold 검증
6. parent first-reaching step, forgetting recall, mean/CVaR damage 회귀 검증
7. 새 runtime contract와 preflight test를 commit

이 검증 전의 출력은 `claim_eligible: false`인 진단 결과일 뿐 Table 1/2에
들어갈 수 없다.

## 지원되는 다음 경로

14B 공식 개발 스윕은 H100 80GB에서
[H100 14B joint sweep guide](H100_14B_JOINT_SWEEP_AGENT_GUIDE.md)를 따른다.
4090 두 장에서는 7B까지
[7B joint sweep guide](LOCAL_4090X2_7B_JOINT_SWEEP_AGENT_GUIDE.md)의
검증된 sharding 경로를 사용한다.
