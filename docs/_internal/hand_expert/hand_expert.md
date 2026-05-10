# 🤖 어댑터 — hand_expert (sharpa 파생 커스텀 PPO + nexus EvalLogger)

> ⚠️ **내부 임시 문서** — 이 폴더(`docs/_internal/`)의 자료는 외부 공개 대상이
> 아니며, hand_expert 통합 테스트 검증이 끝난 뒤 삭제할 예정입니다. 공식
> 어댑터 가이드는 [`docs/adapters/sharpa.md`](../../adapters/sharpa.md) 와
> [`docs/21_ADAPTER_GUIDE.md`](../../21_ADAPTER_GUIDE.md) 를 참고하세요.

`hand_expert` 패키지(커스텀 `PPO` / `ProprioAdapt` 알고리즘, `SharpaEnvWrapper`,
IsaacLab `DirectRLEnv`) 위에 구축된 학습 레포에 대한 종단간 통합 예제입니다.
이 레포는 이미 `make_logger(mode="dual", ...)` 를 통해 **nexus** 로 메트릭을
스트리밍하고 있고, 이제 마지막 단계로 학습 종료 시 observer 의 평가 파이프라인을
실행하여 결과를 동일한 중앙 MLflow 런의 `artifacts/eval/<eval_id>/…` 에 부착하는
것이 목표입니다.

이 어댑터는 [`sharpa.md`](../../adapters/sharpa.md) 와 두 가지 점에서 다릅니다:

1. 학습 패키지가 upstream `rl_isaaclab` 이 아니라 `hand_expert` 입니다 — 평가
   엔트리포인트가 `hand_expert.algo.PPO` 체크포인트를 로드해야 하므로 작은
   로컬 `scripts/eval_cli.py` 가 필요합니다.
2. 다운스트림 로거는 **nexus** 입니다 —
   [`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md)
   의 `EvalLogger.from_run_info(...)` 를 사용해 observer 출력물을 중앙 런에 부착합니다.

## 목차

- [TL;DR](#tldr)
- [전제 조건](#전제-조건)
- [개요](#개요)
- [Env 계측](#env-계측)
- [학습 레포에 추가하거나 수정할 파일](#학습-레포에-추가하거나-수정할-파일)
- [실행](#실행)
- [통합 검증](#통합-검증)
- [트러블슈팅](#트러블슈팅)
- [다음 단계](#다음-단계)

---

## TL;DR

- **수정** `train.py` — `make_logger(...)` 에 `central_tracking_uri=` 를
  추가하고, 학습 종료 후 평가 서브프로세스를 띄웁니다.
- **추가** `scripts/eval_cli.py` — `play.py:_play_custom_algo` 에서 파생된
  observer 컨트랙트용 메트릭 emitter. 유한 에피소드 루프로 다듬어
  `metrics.json` + `episodes.json` 을 작성합니다.
- **추가** `configs/eval_config.yaml` — observer 파이프라인 설정. 처음에는
  `skip_video: true` 로 시작합니다.
- **추가** `scripts/run_eval_and_upload.py` —
  [`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md)
  의 예제를 한 가지만 바꾸어 채택합니다: CLI 플래그를 `--skip-upload` 로
  명명 (upstream 의 `--dry-run` 이 아님). observer 의 YAML `dry_run` 과
  혼동되지 않게 하기 위함입니다. 이 스크립트가 nexus↔observer 의 유일한
  접합 코드이며, 수동 재평가 CLI 로도 사용됩니다.

---

## 전제 조건

- `hand_expert` 학습 레포 + 동작하는 `train.py` (이 어댑터는 사용자
  제공 스크립트의 구조를 가정합니다 — `hand_expert.algo.PPO` /
  `ProprioAdapt`, `SharpaEnvWrapper`,
  `make_logger(mode="dual", tb_dir=train_dir, ...)`).
- 같은 Python 환경에 nexus 설치: `pip install nexus-logger` (또는
  `pip install -e /path/to/nexus`).
- 같은 Python 환경에 observer 설치: `pip install -e /path/to/observer`.
- 트레이너가 `train_dir` 에 `.nexus_run.json` 사이드카를 쓰고 있어야 함
  (`make_logger(tb_dir=...)` 의 기본 동작). 학습 레포의 `scheduled_sync` 가
  이미 런을 중앙 MLflow 로 복제하고 있으며, 그 흐름은 **변경되지 않습니다**.

---

## 개요

`hand_expert` 는 sharpa 파생 스택입니다: IsaacLab `DirectRLEnv` 위의
`Sharpa-InHandRotation-Direct-v0`, `SharpaEnvWrapper` 로 감싸고, 커스텀
`hand_expert.algo.ppo.PPO` / `hand_expert.algo.proprio_adapt.ProprioAdapt` 가
구동합니다. 체크포인트는 알고리즘 서브디렉터리에 `best.pth` / `last.pth` 로
저장됩니다 (`play.py:get_ppo_checkpoint_path` 가 탐색).

| 항목 | 위치 |
|:---|:---|
| Metrics 스크립트 | `scripts/eval_cli.py` (이 어댑터) |
| Record 스크립트 | _보류 — 처음에는 `skip_video: true` 로 시작_ |
| Env 계측 | `hand_expert/envs/.../sharpa_wave_env.py` (또는 동등 파일) |
| 로거 통합 | nexus `EvalLogger.from_run_info(target="central")` |

두 프로세스 경계는 종단간 보존됩니다: `train.py` 가 종료되어
`simulation_app` 을 닫고, 그 뒤 **서브프로세스를 spawn** 해
(`scripts/run_eval_and_upload.py`) Isaac 을 observer 의 metrics/record 단계로
다시 띄웁니다. 이렇게 하면 아직 메모리에 남은 트레이너와 observer 자신의
Isaac 런치 사이의 GPU 메모리 충돌을 피할 수 있습니다
([`docs/22_EXTERNAL_LOGGER_HANDOFF.md:#recommended-consumption-pattern`](../../22_EXTERNAL_LOGGER_HANDOFF.md#recommended-consumption-pattern)).

```
train.py                                                             central MLflow
   │                                                                       ▲
   │ make_logger(central_tracking_uri=...)                                 │
   │     ↳ writes train_dir/.nexus_run.json with central URI               │
   │                                                                       │
   │ agent.train()                                                         │
   │ logger.close(); env.close(); simulation_app.close()                   │
   │                                                                       │
   ├──▶ subprocess: scripts/run_eval_and_upload.py                         │
   │       ├─ EvalConfig.from_yaml(configs/eval_config.yaml)              │
   │       ├─ PipelineOrchestrator.run_single(checkpoint)                  │
   │       │     ↳ subprocess: python -m scripts.eval_cli ...              │
   │       │           writes metrics.json + episodes.json                 │
   │       ├─ result_locator.locate_results(...) → ObserverResults         │
   │       ├─ result_locator.read_metrics(...)   → flat dict               │
   │       └─ EvalLogger.from_run_info(train_dir).upload(...) ─────────────┘
                                                  artifacts/eval/<eval_id>/…
```

---

## Env 계측

observer 는 에피소드 종료 시점의 step별 `info` dict 를 슬라이싱해서
`episodes.json` 을 만듭니다. env 는 매 `done` step 마다 다음 키들을
채워야 합니다 (분류 체계는
[`adapters/sharpa.md:#-env-instrumentation-gym-info-keys`](../../adapters/sharpa.md#-env-instrumentation-gym-info-keys)
와 동일):

| 키 | 타입 | 설명 |
|:---|:---|:---|
| `eval/success` | bool | 에피소드 성공 플래그 |
| `eval/length` | int | 종료 시점의 step 수 |
| `eval/pos_error` | float | 최종 물체 위치 오차 (m) |
| `eval/rot_error_deg` | float | 최종 물체 회전 오차 (deg) |
| `eval/init_roll_deg` | float | 초기 물체 roll (deg) |
| `eval/init_pitch_deg` | float | 초기 물체 pitch (deg) |
| `eval/init_yaw_deg` | float | 초기 물체 yaw (deg) |
| `eval/init_pos_x` | float | 초기 물체 위치 x (m) |
| `eval/init_pos_y` | float | 초기 물체 위치 y (m) |
| `eval/init_pos_z` | float | 초기 물체 위치 z (m) |
| `eval/slip_count` | int | (선택) 촉각 슬립 이벤트 |

기존 `Sharpa-InHandRotation-Direct-v0` env 가 `_get_observations` /
`_get_rewards` / `_get_extras` 에서 이 키들을 채우고 있다면 env 수정은
필요 없습니다. 그렇지 않다면 env 를 패치해 키들을 emit 하도록 만들어야
합니다 — observer 는 사후에 합성할 수 없으며, 키가 빠지면 `episodes.json`
이 비게 되어(silently) observer 의 failure-classification 과 coverage
단계가 비활성화됩니다.

> 📖 이 `info`-dict 채널이 *왜* 존재하고, 데이터가 env → eval CLI →
> `episodes.json` → observer 의 failure-classification / coverage 단계로
> *어떻게* 흘러가는지 처음 보는 경우라면, 먼저
> [`../../23_ENV_INSTRUMENTATION.md`](../../23_ENV_INSTRUMENTATION.md)
> 를 읽어보세요. 최소 패치 예제와 함께 전체 경로를 따라갑니다.

다른 작업을 하기 전에, env 가 무엇을 emit 하고 있는지 먼저 확인합니다.
어떤 방식이든 더 빠른 쪽을 사용하면 됩니다 — 동적 점검은 Isaac 부팅이
필요합니다:

### ── 정적 점검 — env 소스 grep

Isaac 부팅 없이, observer 가 기대하는 할당을 찾습니다:

```bash
grep -nE 'self\.extras\["eval/' \
    hand_expert/envs/in_hand_rotation/direct/v0/sharpa_wave_env.py
```

10줄(`eval/<key>` 당 1줄) → 계측 완료. 0줄 → env 패치 필요.

### ── 동적 점검 — `scripts/instrumentation_probe.py`

`train.py` / `play.py` 의 `AppLauncher` 부팅을 모방하고, step 한 번을
실행한 뒤 어떤 키가 gym `info` dict 에 도착하는지 보고합니다. 학습
레포의 `scripts/instrumentation_probe.py` 에 저장:

```python
"""scripts/instrumentation_probe.py — does the env emit the eval/* info keys?
Run via:  ./isaaclab.sh -p scripts/instrumentation_probe.py
"""
import argparse, sys
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", default="Sharpa-InHandRotation-Direct-v0")
p.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(p)
args, hydra_args = p.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
sim = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config
import hand_expert.envs  # noqa: F401
from hand_expert.utils.env_wrapper import SharpaEnvWrapper

@hydra_task_config(args.task, "agent_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    env = SharpaEnvWrapper(env, clip_actions=1.0)
    obs, *_ = env.reset()  # SharpaEnvWrapper may return more than 2 values
    actions = torch.zeros(
        (args.num_envs, env.action_space.shape[-1]),
        device=env.unwrapped.device,
    )
    step_ret = env.step(actions)
    info = step_ret[-1]  # last element is the info dict regardless of arity
    required = ["eval/success", "eval/length", "eval/pos_error", "eval/rot_error_deg",
                "eval/init_roll_deg", "eval/init_pitch_deg", "eval/init_yaw_deg",
                "eval/init_pos_x", "eval/init_pos_y", "eval/init_pos_z"]
    sample = info if isinstance(info, dict) else info[0]
    missing = [k for k in required if k not in sample]
    print("present:", sorted(set(required) - set(missing)))
    print("missing:", missing or "none")
    # 키별 shape — observer 는 per-env 값에 대해 shape `(num_envs,)` 을 기대.
    # 그 외(0 size, done count 로 packing 등)는 env 가 특정 termination
    # 타입에서만 키를 emit 한다는 뜻이며, eval_cli 의 `_info_at` 헬퍼가
    # 이를 관용적으로 처리해야 함.
    print(f"num_envs = {args.num_envs}; shapes:")
    for k in sorted(k for k in sample if k.startswith("eval/")):
        v = sample[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        print(f"  {k}: {shape}")
    env.close()

if __name__ == "__main__":
    main()
    sim.close()
```

`./isaaclab.sh -p scripts/instrumentation_probe.py` 로 실행 — 일반
`python` 은 `ModuleNotFoundError: No module named 'pxr'` 로 실패합니다.
IsaacLab `DirectRLEnv` 가 USD 라이브러리를 함께 들고 오는 IsaacSim
인터프리터 안에서만 로드되기 때문입니다.

### ── 적용 예제 — sharpa 스타일 env 에 `eval/*` 추가하기

sharpa 파생 env 에서 흔히 보는 시작 상태: `_get_rewards()` 가
`self.extras` 에 트레이너의 TensorBoard/MLflow 텔레메트리용
**스칼라 `.mean()` 리덕션** 을 채우고 있고, `eval/*` 채널은 아직 없는
상태. 이 상태에서 `scripts/instrumentation_probe.py` 출력은
`infos has N keys; 0 match 'eval/*'` 로 나오고,
`scripts/eval_cli.py` 는 0개 에피소드를 기록합니다 (heartbeat 가
`dones>0, skipped==dones` 로 보임).

env 파일(예:
`hand_expert/envs/in_hand_rotation/direct/v0/sharpa_wave_env.py`) 에
대한 두 개의 결합된 편집으로 트레이너를 전혀 건드리지 않고 루프를
닫을 수 있습니다 — 새 `eval/*` 키들은 per-env 텐서이므로, PPO 의
"스칼라 전용" 텔레메트리 필터가 이를 무시하고 기존 학습 메트릭은
그대로 유지됩니다.

#### ▸ a. 에피소드 시작 시 초기 자세 캡처

`eval/init_*` 는 **리셋 직후의** 물체 자세가 필요합니다. 표준 훅은
`_reset_idx(env_ids)` 이지만, 거기 손대지 않고도 다음 `_get_dones()`
시작 시점의 `episode_length_buf == 0` 술어로 "방금 리셋된" env 를
감지할 수 있습니다 (Isaac Lab 은 step 종료 시점에 버퍼를 tick).
`self._refresh_lab()` 직후, `_get_dones` 시작 부분에 다음 블록을
넣습니다:

```python
import math
from isaaclab.utils.math import euler_xyz_from_quat  # top of file

# ── observer eval: capture init pose for envs that just reset ────
just_reset = (self.episode_length_buf == 0)
if not hasattr(self, "_init_obj_pos"):
    self._init_obj_pos     = torch.zeros((self.num_envs, 3), device=self.device)
    self._init_obj_rpy_deg = torch.zeros((self.num_envs, 3), device=self.device)
if just_reset.any():
    self._init_obj_pos[just_reset] = self.object_pos[just_reset]
    r, p, y = euler_xyz_from_quat(self.object_rot[just_reset])
    self._init_obj_rpy_deg[just_reset] = torch.stack(
        [r, p, y], dim=-1
    ) * (180.0 / math.pi)
```

`hasattr` 가드로 자체 완결됩니다 — 별도의 `__init__` 편집 불필요.
버퍼는 step 간 유지되고, env 가 리셋될 때만 다시 쓰입니다.

#### ▸ b. `_get_dones` 끝에서 per-env `eval/*` 텐서 emit

`height_reset` / `time_out` 계산 후 per-env eval 채널을 추가합니다.
성공 휴리스틱에 주의: 이 task 에서는 에이전트가 전체 에피소드 동안
물체를 안정적으로 잡고 있으면 ("`time_out` 도달 + `height_reset` 없음")
"성공"으로 봅니다. task 의미가 다르면 그에 맞는 술어를 고르세요.

```python
# ── observer eval contract: per-env tensors, NO .mean() ──────────
body_axis_world = quat_rotate(self.object_rot, self.rot_axis)
alignment_pe = (body_axis_world * self.rot_axis).sum(-1).clamp(-1.0, 1.0)
rot_err_deg = torch.acos(alignment_pe) * (180.0 / math.pi)

default_pos = self.object_default_pose[:, :3].to(self.device)
pos_err = torch.norm(self.object_pos - default_pos, dim=-1)

self.extras["eval/success"]        = time_out & (~height_reset)         # bool[N]
self.extras["eval/length"]         = self.episode_length_buf.clone()    # int[N]
self.extras["eval/pos_error"]      = pos_err                            # float[N], m
self.extras["eval/rot_error_deg"]  = rot_err_deg                        # float[N], deg
self.extras["eval/init_roll_deg"]  = self._init_obj_rpy_deg[:, 0]
self.extras["eval/init_pitch_deg"] = self._init_obj_rpy_deg[:, 1]
self.extras["eval/init_yaw_deg"]   = self._init_obj_rpy_deg[:, 2]
self.extras["eval/init_pos_x"]     = self._init_obj_pos[:, 0]
self.extras["eval/init_pos_y"]     = self._init_obj_pos[:, 1]
self.extras["eval/init_pos_z"]     = self._init_obj_pos[:, 2]
```

#### ▸ c. 검증

probe 를 다시 실행해 shape 를 확인:

```bash
./isaaclab.sh -p scripts/instrumentation_probe.py
```

이제 모든 `eval/*` 라인이 `shape=(num_envs,)` (또는 설정한
`--num_envs` 와 일치하는 `(N,)`) 로 나와야 합니다. smoke 의 step 3
를 다시 실행하면 eval_cli heartbeat 가 `dones=N skipped=0` 로 나와야
합니다 (첫 step 에서 reset 과 done 이 겹치는 일회성 race 는 예외).

두 가지 부수적인 사실:

- **트레이너는 무영향.** PPO 의 `play_steps()` 는 스칼라 타입으로
  infos 를 필터링합니다 (`isinstance(v, torch.Tensor) and len(v.shape) == 0`);
  per-env eval 텐서는 절대 TB/MLflow 스트림에 들어가지 않습니다.
- **성공은 task 별 정의.** `time_out & ~height_reset` 는 "물체를
  떨어뜨리지 않는" 게 목표인 in-hand rotation 에서만 맞습니다. 명시적인
  목표가 있는 task (예: 누적 회전이 N 도 이상)는 그 신호를 per-env
  버퍼에 추적해 `eval/success` 에 넣어야 합니다.

---

## 학습 레포에 추가하거나 수정할 파일

### ── 1. `train.py` — `central_tracking_uri` 전달 + 평가 서브프로세스 spawn

사용자가 제공한 `train.py` 는 `make_logger(...)` 를
`central_tracking_uri` 없이 호출합니다. 이 경우
`EvalLogger.from_run_info(target="central")` 가 문서화된 마이그레이션
메시지로 실패합니다
([`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md:#migrating-trainers-without-central_tracking_uri`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md#migrating-trainers-without-central_tracking_uri)).
세 곳을 작게 고칩니다:

#### ▸ a. CLI 플래그 3개 추가

```python
parser.add_argument(
    "--central_tracking_uri",
    type=str,
    default="http://nexus-server:5000",
    help="Central MLflow URI written into .nexus_run.json so the post-training eval can "
         "resolve the same run on central MLflow. Pass an empty string to disable.",
)
parser.add_argument(
    "--no_auto_eval", action="store_true",
    help="Skip the automatic post-training observer evaluation + upload.",
)
parser.add_argument(
    "--observer_config", type=str, default="configs/eval_config.yaml",
    help="Path to observer's EvalConfig YAML used by the post-training eval subprocess.",
)
parser.add_argument(
    "--observer_handoff", type=Path, default=None,
    help="If set, write the post-training eval hand-off as JSON here. An outer shell "
         "wrapper (`scripts/train_and_eval.sh`) consumes this file and runs the bridge.",
)
```

#### ▸ b. `make_logger` 에 `central_tracking_uri=` 전달

```python
logger = make_logger(
    mode="dual",
    tb_dir=train_dir,
    run_name=run_name,
    tracking_uri=args_cli.tracking_uri,
    central_tracking_uri=args_cli.central_tracking_uri or None,   # ← NEW
    experiment_name=experiment_name,
    agent_params=agent_cfg,
    env_params=env_cfg,
    tags={
        "task": args_cli.task,
        "train": args_cli.algo,
        "hand": "sharpa_wave_22dof",
        "hand_side": args_cli.hand_side,
    },
)
```

#### ▸ c. JSON 센티넬 파일을 통한 hand-off + 외부 셸 래퍼가 평가 실행

가장 직관적인 패턴인 "`train.py` 의 `__main__` 블록에서 평가를
서브프로세스로 spawn" 은 두 가지 실행 경계 때문에 신뢰성이 떨어집니다:

1. `@hydra_task_config` 가 `main()` 을 `@hydra.main(...)` 으로 감싸기
   때문에 **내부 함수의 반환값이 폐기됩니다** — 즉
   `eval_handoff = main()` 은 항상 `None`.
2. 일부 Isaac Sim 빌드에서 `simulation_app.close()` 가 `os._exit()` 을
   호출합니다. 그 뒤에 오는 모든 것 (인라인 `subprocess.run(...)` 포함)
   은 **절대 실행되지 않습니다**.

견고한 패턴: `main()` 이 hand-off 를 모듈 레벨 dict 에 보관 → 그 dict 를
`--observer_handoff` 로 전달된 경로의 JSON 파일로 작성 → 그 다음
`simulation_app.close()`. 얇은 셸 래퍼 (`scripts/train_and_eval.sh`)
가 `train.py` 가 완전히 종료된 후에만 브리지를 호출합니다 — 엄격한
프로세스 경계, `close()` 와의 race 없음.

```python
import json
from pathlib import Path

_EVAL_HANDOFF: dict | None = None   # module-level; main() writes via `global`

@hydra_task_config(args.task, "agent_cfg_entry_point")
def main(env_cfg, agent_cfg):
    global _EVAL_HANDOFF
    # … existing body up through the agent.train() / finally block …
    try:
        agent.train()
    finally:
        logger.close()
        env.close()

    checkpoint_for_eval = os.path.join(train_dir, "best.pth")
    if not os.path.exists(checkpoint_for_eval):
        checkpoint_for_eval = os.path.join(train_dir, "last.pth")

    _EVAL_HANDOFF = {
        "checkpoint": checkpoint_for_eval,
        "training_output_dir": train_dir,
        "observer_config": args_cli.observer_config,
        "no_auto_eval": args_cli.no_auto_eval,
    }


if __name__ == "__main__":
    main()

    # close() 전에 hand-off 를 먼저 쓴다 — close() 가 os._exit() 으로
    # 빠지면서 그 이후를 통째로 건너뛸 수 있기 때문.
    if (
        _EVAL_HANDOFF
        and not _EVAL_HANDOFF["no_auto_eval"]
        and os.path.exists(_EVAL_HANDOFF["checkpoint"])
        and args_cli.observer_handoff is not None
    ):
        args_cli.observer_handoff.parent.mkdir(parents=True, exist_ok=True)
        args_cli.observer_handoff.write_text(json.dumps(_EVAL_HANDOFF))
        print(f"[INFO] eval handoff written to {args_cli.observer_handoff}", flush=True)

    simulation_app.close()
```

`scripts/train_and_eval.sh` — 두-프로세스 흐름을 한 줄 명령으로 묶어
주는 외부 래퍼:

```bash
#!/usr/bin/env bash
# Run training, then (if a handoff was written) run observer eval + upload.
# Strict sequence: train fully exits before eval starts — no GPU race,
# no os._exit() weirdness from simulation_app.close().
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HANDOFF="$(mktemp -t observer_handoff.XXXXXX.json)"
trap 'rm -f "$HANDOFF"' EXIT

python "$REPO_ROOT/scripts/train.py" "$@" --observer_handoff "$HANDOFF"

if [ ! -s "$HANDOFF" ]; then
    echo "[train_and_eval] no handoff written — skipping eval"
    exit 0
fi

CKPT=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$HANDOFF")
DIR=$(python  -c 'import json,sys; print(json.load(open(sys.argv[1]))["training_output_dir"])' "$HANDOFF")
CFG=$(python  -c 'import json,sys; print(json.load(open(sys.argv[1]))["observer_config"])' "$HANDOFF")

python "$REPO_ROOT/scripts/run_eval_and_upload.py" \
    --checkpoint "$CKPT" \
    --training-output-dir "$DIR" \
    --observer-config "$CFG"
```

```bash
chmod +x scripts/train_and_eval.sh
```

> 💡 왜 Python 오케스트레이터가 아니라 셸 래퍼인가? 핵심은 평가를
> 시작하기 **전에** `train.py` 를 — Isaac 의 모든 atexit / GPU
> 정리까지 — 완전히 종료시키는 것입니다. 셸은 그 경계를 명시적이고
> 저렴하게 만듭니다. 양쪽의 종료 코드가 자연스럽게 전파되고,
> `set -e` 가 실패를 잡아내며, 저장된 handoff JSON 으로 평가만 다시
> 돌리는 것도 간단합니다. Python 부모는 `os._exit()` 과 프로세스
> 그룹 복잡성을 다시 끌어들입니다.

> ⚠️ `simulation_app.close()` 전에 인라인으로 observer 를 호출하지
> **마세요**. 트레이너의 Isaac 인스턴스가 GPU 를 해제한 다음에야
> observer 가 `isaaclab.sh` 로 자기 인스턴스를 띄울 수 있습니다.

### ── 2. `scripts/eval_cli.py` — observer 컨트랙트 메트릭 emitter

이게 유일하게 자명하지 않은 신규 파일입니다. observer 가
`python -m scripts.eval_cli ...` 로 호출하는 모듈입니다.
`hand_expert.algo.PPO` / `ProprioAdapt` 체크포인트를 복원하고, N 개
에피소드를 돌리고, observer 가 기대하는 두 JSON 을 작성합니다.

```python
"""scripts/eval_cli.py
=====================
Observer-contract eval entrypoint for hand_expert PPO/ProprioAdapt checkpoints.

Reads a checkpoint at --load_path, runs --num_episodes evaluation episodes, and writes
metrics.json + episodes.json at the paths given by --metrics_output / --episodes_output.

Invoked by observer as:
    python -m scripts.eval_cli \
        --task=<task> --load_path=<ckpt> --device=<dev> --seed=<seed> \
        --num_envs=<n> --num_episodes=<n> --headless \
        --metrics_output=<path> --episodes_output=<path> \
        --algo=PPO            # forwarded via runtime.extra_eval_args

The schema for both JSON files is documented in
observer/docs/20_INTEGRATION_CONTRACT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

# ── 1. Argument parsing ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Observer-contract eval entrypoint for hand_expert.")
parser.add_argument("--task", required=True)
parser.add_argument("--load_path", required=True)
parser.add_argument("--metrics_output", required=True)
parser.add_argument("--episodes_output", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "ProprioAdapt"])
parser.add_argument("--gravity", type=float, default=-9.81)
parser.add_argument("--cache", type=str, default=None)
# `--device` 와 `--headless` 는 AppLauncher 소유 — 수동으로 등록하면
# `AppLauncher._check_argparser_config_params` 가
# `ValueError: ... already has the field 'device'` 로 실패. AppLauncher 가
# 추가하도록 둘 것.
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# Metrics 단계에서는 항상 headless 로 강제 — observer 는 늘 이 플래그를
# 넘기지만, 사용자가 standalone 으로 호출할 때를 대비해 여기서 한 번 더
# `args.headless = True` 로 덮어쓴다.
args.headless = True

simulation_app = AppLauncher(args).app

# ── 2. Late imports (must follow AppLauncher.app) ─────────────────────
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import hand_expert.envs  # noqa: F401, E402
from hand_expert.algo.ppo.ppo import PPO  # noqa: E402
from hand_expert.algo.proprio_adapt.proprio_adapt import ProprioAdapt  # noqa: E402
from hand_expert.utils.env_wrapper import SharpaEnvWrapper  # noqa: E402


# ── 3. Episode collection ─────────────────────────────────────────────
def _info_at(infos, idx):
    """idx 번째 env 의 info 항목, 또는 env 가 이 done 인덱스에 대해
    에피소드 레벨 eval 데이터를 emit 하지 않은 경우 None. 일부 Isaac
    Lab env 는 특정 termination 타입(예: task 성공/실패는 채우지만
    timeout-reset 은 채우지 않음)에서만 `eval/*` 키를 채우거나, 길이가
    `num_envs` 와 다른 packed 텐서를 사용한다 — 둘 다 env idx 로
    그대로 인덱싱하면
    `IndexError: index N is out of bounds for dimension 0 with size 0` 로
    터진다. 스킵하면 부분 정보를 반환하고, `eval/success` 누락은
    "이 done 은 실제 에피소드 경계가 아니다 — 레코드에서 빼라" 는 신호.
    """
    if not isinstance(infos, dict):
        try:
            return infos[idx]
        except (IndexError, KeyError):
            return None
    out = {}
    for k, v in infos.items():
        if not hasattr(v, "__getitem__"):
            out[k] = v
            continue
        try:
            out[k] = v[idx]
        except (IndexError, KeyError):
            pass  # key not populated for this idx — silently drop
    if "eval/success" not in out:
        return None
    return out


def _episode_record(info: dict, ckpt: str) -> dict:
    """env info 키 → 컨트랙트 스키마 매핑. 키가 없으면 KeyError 가 시끄럽게 터진다."""
    return {
        "success": bool(info["eval/success"]),
        "length": int(info["eval/length"]),
        "final_pos_error_m": float(info["eval/pos_error"]),
        "final_rot_error_deg": float(info["eval/rot_error_deg"]),
        "init_roll_deg": float(info["eval/init_roll_deg"]),
        "init_pitch_deg": float(info["eval/init_pitch_deg"]),
        "init_yaw_deg": float(info["eval/init_yaw_deg"]),
        "init_pos_x": float(info["eval/init_pos_x"]),
        "init_pos_y": float(info["eval/init_pos_y"]),
        "init_pos_z": float(info["eval/init_pos_z"]),
        "slip_count": int(info.get("eval/slip_count", 0)),
        "failure_mode": "unknown",   # observer's failure_classifier fills this in step 2
    }


# ── 4. Main ───────────────────────────────────────────────────────────
@hydra_task_config(args.task, "agent_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    if hasattr(agent_cfg, "to_dict"):
        agent_cfg = agent_cfg.to_dict()

    # play.py:_play_custom_algo 의 override 와 일치 — eval 시점의 분산은 끈다.
    if args.algo == "ProprioAdapt":
        agent_cfg["network"]["proprio_adapt"] = True
    torch.manual_seed(args.seed)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    env_cfg.reset_random_quat = False
    env_cfg.randomize_pd_gains = False
    env_cfg.randomize_friction = True
    env_cfg.randomize_com = False
    env_cfg.randomize_mass = False
    env_cfg.randomize_joint_pos_offset = False
    env_cfg.sim.gravity = (0, 0, args.gravity)
    env_cfg.gravity_curriculum = False
    if args.cache:
        env_cfg.grasp_cache_path = args.cache

    agent_cfg["seed"] = args.seed
    agent_cfg["device"] = args.device
    agent_cfg["algorithm"]["num_actors"] = args.num_envs
    agent_cfg["test"] = True

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    clip_actions = agent_cfg["algorithm"].get("clip_actions", 1.0)
    env = SharpaEnvWrapper(env, clip_actions=clip_actions)

    # Eval 은 read-only 지만, PPO/ProprioAdapt 는 무조건 `output_dir` 아래
    # 사이드 파일을 쓴다. 사용자 logs/ 에 쓰레기가 남지 않도록 tempdir 사용.
    # `output_dir=` 가 유일한 생성자 입력 — 표준 시그니처는
    # play.py:_play_custom_algo 참조. hand_expert 버전이 추가 인자를
    # 받는다면 여기서 맞춰서 적용.
    agent_scratch = tempfile.mkdtemp(prefix="observer_eval_agent_")
    if args.algo == "ProprioAdapt":
        agent = ProprioAdapt(env, output_dir=agent_scratch, agent_cfg=agent_cfg)
    else:
        agent = PPO(env, output_dir=agent_scratch, agent_cfg=agent_cfg)
    agent.restore_test(args.load_path)
    agent.set_eval()  # model + running_mean_std (+ ProprioAdapt 의 sa_mean_std) → eval 모드
                      # 안 그러면 normalizer 가 eval 데이터로 통계를 계속 갱신함

    # hand_expert 의 표준 `test()` body 미러링. 두 알고리즘은 바깥쪽 루프를
    # 공유하지만 inference `input_dict` 의 shape 가 다르다 — 아래와
    # 어긋나면 `hand_expert/algo/{ppo,proprio_adapt}/*.py` 를 교차 확인할 것:
    #   - `SharpaEnvWrapper.reset()` 은 obs dict 를 직접 반환 (단일 반환,
    #     `(obs, info)` tuple 아님).
    #   - `SharpaEnvWrapper.step()` 은 gym old-API 의 4-tuple
    #     `(obs_dict, rewards, dones, infos)` 반환 — gymnasium 의 5-tuple 아님.
    #   - inference 는 `agent.model.act_inference(input_dict)` (`.model` 속성).
    #     agent 자체가 아님.
    #   - PPO inference dict:           {obs, priv_info}.
    #     ProprioAdapt inference dict:  {obs, proprio_hist} — `priv_info` 는
    #     proprio history 로부터 학습된 latent 로 대체되므로, 여기서
    #     `priv_info` 를 넘기면 `ActorCritic.act_inference` 의 다른
    #     (보정되지 않은) 분기로 조용히 라우팅된다.
    #   - `obs` 는 `agent.running_mean_std` 로 정규화; `proprio_hist` 는
    #     추가로 `agent.sa_mean_std` 를 거치며 정규화 전에 detach 된다
    #     (`ProprioAdapt.test()` 와 동일).
    #   - Action 출력은 학습과 동일하게 `[-1, 1]` 로 clamp.
    ep_records: list[dict] = []
    obs_dict = env.reset()
    # Progress heartbeat: 목표 에피소드의 ~10% 마다, 그리고 done 은 많은데
    # eval 데이터가 없는 경우 (`len(ep_records)` 가 느리게 증가) 를
    # 위한 step-rate fallback 으로 200 sim step 마다. `dones=` 와
    # `skipped=` 카운터는 "수많은 step 후에도 0 에피소드" 처럼 보이는
    # 두 가지 실패 모드를 구별:
    #   - `dones=0` (수백 step 후) → env 가 여기서 termination 안 함;
    #     `max_episode_length` 와 env 의 termination 조건을 확인.
    #   - `dones>0` 이고 `skipped==dones` → env 가 done 은 emit 하지만
    #     해당 termination 타입에서 `eval/*` 를 채우지 않음; env 를
    #     계측 (또는 부분 기록 수용).
    # observer 가 Popen 으로 stdout 을 라인별로 스트리밍하므로
    # `flush=True` 가 필수.
    log_every_eps = max(1, args.num_episodes // 10)
    last_logged_eps = 0
    sim_step = 0
    total_dones = 0
    total_skipped = 0
    first_done_dumped = False
    print(f"[eval_cli] collecting {args.num_episodes} episodes "
          f"(num_envs={args.num_envs})", flush=True)
    while len(ep_records) < args.num_episodes:
        with torch.inference_mode():
            if args.algo == "ProprioAdapt":
                input_dict = {
                    "obs": agent.running_mean_std(obs_dict["obs"]),
                    "proprio_hist": agent.sa_mean_std(obs_dict["proprio_hist"].detach()),
                }
            else:
                input_dict = {
                    "obs": agent.running_mean_std(obs_dict["obs"]),
                    "priv_info": obs_dict["priv_info"],
                }
            mu = agent.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)

        obs_dict, _rewards, dones, infos = env.step(mu)
        sim_step += 1

        n_dones = (
            int(dones.sum().item()) if hasattr(dones, "sum")
            else sum(int(d) for d in dones)
        )
        if n_dones > 0:
            total_dones += n_dones
            if not first_done_dumped:
                # 첫 done 배치에서 한 번만 구조 dump. 세 가지 레이아웃을 구별:
                #   (1) eval/* 키를 가진 dict → 이상적인 Isaac Lab 패턴.
                #   (2) eval/* 키 없는 dict → env 가 다른 prefix 에 extras 를
                #       채우거나 wrapper 가 떼어냈음.
                #   (3) per-env dict 의 list/tuple → gym VectorEnv 구식;
                #       `_info_at` 가 `infos[idx]` 로 처리.
                # type + key sample + eval/* shape 를 보여주면 가설별로
                # 따로 돌려보지 않고도 세 가지를 한 번에 가른다.
                print(f"[eval_cli] first dones at step {sim_step}: "
                      f"n_dones={n_dones}", flush=True)
                print(f"[eval_cli]   type(infos) = {type(infos).__name__}",
                      flush=True)
                if isinstance(infos, dict):
                    all_keys = sorted(str(k) for k in infos.keys())
                    eval_keys = [k for k in all_keys if k.startswith("eval/")]
                    print(f"[eval_cli]   infos has {len(all_keys)} keys; "
                          f"{len(eval_keys)} match 'eval/*'", flush=True)
                    print(f"[eval_cli]   first 20 keys: {all_keys[:20]}",
                          flush=True)
                    for k in eval_keys:
                        v = infos[k]
                        shape = (
                            tuple(v.shape) if hasattr(v, "shape")
                            else type(v).__name__
                        )
                        print(f"[eval_cli]   {k}: shape={shape}", flush=True)
                elif hasattr(infos, "__getitem__") and hasattr(infos, "__len__"):
                    print(f"[eval_cli]   len(infos) = {len(infos)}", flush=True)
                    first_done = dones.nonzero(as_tuple=False).flatten().tolist()[0]
                    try:
                        sample = infos[first_done]
                        print(f"[eval_cli]   infos[{first_done}] type = "
                              f"{type(sample).__name__}", flush=True)
                        if isinstance(sample, dict):
                            sk = sorted(str(k) for k in sample.keys())
                            ek = [k for k in sk if k.startswith("eval/")]
                            print(f"[eval_cli]   sample has {len(sk)} keys; "
                                  f"{len(ek)} match 'eval/*'", flush=True)
                            print(f"[eval_cli]   first 20 keys: {sk[:20]}",
                                  flush=True)
                    except Exception as e:
                        print(f"[eval_cli]   infos[{first_done}] indexing "
                              f"failed: {e!r}", flush=True)
                else:
                    print(f"[eval_cli]   repr(infos)[:200] = "
                          f"{repr(infos)[:200]}", flush=True)
                first_done_dumped = True

        done_idx = dones.nonzero(as_tuple=False).flatten().tolist()
        for idx in done_idx:
            if len(ep_records) >= args.num_episodes:
                break
            info_at = _info_at(infos, idx)
            if info_at is None:
                total_skipped += 1
                continue  # done without episode-level eval data — skip
            ep_records.append(_episode_record(info_at, args.load_path))

        if (
            len(ep_records) - last_logged_eps >= log_every_eps
            or sim_step % 200 == 0
        ):
            print(
                f"[eval_cli] step {sim_step:5d} | "
                f"episodes {len(ep_records)}/{args.num_episodes} | "
                f"dones={total_dones} skipped={total_skipped}",
                flush=True,
            )
            last_logged_eps = len(ep_records)

    success_rate = sum(int(e["success"]) for e in ep_records) / max(1, len(ep_records))
    metrics = {
        "checkpoint": Path(args.load_path).name,
        "num_episodes": len(ep_records),
        "success_rate": success_rate,
        "episode_length_mean": sum(e["length"] for e in ep_records) / max(1, len(ep_records)),
        "object_pose_error_mm": 1000.0 * (
            sum(e["final_pos_error_m"] for e in ep_records) / max(1, len(ep_records))
        ),
    }

    Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.episodes_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_output, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(args.episodes_output, "w") as f:
        json.dump(ep_records, f, indent=2)
    print(f"[eval_cli] wrote {len(ep_records)} episodes (sr={success_rate:.3f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

> 💡 위 inference 루프의 레퍼런스는 `hand_expert/algo/ppo/ppo.py` 의
> `PPO.test()` 와 `hand_expert/algo/proprio_adapt/proprio_adapt.py` 의
> `ProprioAdapt.test()` 입니다. 두 알고리즘은 바깥 루프는 공유하지만
> `input_dict` shape 가 다릅니다 — PPO 는 `priv_info`, ProprioAdapt 는
> `proprio_hist`. 사용자 트리에서 normalizer 속성 이름이 다르거나
> `act_inference` 가 `agent.model` 이 아닌 다른 곳에 있다면, 해당
> `test()` 본문 안에서 사용하는 정확한 속성 접근을 그대로 미러링하세요
> — 그게 트레이너가 의존하는 표준 평가 경로입니다.
> `play.py:_play_custom_algo` 가 있다면 보통 무한 루프 케이스용으로
> `agent.test()` 를 감싸는 정도입니다.

### ── 3. `configs/eval_config.yaml` — observer 파이프라인 설정

[`adapters/sharpa.md:#config-stanza`](../../adapters/sharpa.md#config-stanza)
에서 변형. 별도 `record_script` 가 추가되기 전까지는
`skip_video: true` 를 유지합니다.

```yaml
runtime:
  task: "Sharpa-InHandRotation-Direct-v0"
  eval_module: "scripts.eval_cli"
  record_script: "scripts/record.py"   # placeholder; only used when skip_video=False
  num_envs: 8
  device: "cuda:0"
  seed: 42
  isaac_lab_path: "${ISAACLAB_PATH}/isaaclab.sh"
  extra_eval_args:
    - "--algo=PPO"          # forwarded verbatim to scripts/eval_cli.py

metrics:
  num_eval_episodes: 50

video:
  resolution: [1280, 720]
  fps: 30
  concat_views: true

cameras:
  - {name: "front", eye: [0.6, 0.0, 0.4], target: [0.0, 0.0, 0.05], record_steps: 600}
  - {name: "side",  eye: [0.0, 0.6, 0.4], target: [0.0, 0.0, 0.05], record_steps: 600}

skip_video: true
skip_report: false
dry_run: false
```

### ── 4. `scripts/run_eval_and_upload.py` — nexus↔observer 브리지

[`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md:#worked-example-a-training-repo-glue-script`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md#worked-example-a-training-repo-glue-script)
에서 변형. upstream 스니펫과 의도적으로 다른 점은 CLI 플래그 이름
하나뿐 — upstream 은 `--dry-run`, 이 어댑터는 `--skip-upload` 로
명명합니다. observer 자신의 `dry_run` (Isaac 서브프로세스를
게이팅하는 YAML 필드, 아래 [두 dry-run 류 플래그](#-두-dry-run-류-플래그) 참조)
과 명확히 공존시키기 위함.

```python
"""scripts/run_eval_and_upload.py
================================
Lives in the training repo. Imports observer + nexus and glues them together; observer and
nexus do not depend on each other.

Invoked automatically by train.py at the end of training, or manually for re-evaluation:

    python scripts/run_eval_and_upload.py \
        --checkpoint logs/<exp>/<run_name>/PPO/best.pth \
        --training-output-dir logs/<exp>/<run_name>/PPO \
        --observer-config configs/eval_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from nexus.logger.eval_logger import EvalLogger

from observer.configs.eval_config import EvalConfig
from observer.pipeline.orchestrator import PipelineOrchestrator
from observer.pipeline.result_locator import locate_results, read_metrics

# observer 자신의 INFO 로그 (예: "[1/5] Metrics collection (headless)") 를
# 스트리밍되는 eval_cli heartbeat 와 함께 보이게 한다. 이 설정 없이는
# WARNING+ 만 표시됨.
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _eval_commit() -> str:
    env = os.environ.get("OBSERVER_COMMIT")
    if env:
        return env
    try:
        import observer
        repo = Path(observer.__file__).resolve().parent.parent
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--training-output-dir", required=True, type=Path,
                   help="Dir holding .nexus_run.json from make_logger().")
    p.add_argument("--observer-config", required=True, type=Path)
    p.add_argument("--eval-output-dir", type=Path, default=None,
                   help="Defaults to <training-output-dir>/eval.")
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="Run observer end-to-end and stop before any nexus interaction — no "
             "`EvalLogger.from_run_info()` and no upload. Useful for offline smoke "
             "tests where the trainer has no `.nexus_run.json` sidecar yet. "
             "Orthogonal to observer's YAML `dry_run`, which gates the Isaac subprocess.",
    )
    args = p.parse_args()

    eval_root = args.eval_output_dir or (args.training_output_dir / "eval")
    eval_root.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig.from_yaml(str(args.observer_config))
    orch = PipelineOrchestrator(config=cfg, output_root=eval_root)
    result = orch.run_single(args.checkpoint)
    if not result.success:
        print(f"[eval] downstream run_single failed: {result.error_msg}", file=sys.stderr)
        return 2

    obs = locate_results(eval_root, result_dir=result.output_dir)
    metrics = read_metrics(obs)
    if not metrics:
        print("[eval] no metrics produced — aborting upload", file=sys.stderr)
        return 3

    if args.skip_upload:
        print(f"[eval] --skip-upload set; skipping nexus EvalLogger entirely. "
              f"Observer artifacts at: {result.output_dir} ({len(metrics)} metrics ready)")
        return 0

    tags = {
        "observer_commit": _eval_commit(),
        "checkpoint": args.checkpoint.name,
        "algo": os.environ.get("HAND_EXPERT_ALGO", "PPO"),
    }

    ev = EvalLogger.from_run_info(args.training_output_dir, target="central")
    eval_id = ev.upload(
        eval_dir=result.output_dir,
        metrics=metrics,
        tags=tags,
        generate_index=True,
    )
    print(f"[eval] uploaded eval_id={eval_id} ({len(metrics)} metrics) "
          f"under run_name={ev.run_name} on {ev.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## 실행

학습 종료 자동 흐름 (`scripts/train_and_eval.sh` 가 학습을 돌린 뒤
새 프로세스로 평가를 실행; 평가 단계를 건너뛰려면 `--no_auto_eval`):

```bash
bash scripts/train_and_eval.sh \
    --task Sharpa-InHandRotation-Direct-v0 \
    --experiment_name myexp \
    --run_name ppo_v17_seed3 \
    --central_tracking_uri http://nexus-server:5000
```

학습만 실행 (`--no_auto_eval` 와 무관하게 평가 단계는 완전히 생략):

```bash
python scripts/train.py \
    --task Sharpa-InHandRotation-Direct-v0 \
    --experiment_name myexp \
    --run_name ppo_v17_seed3 \
    --central_tracking_uri http://nexus-server:5000
```

과거 런에 대한 수동 재평가 (`.nexus_run.json` 사이드카는 그대로 보존됨):

```bash
python scripts/run_eval_and_upload.py \
    --checkpoint logs/myexp/ppo_v17_seed3/PPO/best.pth \
    --training-output-dir logs/myexp/ppo_v17_seed3/PPO \
    --observer-config configs/eval_config.yaml
```

`EvalLogger.from_run_info(target="central")` 은 중앙 런을 `run_id` 가
아닌 `run_name` 으로 해소합니다 —
[`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md:#why-run_name-not-run_id`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md#-why-run_name-not-run_id)
참조 — 그래서 로컬 MLflow UUID 가 회전된 뒤에도 한참 동안 동작합니다.

---

## 통합 검증

### ── 두 dry-run 류 플래그

이 파이프라인에는 두 개의 독립적인 게이트가 있습니다. 별칭 관계가
아니며 — 각각 다른 단계를 토글합니다. 아래 검증 단계들은 의도적으로
두 플래그를 조합합니다.

| 노브 | 위치 | 게이트 대상 | 사용 용도 |
|:---|:---|:---|:---|
| `dry_run: true/false` | `configs/eval_config.yaml` (observer) | observer 의 Isaac metrics + record 서브프로세스; 실제 메트릭 대신 `_dummy_metrics()` 로 대체 | Isaac/GPU 부팅 없이 파이프라인 배관 검증 |
| `--skip-upload` | `scripts/run_eval_and_upload.py` (이 브리지) | nexus 단계 전체 — `EvalLogger.from_run_info()` 가 호출**되지 않음** + 업로드 없음. observer 는 여전히 디스크에 진짜 아티팩트 생성 | `.nexus_run.json` 이 아직 없을 때 완전 오프라인 smoke |

조합:

| YAML `dry_run` | `--skip-upload` | 결과 |
|:---:|:---:|:---|
| `false` | absent | 풀 런 — 진짜 Isaac 평가 + 진짜 중앙 업로드 |
| `false` | present | 진짜 Isaac 평가, 아티팩트는 디스크에만 저장; nexus 와 인터랙션 없음 |
| `true`  | absent | 더미 메트릭이 업로드됨 (보통 원하는 결과 아님) |
| `true`  | present | 완전 오프라인 smoke — **아래 step 2 가 이 조합 사용** |

### ── 다섯 단계 smoke

순서대로 실행:

```bash
# 1. Sidecar 가 central_tracking_uri 를 들고 있는지.
python train.py --task Sharpa-InHandRotation-Direct-v0 \
    --experiment_name nexus_eval_smoke --max_agent_steps 16 --num_envs 4 --no_auto_eval
cat logs/nexus_eval_smoke/<run_name>/PPO/.nexus_run.json | python -m json.tool \
    | grep central_tracking_uri
#   → null 이 아닌 중앙 URL 이 보여야 함.

# 2. 완전 오프라인 smoke — Isaac 없음, nexus 인터랙션 없음.
#    먼저 configs/eval_config.yaml 에서 dry_run: true 로 설정. --skip-upload 는
#    추가로 EvalLogger.from_run_info() 와 업로드 BOTH 를 우회하므로
#    .nexus_run.json 이 없어도 작동.
python scripts/run_eval_and_upload.py \
    --checkpoint logs/.../PPO/best.pth \
    --training-output-dir logs/.../PPO \
    --observer-config configs/eval_config.yaml \
    --skip-upload

# 3. skip_video: true 로 진짜 메트릭 round-trip (record_script 불필요).
#    dry_run: false 로 변경; skip_video: true 유지; --skip-upload 빼고 publish.
python scripts/run_eval_and_upload.py \
    --checkpoint logs/.../PPO/best.pth \
    --training-output-dir logs/.../PPO \
    --observer-config configs/eval_config.yaml
#   → 중앙 MLflow UI 에서 artifacts/eval/<eval_id>/metrics.json,
#     metrics 탭의 eval/success_rate, eval.last_id 태그 확인.

# 4. 셸 래퍼로 종단간 자동 트리거. train.py 가 완전히 끝난 뒤 JSON
#    handoff 를 읽고 observer 브리지를 새 프로세스로 실행.
bash scripts/train_and_eval.sh --task Sharpa-InHandRotation-Direct-v0 \
    --experiment_name nexus_eval_smoke --max_agent_steps 16 --num_envs 4

# 5. (나중에) scripts/record.py 추가하고 skip_video: false 로 전환.
```

observer 의 `success_rate` 는 같은 체크포인트에 대한 `play.py` 결과와
일치해야 합니다 (± 에피소드 샘플링 노이즈) — 동일 caveat 가
[`adapters/sharpa.md:#verifying-the-integration`](../../adapters/sharpa.md#verifying-the-integration)
에도 있습니다.

---

## 트러블슈팅

**`EvalLogger.from_run_info(...) raised ValueError: ... has no central_tracking_uri`**

→ `train.py` 가 `--central_tracking_uri` 없이 실행됐거나 (혹은 그
kwarg 가 `make_logger` 까지 전달되지 않음). 해결: 플래그를 넣어
재학습하거나, `from_run_info()` 에 `tracking_uri="<central-url>"` 을
넘겨 override.

**`FileNotFoundError: .nexus_run.json not found at logs/.../PPO/.nexus_run.json`**

→ 이 체크포인트를 만든 트레이너가 nexus 의 `make_logger(tb_dir=...)` 를
호출하지 않아서, `EvalLogger.from_run_info()` 가 중앙 런을 해소할 때
읽는 사이드카가 없는 상태. 의도에 따라 두 가지 해결:
- Smoke 테스트: 브리지에 `--skip-upload` 를 넘겨 nexus 단계를 통째로
  우회 — observer 의 오프라인 아티팩트는 여전히 `<eval-output-dir>/`
  에 저장됨.
- 진짜 업로드:
  [step 1 의 `train.py`](#-1-trainpy--central_tracking_uri-전달--평가-서브프로세스-spawn)
  편집을 적용해 사이드카가 작성되도록 재학습한 뒤, 브리지를
  `--skip-upload` 없이 다시 실행.

**`success_rate` 가 `play.py` 결과와 다름**

가능한 원인 (sharpa.md 트러블슈팅과 동일):

- `running_mean_std` 미복원 — `agent.restore_test(...)` 가
  `play.py:_play_custom_algo` 와 일치하는지 확인.
- env `info` 키 누락 —
  [Env 계측](#env-계측) 의 instrumentation probe 재실행.

**`episodes.json` 이 비었거나 길이 0**

→ env 가 `eval/init_*` 키를 채우지 않거나, `dones` 가 한 번도 `True`
가 안 됐다는 뜻 (에피소드 길이 너무 짧음). 다음으로 확인:

```bash
python -c "import json; d=json.load(open('logs/.../eval/<dir>/episodes.json')); print(len(d))"
```

**`AttributeError: 'PPO' object has no attribute 'act_inference'`**

→ hand_expert 에서는 `act_inference` 가 agent 클래스가 아니라 내부
`ActorCritic` 모델에 있습니다. `agent.model.act_inference(input_dict)`
를 사용하고, 알고리즘의 표준 `test()` body 와 일치하도록
`input_dict` 를 빌드:

| 알고리즘 | `input_dict` 키 |
|:---|:---|
| `PPO` | `{"obs": agent.running_mean_std(obs_dict["obs"]), "priv_info": obs_dict["priv_info"]}` |
| `ProprioAdapt` | `{"obs": agent.running_mean_std(obs_dict["obs"]), "proprio_hist": agent.sa_mean_std(obs_dict["proprio_hist"].detach())}` |

PPO 의 dict 를 ProprioAdapt agent (또는 그 반대) 에 넘기면
`ActorCritic.act_inference` 의 트레이너가 사용한 분기와 다른 분기로
라우팅됩니다. 정책은 여전히 액션을 만들지만 보정되지 않은 head 가
구동하므로, `success_rate` 는 명시적 에러 없이 무너집니다. 위
템플릿이 정확히 이 이유로 `args.algo` 에 따라 분기합니다.

**`ValueError: too many values to unpack (expected 2)` (`obs, _ = env.reset()`)**

→ `SharpaEnvWrapper.reset()` 은 gymnasium `(obs, info)` tuple 이 아닌
**observation dict 를 직접 반환**합니다. tuple 풀기는 dict 키를
순회하므로 dict 가 3개 이상 (예: `obs`, `priv_info`, `proprio`) 이면
이 에러가 납니다. 해결책은 **풀기를 하지 않는 것** —
`obs_dict = env.reset()` 으로 받고 `obs_dict["obs"]` /
`obs_dict["priv_info"]` 로 접근. `env.step()` 에도 동일 적용:
래퍼는 gym old-API 4-tuple `(obs_dict, rewards, dones, infos)` 를
반환합니다 (gymnasium 의 5-tuple 아님). 빠른 점검:

```bash
grep -n "env\.reset()\|env\.step(" hand_expert/algo/ppo/ppo.py
```

**`running_mean_std` 통계 drift / eval `success_rate` 가 `play.py` 보다 낮음**

→ `agent.set_eval()` 을 빠뜨린 증상. 호출하지 않으면 `RunningMeanStd`
가 train 모드에 머물면서 eval 시점의 관측으로 통계를 계속 갱신해,
정책이 보는 입력이 왜곡됩니다. 항상 `agent.restore_test(...)` 직후,
inference 루프 시작 전에 `agent.set_eval()` 호출.

**`IndexError: index N is out of bounds for dimension 0 with size 0` (`_info_at` 에서)**

→ env 가 일부 termination 타입에서만 `eval/*` info 키를 채움 (보통:
실제 task 성공/실패만 채우고 timeout-reset 은 비움) — `dones[idx]=True`
인 step 이 완전히 비어있는 `eval/*` 텐서에 떨어질 수 있다는 뜻.
위의 `_info_at` 헬퍼가 그런 경우 `None` 을 반환해 루프가 done
인덱스를 건너뛰도록 처리 — 만약 모든 키에 대해 무조건 `v[idx]` 를
하는 옛날 eval_cli 버전을 복붙했다면, 회복력 있는 형태와 짝이 되는
`if info_at is None: continue` 가드로 업데이트하세요.

env 의 실제 emit 패턴을 확인하려면
[`scripts/instrumentation_probe.py`](#-동적-점검--scriptsinstrumentation_probepy)
를 실행 — 위에 갱신된 형태는 모든 `eval/*` 키의 shape 를 출력합니다.
`(num_envs,)` 가 아닌 것은 env 내부에서 이벤트별 필터링이 일어나고
있다는 뜻. 그것들도 풀 에피소드로 카운트하고 싶다면, env 가 task
완료뿐 아니라 모든 termination 타입에서 `eval/*` 를 채우도록 수정.

**`TypeError: __init__() got an unexpected keyword argument 'create_output_dir'`**

→ 일부 버전의 hand_expert `PPO` / `ProprioAdapt` 생성자는
`(env, output_dir, agent_cfg)` 만 받습니다 (`create_output_dir`
kwarg 없음). kwarg 를 빼고 일회용 `output_dir` 을 넘기세요 — 위
템플릿은 `tempfile.mkdtemp(prefix="observer_eval_agent_")` 를 사용해
eval 아티팩트가 트레이너 `logs/` 에 절대 남지 않도록 합니다. 항상
사용자 트리의 `play.py:_play_custom_algo` 와 생성자 시그니처를
교차 확인.

**`FileNotFoundError: ... best.pth` (`restore_test` 에서)**

→ eval 모듈의 `@hydra_task_config` 데코레이터가 `main()` 실행 전에
CWD 를 hydra run dir 로 바꾸므로, 상대 경로 `--load_path` 가 더 이상
해소되지 않습니다. 이 fix 가 적용된 observer ≥ 는 절대 경로를
넘깁니다; 구버전 observer 를 쓰는 중이라면 업그레이드 전까지
`python scripts/run_eval_and_upload.py --checkpoint
$(realpath logs/.../best.pth) ...` 로 회피.

**`ModuleNotFoundError: scripts.eval_cli`**

→ `python -m scripts.eval_cli` 는 `scripts/` 가 `sys.path` 에 있고
`__init__.py` 를 가지고 있어야 합니다. 빈 `scripts/__init__.py` 추가
또는 `PYTHONPATH=.` 로 학습 레포 루트에서 실행.

**`ValueError: The passed ArgParser object already has the field 'device'`**

→ `scripts/eval_cli.py` 가 `--device` (또는 `--headless`) 를 수동으로
등록한 뒤 `AppLauncher.add_app_launcher_args(parser)` 를 호출 — 같은
플래그를 추가하려고 시도. 둘 다 AppLauncher 소유; 수동
`parser.add_argument("--device", ...)` 와 `--headless` 라인을
제거하고 AppLauncher 가 추가하도록 두세요. `parser.parse_known_args()`
이후에도 `args.device` 와 `args.headless` 는 채워집니다.

**중앙 MLflow 에서 런을 찾을 수 없음**

→ 학습 런이 아직 `scheduled_sync` 로 복제되지 않은 상태. 다음 동기화
사이클까지 기다리거나, `EvalLogger.from_run_info(...)` 에
`target="local"` 을 넘겨 GPU-노드 로컬 릴레이에 먼저 떨어뜨리세요;
다음 동기화에 중앙으로 전파됩니다.

---

## 다음 단계

| 문서 | 내용 |
|:---|:---|
| [`../../20_INTEGRATION_CONTRACT.md`](../../20_INTEGRATION_CONTRACT.md) | Eval / record 컨트랙트 세부 |
| [`../../21_ADAPTER_GUIDE.md`](../../21_ADAPTER_GUIDE.md) | 일반 어댑터 작성 가이드 |
| [`../../22_EXTERNAL_LOGGER_HANDOFF.md`](../../22_EXTERNAL_LOGGER_HANDOFF.md) | observer→consumer-logger 컨트랙트 |
| [`../../adapters/sharpa.md`](../../adapters/sharpa.md) | Upstream sharpa-rl-lab 어댑터 (`rl_isaaclab.scripts.eval` 가 동작할 때 사용) |
| `nexus/docs/32_EVAL_ARTIFACT_INGESTION.md` | nexus `EvalLogger` API + 사이드카 스키마 |
