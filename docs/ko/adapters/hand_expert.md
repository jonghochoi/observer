# 🤖 어댑터 — hand_expert (sharpa 파생 커스텀 PPO + nexus EvalLogger)

> 🇰🇷 임시 한글 번역본 — 원문은 [`../../adapters/hand_expert.md`](../../adapters/hand_expert.md). 정합성은 원문이 우선이며, 본 문서는 빠른 사내 공유용입니다.

`hand_expert` 패키지(커스텀 `PPO` / `ProprioAdapt` 알고리즘, `SharpaEnvWrapper`,
IsaacLab `DirectRLEnv`)로 만든 학습 레포가 이미 `make_logger(mode="dual", ...)`로 **nexus**에
메트릭을 흘리고 있고, 이제 학습 끝에 **observer 평가 파이프라인을 자동으로 돌려 그 결과를 동일한
중앙 MLflow run에 `artifacts/eval/<eval_id>/...`로 첨부**하는 루프를 닫는 워크드 예제입니다.

이 어댑터가 [`sharpa.md`](../../adapters/sharpa.md)와 다른 점은 두 가지:

1. 학습 패키지가 upstream `rl_isaaclab`이 아니라 `hand_expert` — eval 진입점이
   `hand_expert.algo.PPO` 체크포인트를 로드해야 하므로 작은 로컬 `scripts/eval_cli.py`가 필요.
2. 다운스트림 로거가 **nexus** — 통합은
   [`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md)의
   `EvalLogger.from_run_info(...)`로 수행.

## 목차

- [TL;DR](#tldr)
- [사전 준비](#사전-준비)
- [개요](#개요)
- [환경 instrumentation](#환경-instrumentation)
- [학습 레포에 추가/수정할 파일](#학습-레포에-추가수정할-파일)
- [실행](#실행)
- [통합 검증](#통합-검증)
- [Troubleshooting](#troubleshooting)
- [다음 단계](#다음-단계)

---

## TL;DR

- **수정** `train.py` — `make_logger(...)`에 `central_tracking_uri=` 추가하고, 학습 후 eval
  subprocess를 띄움.
- **추가** `scripts/eval_cli.py` — observer 컨트랙트를 만족하는 metrics 진입점. `play.py`의
  `_play_custom_algo`에서 파생하되 N 에피소드 유한 루프로 줄여 `metrics.json` + `episodes.json`을
  떨굼.
- **추가** `configs/eval_config.yaml` — observer 파이프라인 설정. 처음에는 `skip_video: true`로
  시작.
- **추가** `scripts/run_eval_and_upload.py` —
  [`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md)의
  워크드 예제를 거의 그대로 복사. nexus↔observer 글루이자 수동 재평가 CLI로도 같이 사용.

---

## 사전 준비

- `hand_expert` 학습 레포에 동작하는 `train.py`가 있을 것 (사용자가 공유한 스크립트 구조 가정 —
  `hand_expert.algo.PPO` / `ProprioAdapt`, `SharpaEnvWrapper`,
  `make_logger(mode="dual", tb_dir=train_dir, ...)`).
- 동일 Python 환경에 nexus 설치: `pip install nexus-logger` 또는
  `pip install -e /path/to/nexus`.
- 동일 Python 환경에 observer 설치: `pip install -e /path/to/observer`.
- 학습 시 `.nexus_run.json` 사이드카가 `train_dir` 아래에 작성되고 있을 것
  (`make_logger(tb_dir=...)`의 기본 동작). `scheduled_sync`로 중앙 MLflow에 복제되는 흐름은
  **그대로 유지** — 이 어댑터는 그 위에 얹는 형태.

---

## 개요

`hand_expert`는 sharpa 파생 스택입니다 — IsaacLab `DirectRLEnv` 위의
`Sharpa-InHandRotation-Direct-v0` 태스크를 `SharpaEnvWrapper`로 감싸고,
커스텀 `hand_expert.algo.ppo.PPO` / `hand_expert.algo.proprio_adapt.ProprioAdapt`로 구동합니다.
체크포인트는 algo 서브디렉토리 아래 `best.pth` / `last.pth`로 저장됩니다 (`play.py`의
`get_ppo_checkpoint_path`가 이 순서로 탐색).

| 항목 | 위치 |
|:---|:---|
| Metrics 스크립트 | `scripts/eval_cli.py` (이 어댑터에서 신규) |
| Record 스크립트 | _뒤로 미룸 — 처음에는 `skip_video: true`로 시작_ |
| 환경 instrumentation | `hand_expert/envs/.../sharpa_wave_env.py` (또는 동등 위치) |
| 로거 통합 | nexus `EvalLogger.from_run_info(target="central")` |

**두 프로세스 분리는 끝까지 유지합니다.** `train.py`가 끝나고 `simulation_app.close()`까지
호출한 다음 **subprocess로** `scripts/run_eval_and_upload.py`를 띄워, observer가 자기
metrics/record 단계용 Isaac을 **새 인터프리터에서** 다시 띄우게 합니다. 같은 프로세스에서
trainer Isaac과 observer Isaac이 GPU를 공유하면 충돌하기 쉽습니다 — 근거:
[`docs/22_EXTERNAL_LOGGER_HANDOFF.md`](../../22_EXTERNAL_LOGGER_HANDOFF.md)의
"Recommended consumption pattern".

```
train.py                                                             중앙 MLflow
   │                                                                       ▲
   │ make_logger(central_tracking_uri=...)                                 │
   │     ↳ train_dir/.nexus_run.json에 central URI 기록                    │
   │                                                                       │
   │ agent.train()                                                         │
   │ logger.close(); env.close(); simulation_app.close()                   │
   │                                                                       │
   ├──▶ subprocess: scripts/run_eval_and_upload.py                         │
   │       ├─ EvalConfig.from_yaml(configs/eval_config.yaml)              │
   │       ├─ PipelineOrchestrator.run_single(checkpoint)                  │
   │       │     ↳ subprocess: python -m scripts.eval_cli ...              │
   │       │           metrics.json + episodes.json 작성                   │
   │       ├─ result_locator.locate_results(...) → ObserverResults         │
   │       ├─ result_locator.read_metrics(...)   → 평탄화된 dict           │
   │       └─ EvalLogger.from_run_info(train_dir).upload(...) ─────────────┘
                                                  artifacts/eval/<eval_id>/…
```

---

## 환경 instrumentation

observer는 매 step의 `info` dict를 episode 종료 시점에 스냅샷해 `episodes.json`을 만듭니다.
환경은 매 `done` step에 다음 키들을 채워야 합니다 (분류 체계는
[`adapters/sharpa.md`](../../adapters/sharpa.md)의 instrumentation 섹션과 동일):

| 키 | 타입 | 설명 |
|:---|:---|:---|
| `eval/success` | bool | 에피소드 성공 여부 |
| `eval/length` | int | 종료 시점의 step 수 |
| `eval/pos_error` | float | 최종 객체 위치 오차 (m) |
| `eval/rot_error_deg` | float | 최종 객체 회전 오차 (deg) |
| `eval/init_roll_deg` | float | 초기 객체 roll (deg) |
| `eval/init_pitch_deg` | float | 초기 객체 pitch (deg) |
| `eval/init_yaw_deg` | float | 초기 객체 yaw (deg) |
| `eval/init_pos_x` | float | 초기 객체 위치 x (m) |
| `eval/init_pos_y` | float | 초기 객체 위치 y (m) |
| `eval/init_pos_z` | float | 초기 객체 위치 z (m) |
| `eval/slip_count` | int | (선택) 촉각 slip 이벤트 수 |

기존 `Sharpa-InHandRotation-Direct-v0` 환경이 `_get_observations` / `_get_rewards` /
`_get_extras` 안에서 이 키들을 이미 채우고 있다면 환경 코드 수정은 필요 없습니다. 그렇지 않다면
환경에 패치가 필요 — observer는 사후에 합성할 수 없고, 누락 시 `episodes.json`이 비며
**failure-classification과 coverage 단계가 조용히 비활성화**됩니다.

> 📖 이 `info`-dict 채널이 *왜* 존재하는지, env → eval CLI → `episodes.json` → observer
> failure-classification / coverage 까지 데이터가 *어떻게* 흐르는지 처음 본다면
> [`../../23_ENV_INSTRUMENTATION.md`](../../23_ENV_INSTRUMENTATION.md)을 먼저 읽으세요. 최소
> 패치 워크드 예제와 함께 전체 경로를 보여줍니다.

다른 단계로 넘어가기 전에 환경이 어느 키를 이미 떨구는지 확인 — 둘 중 빠른 쪽으로:

### ── 정적 확인 — env 소스 grep

Isaac 부팅 불필요. observer가 기대하는 대입 라인을 직접 찾음:

```bash
grep -nE 'self\.extras\["eval/' \
    hand_expert/envs/in_hand_rotation/direct/v0/sharpa_wave_env.py
```

10개 라인이 보이면 instrumentation 완료. 0개면 env 패치 필요.

### ── 동적 확인 — `scripts/instrumentation_probe.py`

`train.py` / `play.py`와 동일한 `AppLauncher` 부트로 한 step만 돌려, 어떤 키가 gym `info`
dict에 도착하는지 출력. 학습 레포의 `scripts/instrumentation_probe.py`에 저장:

```python
"""scripts/instrumentation_probe.py — env이 eval/* info 키를 떨구는지 확인.
실행: ./isaaclab.sh -p scripts/instrumentation_probe.py
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
    obs, _ = env.reset()
    actions = torch.zeros(
        (args.num_envs, env.action_space.shape[-1]),
        device=env.unwrapped.device,
    )
    _, _, _, _, info = env.step(actions)
    required = ["eval/success", "eval/length", "eval/pos_error", "eval/rot_error_deg",
                "eval/init_roll_deg", "eval/init_pitch_deg", "eval/init_yaw_deg",
                "eval/init_pos_x", "eval/init_pos_y", "eval/init_pos_z"]
    sample = info if isinstance(info, dict) else info[0]
    missing = [k for k in required if k not in sample]
    print("present:", sorted(set(required) - set(missing)))
    print("missing:", missing or "none")
    print("all eval/ keys:", sorted(k for k in sample if k.startswith("eval/")))
    env.close()

if __name__ == "__main__":
    main()
    sim.close()
```

`./isaaclab.sh -p scripts/instrumentation_probe.py`로 실행 — 일반 `python`은
`ModuleNotFoundError: No module named 'pxr'`로 실패합니다. IsaacLab `DirectRLEnv`는 USD
라이브러리(`pxr`)를 함께 띄우는 IsaacSim 인터프리터에서만 로드되기 때문.

---

## 학습 레포에 추가/수정할 파일

### ── 1. `train.py` — `central_tracking_uri` 전달 + eval subprocess 호출

사용자의 `train.py`는 `make_logger(...)`를 `central_tracking_uri` 없이 호출하고 있어,
`EvalLogger.from_run_info(target="central")`이 그대로는 마이그레이션 메시지로 raise합니다
([`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md)의
"Migrating trainers without central_tracking_uri" 참고). 작은 수정 세 군데:

#### ▸ a. CLI 플래그 3개 추가

```python
parser.add_argument(
    "--central_tracking_uri",
    type=str,
    default="http://nexus-server:5000",
    help=".nexus_run.json에 기록되는 중앙 MLflow URI. 학습 후 eval이 같은 run을 중앙에서 "
         "resolve할 수 있게 함. 빈 문자열이면 비활성화.",
)
parser.add_argument(
    "--no_auto_eval", action="store_true",
    help="학습 후 자동 observer 평가 + 업로드를 건너뜀.",
)
parser.add_argument(
    "--observer_config", type=str, default="configs/eval_config.yaml",
    help="학습 후 eval subprocess가 사용할 observer EvalConfig YAML 경로.",
)
```

#### ▸ b. `make_logger`에 `central_tracking_uri=` 전달

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

#### ▸ c. `main()`이 hand-off dict를 반환하고, `simulation_app.close()` 이후 eval subprocess 띄우기

eval 파이프라인은 새 Python 인터프리터에서 돌아야 합니다 — observer는 비디오 단계에서
`isaaclab.sh -p record.py`로 IsaacLab을 다시 띄우는데, 학습 인터프리터가 살아 있는 동안
이를 시도하면 Isaac이 죽습니다. `main()`이 작은 dict를 반환하게 하고, 브릿지는
`simulation_app.close()` 이후에 띄웁니다:

```python
def main(env_cfg, agent_cfg):
    # … 기존 본문, agent.train() / finally 블록까지 …
    try:
        agent.train()
    finally:
        logger.close()
        env.close()

    checkpoint_for_eval = os.path.join(train_dir, "best.pth")
    if not os.path.exists(checkpoint_for_eval):
        checkpoint_for_eval = os.path.join(train_dir, "last.pth")

    return {
        "checkpoint": checkpoint_for_eval,
        "training_output_dir": train_dir,
        "observer_config": args_cli.observer_config,
        "no_auto_eval": args_cli.no_auto_eval,
    }


if __name__ == "__main__":
    eval_handoff = main()
    simulation_app.close()
    if eval_handoff and not eval_handoff["no_auto_eval"] and \
       os.path.exists(eval_handoff["checkpoint"]):
        import subprocess
        subprocess.run(
            [
                sys.executable, "scripts/run_eval_and_upload.py",
                "--checkpoint", eval_handoff["checkpoint"],
                "--training-output-dir", eval_handoff["training_output_dir"],
                "--observer-config", eval_handoff["observer_config"],
            ],
            check=False,
        )
```

> ⚠️ `simulation_app.close()` **이전에** observer를 인라인으로 호출하지 마세요. trainer Isaac이
> GPU를 놓아준 다음에야 observer가 `isaaclab.sh` 아래에서 자기 Isaac을 띄울 수 있습니다.

### ── 2. `scripts/eval_cli.py` — observer 컨트랙트 metrics 진입점

이 어댑터의 유일한 비-trivial 신규 파일. observer가 `python -m scripts.eval_cli ...`로
호출하는 모듈입니다. `hand_expert.algo.PPO` / `ProprioAdapt` 체크포인트를 복원해 N 에피소드를
돌리고, observer가 기대하는 두 JSON을 작성합니다.

```python
"""scripts/eval_cli.py
=====================
hand_expert PPO/ProprioAdapt 체크포인트를 위한 observer 컨트랙트 eval 진입점.

--load_path의 체크포인트를 읽고 --num_episodes만큼 평가 에피소드를 돌린 뒤,
--metrics_output / --episodes_output 경로에 metrics.json + episodes.json을 씁니다.

observer가 다음과 같이 호출:
    python -m scripts.eval_cli \
        --task=<task> --load_path=<ckpt> --device=<dev> --seed=<seed> \
        --num_envs=<n> --num_episodes=<n> --headless \
        --metrics_output=<path> --episodes_output=<path> \
        --algo=PPO            # runtime.extra_eval_args로 전달

두 JSON의 스키마는 observer/docs/20_INTEGRATION_CONTRACT.md 참고.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab.app import AppLauncher

# ── 1. 인자 파싱 ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="hand_expert observer-contract eval entrypoint.")
parser.add_argument("--task", required=True)
parser.add_argument("--load_path", required=True)
parser.add_argument("--metrics_output", required=True)
parser.add_argument("--episodes_output", required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "ProprioAdapt"])
parser.add_argument("--headless", action="store_true")
parser.add_argument("--gravity", type=float, default=-9.81)
parser.add_argument("--cache", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# observer는 항상 --headless를 보냄 — 강제 적용.
args.headless = True

simulation_app = AppLauncher(args).app

# ── 2. AppLauncher 이후 import ────────────────────────────────────────
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import hand_expert.envs  # noqa: F401, E402
from hand_expert.algo.ppo.ppo import PPO  # noqa: E402
from hand_expert.algo.proprio_adapt.proprio_adapt import ProprioAdapt  # noqa: E402
from hand_expert.utils.env_wrapper import SharpaEnvWrapper  # noqa: E402


# ── 3. 에피소드 수집 ──────────────────────────────────────────────────
def _info_at(infos, idx):
    """env별 info 엔트리 추출 — gymnasium VectorEnv가 list/dict 레이아웃을 뒤집을 수 있음."""
    if isinstance(infos, dict):
        return {k: (v[idx] if hasattr(v, "__getitem__") else v) for k, v in infos.items()}
    return infos[idx]


def _episode_record(info: dict, ckpt: str) -> dict:
    """환경 info 키 → 컨트랙트 스키마 매핑. 누락 키는 KeyError로 시끄럽게 실패."""
    return {
        "checkpoint": Path(ckpt).name,
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
        "failure_mode": "unknown",   # observer step 2의 failure_classifier가 채움
    }


# ── 4. 메인 ───────────────────────────────────────────────────────────
@hydra_task_config(args.task, "agent_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):
    if hasattr(agent_cfg, "to_dict"):
        agent_cfg = agent_cfg.to_dict()

    # play.py:_play_custom_algo의 오버라이드 — eval-time 분산 끄기.
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

    if args.algo == "ProprioAdapt":
        agent = ProprioAdapt(env, output_dir="", agent_cfg=agent_cfg, create_output_dir=False)
    else:
        agent = PPO(env, output_dir="", agent_cfg=agent_cfg, create_output_dir=False)
    agent.restore_test(args.load_path)

    ep_records: list[dict] = []
    obs, _ = env.reset()
    while len(ep_records) < args.num_episodes:
        with torch.inference_mode():
            actions = agent.act_inference(obs)   # 이름이 다르면 Troubleshooting 참고
        obs, _, dones, _, infos = env.step(actions)
        done_idx = dones.nonzero().flatten().tolist() if hasattr(dones, "nonzero") else \
            [i for i, d in enumerate(dones) if d]
        for idx in done_idx:
            if len(ep_records) >= args.num_episodes:
                break
            ep_records.append(_episode_record(_info_at(infos, idx), args.load_path))

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

> 💡 정확한 추론 호출(`agent.act_inference(obs)`)은 `hand_expert.algo.PPO`의 공개 API에
> 달려 있습니다. 클래스가 `agent.policy(obs)`를 노출하거나 다른 인자 모양을 요구하면,
> `play.py:_play_custom_algo`가 `agent.restore_test(...)` 이후에 사용하는 호출을 그대로
> 따르세요 — 그쪽이 정답입니다.

### ── 3. `configs/eval_config.yaml` — observer 파이프라인 설정

[`adapters/sharpa.md`](../../adapters/sharpa.md)의 config 스탠자를 변형. 별도 `record_script`가
추가되기 전까지 `skip_video: true`로 둡니다.

```yaml
runtime:
  task: "Sharpa-InHandRotation-Direct-v0"
  eval_module: "scripts.eval_cli"
  record_script: "scripts/record.py"   # 자리표시; skip_video=False일 때만 사용
  num_envs: 8
  device: "cuda:0"
  seed: 42
  isaac_lab_path: "${ISAACLAB_PATH}/isaaclab.sh"
  extra_eval_args:
    - "--algo=PPO"          # scripts/eval_cli.py로 verbatim 전달

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

### ── 4. `scripts/run_eval_and_upload.py` — nexus↔observer 브릿지

[`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md)의
"Worked example"을 거의 그대로 복사. 아래는 편의용 사본 — 복사 전에 원문과 diff해서
upstream 변경을 반영하세요.

```python
"""scripts/run_eval_and_upload.py
================================
학습 레포에 사는 글루. observer + nexus를 import해서 둘을 묶어 줌. observer와 nexus는
서로를 import하지 않음.

학습 끝에 train.py가 자동 호출하거나, 수동 재평가 시 직접 실행:

    python scripts/run_eval_and_upload.py \
        --checkpoint logs/<exp>/<run_name>/PPO/best.pth \
        --training-output-dir logs/<exp>/<run_name>/PPO \
        --observer-config configs/eval_config.yaml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from nexus.logger.eval_logger import EvalLogger

from observer.configs.eval_config import EvalConfig
from observer.pipeline.orchestrator import PipelineOrchestrator
from observer.pipeline.result_locator import locate_results, read_metrics


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
                   help="make_logger()가 작성한 .nexus_run.json이 있는 디렉토리.")
    p.add_argument("--observer-config", required=True, type=Path)
    p.add_argument("--eval-output-dir", type=Path, default=None,
                   help="기본값은 <training-output-dir>/eval.")
    p.add_argument("--dry-run", action="store_true")
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
        dry_run=args.dry_run,
    )
    print(f"[eval] uploaded eval_id={eval_id} ({len(metrics)} metrics) "
          f"under run_name={ev.run_name} on {ev.tracking_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## 실행

학습 종료 자동 흐름 (기본값 — 끄려면 `--no_auto_eval`):

```bash
python train.py \
    --task Sharpa-InHandRotation-Direct-v0 \
    --experiment_name myexp \
    --run_name ppo_v17_seed3 \
    --central_tracking_uri http://nexus-server:5000
```

과거 임의의 run에 대한 수동 재평가 (`.nexus_run.json` 사이드카는 디스크에 남아 있음):

```bash
python scripts/run_eval_and_upload.py \
    --checkpoint logs/myexp/ppo_v17_seed3/PPO/best.pth \
    --training-output-dir logs/myexp/ppo_v17_seed3/PPO \
    --observer-config configs/eval_config.yaml
```

`EvalLogger.from_run_info(target="central")`은 `run_id`가 아니라 `run_name`으로 중앙 run을
resolve합니다 — 근거:
[`nexus/docs/32_EVAL_ARTIFACT_INGESTION.md`](https://github.com/jonghochoi/nexus/blob/main/docs/32_EVAL_ARTIFACT_INGESTION.md)의
"Why run_name, not run_id". 로컬 MLflow UUID가 회전한 뒤에도 계속 동작합니다.

---

## 통합 검증

5단계 스모크 (이 순서대로):

```bash
# 1. 사이드카가 central_tracking_uri를 담고 있는지.
python train.py --task Sharpa-InHandRotation-Direct-v0 \
    --experiment_name nexus_eval_smoke --max_agent_steps 16 --num_envs 4 --no_auto_eval
cat logs/nexus_eval_smoke/<run_name>/PPO/.nexus_run.json | python -m json.tool \
    | grep central_tracking_uri
#   → null이 아닌 중앙 URL이 찍혀야 함.

# 2. Isaac 부팅 없이 observer dry-run round-trip.
#    먼저 configs/eval_config.yaml에 dry_run: true 설정.
python scripts/run_eval_and_upload.py \
    --checkpoint logs/.../PPO/best.pth \
    --training-output-dir logs/.../PPO \
    --observer-config configs/eval_config.yaml \
    --dry-run

# 3. skip_video: true 상태로 실제 metrics round-trip (record_script 없어도 됨).
#    dry_run을 false로 돌리고, skip_video는 true 유지.
python scripts/run_eval_and_upload.py \
    --checkpoint logs/.../PPO/best.pth \
    --training-output-dir logs/.../PPO \
    --observer-config configs/eval_config.yaml
#   → 중앙 MLflow UI에서 artifacts/eval/<eval_id>/metrics.json 확인,
#     metrics 탭에서 eval/success_rate 확인, eval.last_id 태그 확인.

# 4. train.py 자동 트리거 (--no_auto_eval 빼고).
python train.py --task Sharpa-InHandRotation-Direct-v0 \
    --experiment_name nexus_eval_smoke --max_agent_steps 16 --num_envs 4

# 5. (나중에) scripts/record.py 추가 후 skip_video: false로 전환.
```

같은 체크포인트에 대해 observer가 보고하는 `success_rate`는 `play.py`가 보고하는 값과 일치해야
합니다 (에피소드 샘플링 노이즈 ±) — 같은 caveat:
[`adapters/sharpa.md`](../../adapters/sharpa.md)의 "Verifying the integration".

---

## Troubleshooting

**`EvalLogger.from_run_info(...) raised ValueError: ... has no central_tracking_uri`**

→ `train.py`를 `--central_tracking_uri` 없이 돌렸거나, kwarg가 `make_logger`까지 전달되지
않은 것. 학습을 그 플래그와 함께 다시 돌리거나, `from_run_info()`에
`tracking_uri="<central-url>"`을 명시해 오버라이드.

**`success_rate`가 `play.py` 결과와 다름**

가능성 (sharpa.md의 Troubleshooting과 동일 패턴):

- `running_mean_std`가 복원되지 않음 — `agent.restore_test(...)`가
  `play.py:_play_custom_algo`와 동일한 동작을 하는지 확인.
- 환경 `info` 키 누락 — 위 [환경 instrumentation](#환경-instrumentation) 프로브 재실행.

**`episodes.json`이 비어 있거나 길이 0**

→ 환경이 `eval/init_*` 키들을 안 떨구거나, `dones`가 한 번도 True가 아니었음 (에피소드 한도 너무
짧음). 확인:

```bash
python -c "import json; d=json.load(open('logs/.../eval/<dir>/episodes.json')); print(len(d))"
```

**`AttributeError: 'PPO' object has no attribute 'act_inference'`**

→ `scripts/eval_cli.py`의 추론 호출을 `play.py:_play_custom_algo`가 복원된 agent에 사용하는
호출로 교체. 사용자의 `play.py`는 무한 루프용 `agent.test()`를 사용하지만, `eval_cli.py`는
에피소드를 카운트해야 하므로 **단일-step 추론 호출**이 필요.

**`ModuleNotFoundError: scripts.eval_cli`**

→ `python -m scripts.eval_cli`는 `scripts/`가 `sys.path`에 있고 `__init__.py`를 가져야 함.
빈 `scripts/__init__.py`를 추가하거나, 학습 레포 루트에서 `PYTHONPATH=.`로 실행.

**중앙 MLflow에서 run을 못 찾음**

→ 해당 학습 run이 아직 `scheduled_sync`로 복제되지 않았음. 다음 sync 사이클을 기다리거나,
`EvalLogger.from_run_info(..., target="local")`로 GPU 노드 로컬 릴레이에 먼저 올림 — 다음
sync 때 중앙으로 전파됨.

---

## 다음 단계

| 문서 | 내용 |
|:---|:---|
| [`../../20_INTEGRATION_CONTRACT.md`](../../20_INTEGRATION_CONTRACT.md) | Eval / record 컨트랙트 상세 |
| [`../../21_ADAPTER_GUIDE.md`](../../21_ADAPTER_GUIDE.md) | 일반 어댑터 작성 가이드 |
| [`../../22_EXTERNAL_LOGGER_HANDOFF.md`](../../22_EXTERNAL_LOGGER_HANDOFF.md) | observer→소비자 로거 컨트랙트 |
| [`../../adapters/sharpa.md`](../../adapters/sharpa.md) | Upstream sharpa-rl-lab 어댑터 (`rl_isaaclab.scripts.eval`이 동작할 때 사용) |
| `nexus/docs/32_EVAL_ARTIFACT_INGESTION.md` | nexus `EvalLogger` API + 사이드카 스키마 |
