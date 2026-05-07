# 🧪 02 · 설치·검증·첫 결과물 생성 가이드

> 바이브 코딩만 하고 실제로 실행해 본 적 없을 때 읽는 실전 가이드.
> 설치 → 검증 → dry_run → 실제 eval 결과물 순서로 진행한다.

## 목차

- [Step 0 — 전제 조건 확인](#step-0--전제-조건-확인)
- [Step 1 — 설치](#step-1--설치)
- [Step 2 — 설치 검증 (observer doctor)](#step-2--설치-검증-observer-doctor)
- [Step 3 — dry_run으로 파이프라인 연기](#step-3--dry_run으로-파이프라인-연기)
- [Step 4 — dry_run 결과물 확인](#step-4--dry_run-결과물-확인)
- [Step 5 — 실제 eval 실행 (Isaac 필요)](#step-5--실제-eval-실행-isaac-필요)
- [Step 6 — 결과물 디렉터리 구조](#step-6--결과물-디렉터리-구조)
- [Troubleshooting](#troubleshooting)

---

## Step 0 — 전제 조건 확인

| 항목 | 최소 버전 | 확인 명령 |
|:---|:---|:---|
| Python | 3.10+ | `python --version` |
| pip | 최신 권장 | `pip --version` |
| ffmpeg | any | `ffmpeg -version` |
| Git | any | `git --version` |

ffmpeg가 없으면 비디오 단계에서 실패한다. 미리 설치:

```bash
sudo apt install ffmpeg          # Ubuntu/Debian
brew install ffmpeg              # macOS
```

---

## Step 1 — 설치

### ── 방법 A: setup.sh 사용 (권장 — 격리 venv 자동 생성)

```bash
cd /path/to/observer             # 레포 루트

# 격리된 venv를 ~/.observer/venv 에 생성하고 observer 설치
bash setup.sh --venv

# 이후 매 세션마다 활성화
source ~/.observer/activate.sh
```

`--alias` 옵션을 추가하면 `~/.bashrc`에 `observer-activate` 단축어가 등록된다:

```bash
bash setup.sh --venv --alias
# 이후: observer-activate
```

### ── 방법 B: 직접 설치 (기존 환경 / Docker / CI)

```bash
cd /path/to/observer             # 레포 루트

# numpy, pyyaml, matplotlib은 pyproject.toml에 선언돼 있어
# pip install -e . 한 번으로 함께 설치된다
pip install -e .

# 선택: 실험 트래킹
pip install tensorboard

# 선택: tactile 오버레이
pip install opencv-python
```

> ⚠️ `pip install -e .`는 **레포 루트**에서 실행해야 한다.
> `observer/` 하위 디렉터리에서 실행하면 `pyproject.toml`을 찾지 못한다.

설치 후 다음이 동작하면 성공:

```bash
observer --help
```

---

## Step 2 — 설치 검증 (observer doctor)

```bash
observer doctor
```

정상 출력 예시:

```
══════════════════════════════════════════ OBSERVER Doctor ══
Environment
  ✓ Python 3.11.4
  ✓ import numpy
  ✓ import yaml
  ✓ import matplotlib
  ! tensorboard not installed — pip install tensorboard (optional)
  ✓ ffmpeg in PATH
  ✓ nvidia-smi works

Configuration
  ✗ runtime.task is empty — fill it in eval_config.yaml
  ✗ runtime.eval_module is empty

══════════════════════════════════════════════════════════════
  👁 Issues detected. Fix the ✗ items above.
```

**`✗` 항목 두 개는 정상이다** — `eval_config.yaml`을 아직 안 채웠기 때문이다.
핵심 의존성(`✓ numpy`, `✓ yaml`, `✓ matplotlib`, `✓ ffmpeg`)이 모두 통과했으면 다음 단계로 진행할 수 있다.

`--skip-runtime` 옵션으로 config 검사를 건너뛸 수 있다:

```bash
observer doctor --skip-runtime
```

---

## Step 3 — dry_run으로 파이프라인 연기

**dry_run은 Isaac / 사용자 eval 스크립트 없이 전체 파이프라인을 통과시킨다.**
더미 메트릭을 주입해 랭킹·커버리지·리포트 코드 경로를 검증한다.

### ── 더미 체크포인트 파일 만들기

dry_run은 `.pth` 파일이 존재하기만 하면 된다 (내용 무관):

```bash
mkdir -p runs/test_exp
touch runs/test_exp/model_1000.pth
touch runs/test_exp/model_2000.pth
touch runs/test_exp/model_3000.pth
```

### ── dry_run 실행

```bash
python eval_runner.py \
    --checkpoint_dir runs/test_exp/ \
    --dry_run \
    --skip_video \
    --no_tracking
```

`--dry_run` — Isaac 서브프로세스 생략, 더미 메트릭 주입  
`--skip_video` — ffmpeg 비디오 단계 생략  
`--no_tracking` — TensorBoard 로깅 비활성화

### ── 랭킹까지 포함한 완전 dry_run

```bash
python eval_runner.py \
    --checkpoint_dir runs/test_exp/ \
    --dry_run \
    --skip_video \
    --no_tracking \
    --auto_select \
    --select_weights hardware_safe \
    --deploy_top_k 2
```

오류 없이 `Scan Complete`가 뜨면 파이프라인 전체가 정상이다.

---

## Step 4 — dry_run 결과물 확인

기본 출력 경로는 `eval_results/`:

```bash
ls eval_results/
```

```
eval_results/
├── eval_report.html                     ← 브라우저에서 열기
├── best/
│   ├── rank01__model_3000.pth           원본에 대한 심볼릭 링크
│   ├── rank02__model_2000.pth
│   └── selection_meta.json
└── test_exp__model_3000__20260506_143021/
    ├── eval_config_snapshot.yaml
    ├── metrics.json                     ← 더미 메트릭 확인
    └── episodes.json                    ← dry_run에선 없을 수 있음
```

**metrics.json 확인:**

```bash
cat eval_results/test_exp__model_3000__*/metrics.json | python -m json.tool
```

```json
{
    "checkpoint": "model_3000.pth",
    "success_rate": 0.847,
    "slip_events_per_episode": 1.23,
    "energy_J_per_episode": 2.91,
    "note": "dry_run dummy data"
}
```

**HTML 리포트 열기:**

```bash
# 로컬 머신
open eval_results/eval_report.html          # macOS
xdg-open eval_results/eval_report.html     # Linux

# 원격 서버라면 scp로 가져오거나 Python HTTP 서버 사용
python -m http.server 8080 --directory eval_results/
# 브라우저에서 http://<서버IP>:8080/eval_report.html
```

---

## Step 5 — 실제 eval 실행 (Isaac 필요)

Isaac Lab / 사용자 eval 스크립트가 준비된 경우에만 진행한다.

### ── 5-1. eval_config.yaml 설정

```bash
cp configs/eval_config.yaml configs/my_exp.yaml
```

`configs/my_exp.yaml`의 최소 필수 항목:

```yaml
runtime:
  task: "Isaac-MyTask-v0"              # 필수 — 본인 태스크 ID
  eval_module: "my_pkg.scripts.eval"   # 필수 — python -m으로 실행될 모듈
  record_script: "my_pkg/scripts/record.py"  # skip_video: true면 비워도 됨
  num_envs: 4
  device: "cuda:0"
  seed: 42
  isaac_lab_path: "${ISAACLAB_PATH}/isaaclab.sh"
  extra_eval_args: []

metrics:
  num_eval_episodes: 50

skip_video: true      # 처음엔 true로 시작, 검증 후 false로 전환
skip_report: false
dry_run: false
```

eval 스크립트가 만족해야 할 계약:

```
docs/20_INTEGRATION_CONTRACT.md
```

핵심만 요약: eval 스크립트는 `--metrics_output`과 `--episodes_output` 경로에
`metrics.json`과 `episodes.json`을 출력해야 한다.

### ── 5-2. 설정 검증

```bash
observer doctor --config configs/my_exp.yaml
```

모든 `✓`가 뜨면 실행 준비 완료.

### ── 5-3. 단일 체크포인트 실행

```bash
python eval_runner.py \
    --checkpoint runs/my_exp/model_5000.pth \
    --config configs/my_exp.yaml \
    --skip_video \
    --no_tracking
```

### ── 5-4. 디렉터리 전체 스윕

```bash
python eval_runner.py \
    --checkpoint_dir runs/my_exp/ \
    --config configs/my_exp.yaml \
    --skip_video \
    --auto_select \
    --select_weights balanced
```

### ── 5-5. 재귀 + 최신 체크포인트만 + 비디오 포함

```bash
python eval_runner.py \
    --checkpoint_dir runs/ \
    --recursive \
    --latest_only \
    --config configs/my_exp.yaml \
    --auto_select \
    --select_weights hardware_safe \
    --deploy_top_k 2
```

---

## Step 6 — 결과물 디렉터리 구조

```
eval_results/
├── eval_report.html                     ← 최종 HTML 리포트 (시작점)
├── best/
│   ├── rank01__model_5000.pth           최우수 체크포인트 심볼릭 링크
│   └── selection_meta.json              랭킹 점수 상세
│
└── my_exp__model_5000__20260506_150000/
    ├── eval_config_snapshot.yaml        재현성용 설정 스냅샷
    ├── metrics.json                     에피소드 집계 지표 (8개)
    ├── episodes.json                    에피소드별 원본 데이터
    ├── camera_poses.json                비디오 촬영에 쓰인 카메라 설정
    ├── coverage/
    │   ├── success_heatmap.png          roll × pitch 성공률 히트맵
    │   ├── coverage_scatter.png         실패 모드별 분산 플롯
    │   └── pose_histogram.png           초기 포즈 샘플링 분포
    └── videos/                          (skip_video: false일 때)
        ├── front.mp4
        ├── front_left.mp4
        ├── side.mp4
        ├── rear.mp4
        ├── top.mp4
        └── combined_grid.mp4            5개 시점 2×3 그리드
```

### ── 지표 빠른 해석

| 지표 | 이 값이 크면 |
|:---|:---|
| `success_rate` | 높을수록 좋음 |
| `slip_events_per_episode` | 하드웨어 위험 — 낮출 것 |
| `joint_velocity_rms` | 제어 불안정 신호 |
| `contact_force_rms` | 과도한 쥐는 힘 |
| `energy_J_per_episode` | 에너지 비효율 |
| `object_pos_error_mm` | 목표 포즈 도달 실패 |

---

## Troubleshooting

**`observer` 명령어를 찾을 수 없음**

```bash
pip install -e .   # 레포 루트에서 재실행
which observer     # 경로 확인
```

**`No .pth files found` 오류**

체크포인트 파일이 실제로 `.pth` 확장자를 가지는지 확인:

```bash
find runs/ -name "*.pth"
```

**dry_run인데도 오류 발생**

`configs/eval_config.yaml`의 `runtime.task`와 `runtime.eval_module`이 비어있어도
dry_run은 통과해야 한다. `EvalConfig.from_yaml` 단계에서 오류가 나면
YAML 문법 오류일 가능성이 높다 — 들여쓰기(스페이스 vs 탭) 확인.

**`coverage/` 디렉터리가 비어있거나 없음**

`episodes.json`이 생성되지 않으면 커버리지 분석이 건너뛰어진다.
eval 스크립트가 `--episodes_output` 경로에 파일을 쓰는지 단독 실행으로 확인:

```bash
python -m <eval_module> \
    --task=... --load_path=... --num_episodes=5 \
    --metrics_output=/tmp/m.json \
    --episodes_output=/tmp/ep.json \
    --headless

cat /tmp/ep.json | python -m json.tool | head -40
```

**GUI 없는 서버에서 비디오 촬영 실패**

```bash
Xvfb :99 -screen 0 1920x1080x24 &
DISPLAY=:99 python eval_runner.py --checkpoint_dir runs/ --config configs/my_exp.yaml
```

**`success_rate`가 0.0으로 나옴**

observer는 정책을 직접 로드하지 않는다. 체크포인트 로딩과 observation normalization
복원은 사용자 eval 스크립트의 책임이다. 스크립트를 단독 실행해서 로딩이 정상인지 먼저 확인한다.

---

*실전 테스팅 가이드 · OBSERVER v0.1.0*
