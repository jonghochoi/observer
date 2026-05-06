# 👁️ 01 · OBSERVER 온보딩 가이드

## 목차

- [TL;DR](#tldr)
- [1. 이 프로젝트는 무엇을 하나요?](#1-이-프로젝트는-무엇을-하나요)
- [2. 핵심 개념 한눈에 보기](#2-핵심-개념-한눈에-보기)
- [3. 폴더/파일 구조](#3-폴더파일-구조)
- [4. 최초 셋업 (10분 코스)](#4-최초-셋업-10분-코스)
- [5. 첫 실행 (드라이런부터)](#5-첫-실행-드라이런부터)
- [6. 출력물 읽는 법](#6-출력물-읽는-법)
- [7. 체크포인트 랭킹 프리셋](#7-체크포인트-랭킹-프리셋)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

---

## TL;DR

- OBSERVER = 체크포인트를 **자동으로 평가·분류·랭킹·기록**하는 파이프라인
- 입력: 체크포인트 파일(들) / 출력: 메트릭 JSON, 실패 분포, 히트맵, 비디오, HTML 리포트
- 한 줄 요약: **"영상 보지 말고, 데이터 읽자."**

---

## 1. 이 프로젝트는 무엇을 하나요?

**OBSERVER**는 고DOF dexterous manipulation 정책의
체크포인트를 **자동으로 평가·분류·랭킹·기록**하는 파이프라인입니다.

### ── 왜 필요한가?

- 체크포인트는 빠르게 쌓이지만 사람이 눈으로 비교하는 건 한계가 있다
- `success_rate` 하나로는 slip, 관절 속도 스파이크, 에너지 과다 사용 등의 **하드웨어 위험 신호**를 놓친다
- 수동 리뷰에는 선택 편향·최신 편향·피로가 낀다

> 💡 자세한 배경: [`../00_PRINCIPLES.md`](../00_PRINCIPLES.md)

### ── 7단계 자동화

| # | 단계 | 출력 |
|:---:|:---|:---|
| 1 | 📦 **메트릭 수집** | `metrics.json` — 8개 정량 지표 |
| 2 | 🔍 **실패 모드 분류** | 6-class 규칙 체인, Isaac 의존 없음 |
| 3 | 🗺️ **상태 커버리지 분석** | roll × pitch 성공 히트맵 |
| 4 | 🎬 **다각도 비디오 기록** | 5 시점 + 2×3 그리드 mp4 |
| 5 | 📡 **실험 트래킹** | W&B / TensorBoard 자동 감지 |
| 6 | 🏆 **다목적 랭킹** | 가중 점수: 성공률, slip, 에너지, 포즈 오차 |
| 7 | 📄 **HTML 리포트** | 차트·파이·비디오·히트맵 포함 |

---

## 2. 핵심 개념 한눈에 보기

| 컴포넌트 | 설명 |
|:---|:---|
| 📦 **MetricsCollector** | 에피소드 스텝마다 성공률·접촉력·slip 등 8개 지표 수집 |
| 🔍 **FailureModeClassifier** | 우선순위 규칙 체인으로 에피소드별 실패 유형 분류 (학습 데이터 불필요) |
| 🗺️ **StateCoverageAnalyzer** | roll × pitch 초기 포즈 공간에서 어디가 약한지 히트맵 |
| 🎬 **CameraController + VideoRecorder** | 5개 시점 + 2×3 그리드 비디오 저장 |
| 📡 **ExperimentTracker** | W&B / TensorBoard 자동 감지 및 로깅 |
| 🏆 **CheckpointSelector** | 다목적 가중치 점수로 top-k 체크포인트 선정 |
| 📄 **ReportGenerator** | 차트·파이·비디오·히트맵이 포함된 self-contained HTML |
| 🔄 **PipelineOrchestrator** | 체크포인트 1개당 위 단계들을 조율 |

---

## 3. 폴더/파일 구조

```
observer/
├── eval_runner.py              진입점 (설치 시 `observer` CLI로 노출)
├── brand.py                    콘솔 배너/브랜딩
├── configs/
│   ├── eval_config.py          Config 데이터클래스
│   └── eval_config.yaml        ← 실험마다 편집할 파일
├── pipeline/
│   ├── orchestrator.py         체크포인트 단위 조율자
│   ├── metrics_collector.py    스텝별 지표 수집
│   ├── failure_classifier.py   규칙 기반 실패 모드 분류
│   ├── state_coverage.py       초기 포즈 커버리지 분석
│   ├── experiment_tracker.py   W&B / TensorBoard 연동
│   └── auto_select.py          다목적 점수 체크포인트 선정
├── isaac/
│   ├── camera_controller.py    Isaac Sim 뷰포트 제어 (유틸)
│   └── recorder.py             Replicator 기반 비디오 캡처 (유틸)
└── docs/
    ├── 20_INTEGRATION_CONTRACT.md  ← 프레임워크 붙일 때 핵심
    └── adapters/sharpa.md          ← sharpa-rl-lab 예시
```

> 📌 **자주 보게 될 파일 TOP 3**
> 1. `configs/eval_config.yaml` — 실험마다 값 바꿔가며 작업
> 2. `docs/20_INTEGRATION_CONTRACT.md` — 본인 프레임워크 붙일 때 이 계약만 맞추면 됨
> 3. `eval_runner.py` — CLI 플래그 확인

---

## 4. 최초 셋업 (10분 코스)

```bash
# 1) 핵심 의존성
pip install numpy pyyaml matplotlib

# 2) ffmpeg (비디오 인코딩)
sudo apt install ffmpeg

# 3) 선택: 실험 트래킹
pip install wandb tensorboard

# 4) 선택: tactile 오버레이
pip install opencv-python

# 5) observer CLI 설치
pip install -e .
```

### ── 설치 확인

```bash
observer doctor
```

정상 설치 시 설정 검증 결과가 출력된다. 오류가 있으면 메시지에 따라 조치.

---

## 5. 첫 실행 (드라이런부터)

```bash
# 파이프라인 구조만 검증 (Isaac 실행 X)
python eval_runner.py --checkpoint_dir runs/ --dry_run

# 단일 체크포인트
python eval_runner.py --checkpoint runs/exp_001/model_5000.pth

# 디렉토리 전체 스윕
python eval_runner.py --checkpoint_dir runs/exp_001/

# 여러 실험 재귀 + 최신 체크포인트만
python eval_runner.py --checkpoint_dir runs/ --recursive --latest_only

# 하드웨어 안전 우선 랭킹 + top-2 배포
python eval_runner.py --checkpoint_dir runs/ \
    --auto_select --select_weights hardware_safe --deploy_top_k 2

# 비디오 생략 + W&B 로깅만
python eval_runner.py --checkpoint_dir runs/ \
    --skip_video --wandb_project my-project
```

### ── 실행 전 체크리스트

- [ ] `configs/eval_config.yaml`의 `runtime.task` / `runtime.eval_module` / `runtime.record_script` 설정
  (sharpa-rl-lab 사용자: `docs/adapters/sharpa.md` 참고)
- [ ] `runtime.isaac_lab_path`가 본인 환경에 맞게 설정
- [ ] eval 스크립트가 `docs/20_INTEGRATION_CONTRACT.md`의 `episodes.json` 스키마 만족
- [ ] 헤드리스 서버라면 `Xvfb :99 -screen 0 1920x1080x24 &` 후 `DISPLAY=:99` 환경변수로 실행

---

## 6. 출력물 읽는 법

```
eval_results/
├── eval_report.html                 ← 브라우저에서 열기 (시작점)
├── best/
│   ├── rank01__model_6000.pth       원본으로 심볼릭 링크
│   └── selection_meta.json
└── exp_001__model_5000__YYYYMMDD_HHMMSS/
    ├── eval_config_snapshot.yaml    재현성용 설정 스냅샷
    ├── metrics.json                 에피소드 집계 지표
    ├── episodes.json                에피소드별 원본(있을 때)
    ├── coverage/
    │   ├── success_heatmap.png      roll × pitch 성공률
    │   ├── coverage_scatter.png     실패 모드별 색상
    │   └── pose_histogram.png       샘플링 분포
    └── videos/
        ├── front.mp4 / side.mp4 / top.mp4
        └── combined_grid.mp4        모든 뷰 한 파일
```

### ── 지표 해석 요점

| 지표 | 이게 나쁘면 |
|:---|:---|
| `slip_events_per_episode` | 손끝 액추에이터 마모·하드웨어 위험 |
| `joint_velocity_rms` | 제어 불안정/특이점 근접 |
| `contact_force_rms` | 쥐는 힘 과다 혹은 불안정 |
| `energy_J` | 배터리·발열 문제, 정책 비효율 |
| `object_pos_error_mm` / `rot_error_deg` | 최종 목표 포즈 도달 실패 |

### ── 실패 모드 분포로 할 일 정하기

| 지배적 실패 모드 | 권장 조치 |
|:---|:---|
| `late_slip` 비중 높음 | 보상 함수의 **slip penalty** 강화 |
| `early_drop` 많음 | 초기 그립/커리큘럼 설계 재검토 |
| `singularity_hit` 많음 | 관절 속도 제한/특이점 회피 |
| 히트맵 빨간 구역 | 다음 커리큘럼에서 해당 pose 범위를 더 많이 샘플링 |

---

## 7. 체크포인트 랭킹 프리셋

`--select_weights` 옵션으로 프리셋 전환:

| 프리셋 | 언제 쓰나 |
|:---|:---|
| `balanced` | 평소 실험 비교 |
| `hardware_safe` | 실제 하드웨어 배포 직전 (slip/energy 패널티 큼) |
| `performance_first` | ablation 연구, 순수 성공률 중심 |

---

## Troubleshooting

**GUI 없는 서버에서 비디오가 안 찍힘**
→ Xvfb로 가상 디스플레이 띄우기:
```bash
Xvfb :99 -screen 0 1920x1080x24 &
DISPLAY=:99 python eval_runner.py --checkpoint_dir runs/
```

**정책 로딩 실패**
→ observer는 정책을 직접 로드하지 않는다. `runtime.eval_module`에 지정된 사용자 측 스크립트가 체크포인트/정규화 복원을 담당한다. [`../20_INTEGRATION_CONTRACT.md`](../20_INTEGRATION_CONTRACT.md) §1 참조.

**커버리지/실패분류가 비어있음**
→ 보통 `episodes.json` 생성 실패가 원인. eval 스크립트가 해당 파일을 쓰고 있는지, 스키마가 `EpisodeStats`와 맞는지 확인. sharpa-rl-lab 사용자는 [`../adapters/sharpa.md`](../adapters/sharpa.md) 트러블슈팅 섹션 참고.

**`success_rate`가 0.0으로 나옴**
→ 체크포인트 로딩 또는 observation normalization 복원 실패. eval 스크립트를 단독 실행해서 확인:
```bash
python -m <eval_module> --task=... --load_path=... --num_episodes=5 \
    --metrics_output=/tmp/m.json --episodes_output=/tmp/ep.json --headless
cat /tmp/m.json | python -m json.tool
```

---

## Next steps

| 문서 | 내용 |
|:---|:---|
| [`../20_INTEGRATION_CONTRACT.md`](../20_INTEGRATION_CONTRACT.md) | 본인 프레임워크 붙이는 방법 |
| [`../21_ADAPTER_GUIDE.md`](../21_ADAPTER_GUIDE.md) | 새 어댑터 작성 가이드 |
| [`../30_METRICS_REFERENCE.md`](../30_METRICS_REFERENCE.md) | 지표 + 실패 분류 전체 레퍼런스 |
| [`../31_CHECKPOINT_RANKING.md`](../31_CHECKPOINT_RANKING.md) | 랭킹 알고리즘 + 상태 커버리지 상세 |

---

*팀 온보딩 한글 가이드 · OBSERVER*
