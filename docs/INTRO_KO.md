# 👁️ OBSERVER 온보딩 가이드 (팀 내부용)

> 이 문서는 팀을 위한 한글 요약본입니다.
> 전체 내용은 루트의 [`README.md`](../README.md)를 참고하세요.

---

## 1. 이 프로젝트는 무엇을 하나요?

**OBSERVER**는 고DOF dexterous manipulation 정책의
체크포인트를 **자동으로 평가·분류·랭킹·기록**하는 파이프라인입니다.

- 입력: 학습된 체크포인트 (단일 파일 또는 디렉토리)
- 출력: 메트릭 JSON, 실패모드 분포, 상태 커버리지 히트맵, 다각도 비디오,
  W&B/TensorBoard 로그, HTML 리포트

> 한 줄 요약: **"영상 보지 말고, 데이터 읽자."**

### 왜 필요한가?

- 체크포인트는 빠르게 쌓이지만 사람이 눈으로 보며 비교하는 건 한계
- `success_rate` 하나로는 slip, 관절 속도 스파이크, 에너지 과다 사용 등의
  하드웨어 위험 신호를 놓침
- 수동 리뷰에는 선택 편향·최신 편향·피로가 낀다

---

## 2. 핵심 개념 한눈에 보기

| 개념 | 설명 |
|:---|:---|
| 📦 **MetricsCollector** | 에피소드 스텝마다 성공률·접촉력·slip 등 8개 지표 수집 |
| 🔍 **FailureModeClassifier** | 우선순위 규칙 체인으로 에피소드별 실패 유형 분류 (학습 데이터 불필요) |
| 🗺️ **StateCoverageAnalyzer** | roll × pitch 초기 포즈 공간에서 어디가 약한지 히트맵 |
| 🎬 **CameraController + VideoRecorder** | 5개 시점 + 2×3 그리드 비디오 저장 |
| 📡 **ExperimentTracker** | W&B / TensorBoard 자동 감지 및 로깅 |
| 🏆 **CheckpointSelector** | 다목적 가중치 점수로 top-k 체크포인트 선정 |
| 📄 **ReportGenerator** | 차트·파이·비디오·히트맵이 포함된 self-contained HTML 리포트 |
| 🔄 **PipelineOrchestrator** | 체크포인트 1개당 위 단계들을 조율 |

---

## 3. 폴더/파일 구조

실제 리포지토리 레이아웃은 다음과 같습니다.

```
observer/
├── eval_runner.py              진입점 (설치 시 `observer` CLI로 노출)
├── brand.py                    콘솔 배너/브랜딩
├── requirements.txt            핵심 의존성
├── setup.py                    패키지 설치 스크립트
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
│   ├── camera_controller.py    Isaac Sim 뷰포트 제어 (유틸 라이브러리)
│   └── recorder.py             Replicator 기반 비디오 캡처 (유틸 라이브러리)
├── docs/
│   ├── INTEGRATION.md          프레임워크 통합 계약 (eval/record 스크립트 규약)
│   └── adapters/               프레임워크별 설정 예시 (sharpa 등)
├── report/
│   └── report_generator.py     HTML 리포트 생성
└── tactile/
    └── overlay.py              Deform map 비디오 오버레이
```

> 📌 **자주 보게 될 파일 TOP 3**
> 1. `configs/eval_config.yaml` — 실험마다 값 바꿔가며 작업
> 2. `docs/INTEGRATION.md` — 본인 프레임워크 붙일 때 이 계약만 맞추면 됨
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

# 5) Isaac Lab 환경 (이미 설치되어 있다고 가정)
#    omni.isaac.lab · omni.replicator.core · omni.kit.viewport.utility
```

설치 후:

```bash
pip install -e .   # `observer` CLI 사용 가능
```

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

### 실행 전 체크리스트

- [ ] `configs/eval_config.yaml`의 `runtime.task` / `runtime.eval_module` /
      `runtime.record_script`가 본인 프레임워크 값으로 채워졌는가?
      (sharpa-rl-lab 사용자는 `docs/adapters/sharpa.md` 참고)
- [ ] `runtime.isaac_lab_path`가 본인 환경에 맞게 설정되어 있는가?
- [ ] eval 스크립트가 `docs/INTEGRATION.md`의 `episodes.json` 스키마를 만족하는가?
- [ ] 헤드리스 서버라면 `Xvfb :99 -screen 0 1920x1080x24 &` 후
      `DISPLAY=:99` 환경변수로 실행

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
    ├── camera_poses.json
    ├── coverage/
    │   ├── success_heatmap.png      roll × pitch 성공률
    │   ├── coverage_scatter.png     실패 모드별 색상
    │   └── pose_histogram.png       샘플링 분포
    └── videos/
        ├── front.mp4 / side.mp4 / top.mp4
        └── combined_grid.mp4        모든 뷰 한 파일
```

### 지표 해석 요점

| 지표 | 이게 나쁘면 |
|:---|:---|
| `slip_events_per_episode` | 손끝 액추에이터 마모·하드웨어 위험 |
| `joint_velocity_rms` | 제어 불안정/특이점 근접 |
| `contact_force_rms` | 쥐는 힘 과다 혹은 불안정 |
| `energy_J` | 배터리·발열 문제, 정책 비효율 |
| `object_pos_error_mm` / `rot_error_deg` | 최종 목표 포즈 도달 실패 |

### 실패 모드 분포로 할 일 정하기

- `late_slip` 비중이 크다 → 보상함수의 **slip penalty** 강화
- `early_drop`이 많다 → 초기 그립/커리큘럼 설계 재검토
- `singularity_hit`이 많다 → 관절 속도 제한/특이점 회피
- 히트맵에서 **빨간 구역** = 다음 커리큘럼에서 더 많이 샘플링할 지역

---

## 7. 체크포인트 랭킹 프리셋

`--select_weights` 옵션으로 프리셋 전환:

| 프리셋 | 언제 쓰나 |
|:---|:---|
| `balanced` | 평소 실험 비교 |
| `hardware_safe` | 실제 하드웨어 배포 직전 (slip/energy 패널티 큼) |
| `performance_first` | ablation 연구, 순수 성공률 중심 |

---

## 8. 자주 만나는 이슈

- **GUI 없는 서버에서 비디오가 안 찍힘** → Xvfb로 가상 디스플레이 띄우기
- **정책 로딩 실패** → observer 는 정책을 직접 로드하지 않는다. `runtime.eval_module`
  에 지정된 사용자 측 스크립트가 체크포인트/정규화 복원을 담당한다.
  `docs/INTEGRATION.md` §1의 계약을 참조.
- **커버리지/실패분류가 비어있음** → 보통 `episodes.json` 생성 실패가 원인.
  eval 스크립트가 해당 파일을 쓰고 있는지, 스키마가
  `EpisodeStats` 와 맞는지 확인. sharpa-rl-lab 사용자는
  `docs/adapters/sharpa.md` 트러블슈팅 섹션 참고.

---

## 9. 더 읽기

- 전체 영문 README: [`../README.md`](../README.md)
- Isaac Lab 공식 문서: https://isaac-sim.github.io/IsaacLab/
- W&B: https://wandb.ai

---

*Last updated: 2026-04 · 팀 온보딩 한글 요약본*
