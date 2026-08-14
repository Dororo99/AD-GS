# AD-GS SplatAD Training Handoff

기준 시각: 2026-08-14 04:35 UTC

이 문서는 선택한 Waymo, nuScenes, Argoverse 2(AV2) 씬을 SplatAD와 같은
데이터 분할 및 센서 조건으로 AD-GS에서 학습하기 위해 적용한 변경과 현재
실행 상태를 정리한다. Git 저장소에는 코드와 이 문서만 포함된다. 원본 데이터,
전처리 결과, 학습 출력, W&B 인증 정보, Codex 대화 원문은 포함되지 않는다.

## 1. 저장소 기준점

- 원격 저장소: `https://github.com/Dororo99/AD-GS.git`
- 작업 브랜치: `main`
- 학습 설정 구현 커밋: `9c9e230` (`Configure SplatAD training setup`)
- 원본 upstream: `https://github.com/JiaweiXu8/AD-GS.git`
- 전체 설정과 사용법은 `README.md`의 **SplatAD-compatible selected-scene
  protocol** 부분도 함께 참고한다.

설정 커밋은 데이터 변환기, 센서별 split, prior 생성, strict validator,
단일-GPU 순차 학습 런처, W&B 로깅 및 관련 테스트를 추가/수정했다.

## 2. 실험 계약

| Dataset | Cameras | LiDAR | Split | 실행 GPU | W&B project |
| --- | --- | --- | --- | --- | --- |
| Waymo | `FRONT`, `FRONT_LEFT`, `FRONT_RIGHT` | `TOP` | 센서별 LINSPACE 50% train, 나머지 validation | 완료 실행은 GPU 2 | `SplatAD_Waymo_AD-GS` |
| nuScenes | 6개 카메라 전체 | `LIDAR_TOP` | 센서별 LINSPACE 50% train, 나머지 validation | GPU 4, 순차 | `SplatAD_nuScenes_AD-GS` |
| AV2 | 7개 ring 카메라 전체 | paired LiDAR metadata | 센서별 LINSPACE 50% train, 나머지 validation | GPU 5, 순차 | `SplatAD_Argoverse2_AD-GS` |

반드시 지켜야 하는 조건은 다음과 같다.

- 각 센서 스트림에서 독립적으로 LINSPACE 50%를 train으로 선택하고 그
  complement를 validation으로 사용한다.
- Gaussian 초기화에는 train split에 속한 LiDAR point만 사용한다. held-out
  LiDAR point가 들어오면 loader와 validator가 거부한다.
- Waymo pose/timestamp는 SplatAD의 exposure-center 계산을 따른다.
- 모든 prior가 준비되고 `scripts/validate_splatad_scene.py`의 strict 검증을
  통과하기 전에는 학습하지 않는다.
- `points3d.ply`에는 point segmentation으로 생성된 `obj` field가 있어야 한다.
  누락 시 0으로 채우지 말고 `segment_pcd.py`가 포함된 전체 prior pipeline을
  실행한다.

AD-GS는 SplatAD와 다른 모델이다. 선택 센서, observation ordering, split,
calibration/crop, Waymo exposure-center pose, train-only LiDAR seed는 맞췄지만
SplatAD의 per-column rolling-shutter renderer, point-level rolling-LiDAR timing,
LiDAR-ray loss 및 synthetic missing-return augmentation까지 재현한 것은 아니다.

## 3. 선택한 씬

### nuScenes (10)

```text
scene-0101
scene-0689
scene-0716
scene-1096
scene-0683
scene-0758
scene-1017
scene-0100
scene-0235
scene-0252
```

### Waymo (10)

```text
4986495627634617319_2980_000_3000_000
4672649953433758614_2700_000_2720_000
6791933003490312185_2607_000_2627_000
17364342162691622478_780_000_800_000
3385534893506316900_4252_000_4272_000
9747453753779078631_940_000_960_000
14940138913070850675_5755_330_5775_330
204421859195625800_1080_000_1100_000
7566697458525030390_1440_000_1460_000
17159836069183024120_640_000_660_000
```

### AV2 (10)

```text
a7bcdabb-f9b7-3c16-806d-3ddf1c2d49a2
76c3f58f-9003-3bdb-90a3-b87cfbfa1c3b
5f2b8881-3447-3905-99f8-def9d72aae42
d201af7e-48c8-34ad-be1c-e649af2cb5c2
4d9e3bdf-7216-3161-8281-72863f3c2bf6
38f30522-2d43-3ff3-a94b-84887ab1671d
756f4ed0-5352-31e4-b3c6-2841b9e779d7
91cded81-9f72-3930-bab7-5d3e3fa0a220
511b93af-f16e-3195-8628-fbb972a17f74
f5a3ee79-a131-3f8a-91e9-a6475d778149
```

## 4. 구현 구조

주요 진입점은 다음과 같다.

- 데이터 변환:
  - `scripts/nuscene/prepare-nuscenes.sh`
  - `scripts/waymo/prepare-waymo.sh`
  - `scripts/av2/prepare-av2.sh`
- 전체 prior 생성: `scripts/prepare_splatad_priors.sh`
- strict 검증: `scripts/validate_splatad_scene.py`
- end-to-end 전처리 후 학습:
  - `scripts/run_nuscenes_preprocess_then_train.sh`
  - `scripts/run_waymo_preprocess_then_train.sh`
  - `scripts/run_av2_preprocess_then_train.sh`
- 학습만 실행:
  - `scripts/train_nuscenes_splatad.sh`
  - `scripts/train_waymo_splatad.sh`
  - `scripts/train_av2_splatad.sh`
- 공통 순차 학습기: `scripts/train_splatad_split.sh`
- split 구현: `scripts/splatad_split.py`

prior pipeline은 씬별 private staging인
`data/processed/<dataset>/<scene>/.adgs-priors-work`에서 다음을 수행한다.

1. Depth Anything V2 depth
2. 카메라 스트림을 섞지 않는 Grounded-SAM-2 sky/semantic mask
3. optical flow
4. `segment_pcd.py`를 통한 `obj` field 생성
5. headless CPU COLMAP
6. 전체 산출물 strict 검증 후 최종 위치에 설치

depth는 압축 `float16`, semantic/sky는 lossless bit-packing, flow는 compressed
NPZ로 저장한다. 중단된 staging은 manifest를 검증한 뒤에만 `RESUME=1`로
재사용한다. 자동 삭제나 자동 overwrite는 하지 않는다.

같은 물리 GPU의 prior builder는 다음 GPU-scoped lock으로 직렬화된다.

```text
output/.adgs-prior-builder.gpu-<GPU>.lock
```

따라서 GPU 4의 nuScenes와 GPU 5의 AV2는 서로 막지 않고 동시에 전처리할 수
있다. 각 파이프라인은 씬마다 free disk를 확인하며 기본 하한은 100 GiB이다.

## 5. 환경과 로컬 의존성

현재 서버의 런처 기본 경로는 다음과 같다.

```text
AD-GS Python: /venv/ad-gs/bin/python
DPT Python:   /venv/ad-gs-dpt/bin/python
SAM Python:   /venv/ad-gs-sam/bin/python
DPT repo:     ./Depth-Anything-V2
SAM repo:     ./Grounded-SAM-2
COLMAP:       /venv/ad-gs/bin/colmap
Compiler:     gcc-11 / g++-11
CUDA arch:    8.9
```

`Depth-Anything-V2/`, `Grounded-SAM-2/`, 모델 checkpoint 및 로컬 environment는
`.gitignore` 대상이다. 새 서버에서는 `README.md`에 따라 환경을 만들고 두
외부 저장소와 checkpoint를 별도로 복구해야 한다. W&B API key도 Git에 넣지
말고 새 서버에서 로그인하거나 환경 변수/secret store로 제공한다.

데이터 배치는 다음과 같다.

```text
data/nuscenes -> <raw nuScenes location>
data/waymo    -> <raw Waymo location>
data/av2      -> <raw AV2 location>
data/processed/<dataset>/<scene>/...
output/<dataset>_splatad/<scene>/...
```

`data/`와 `output/` 전체가 Git에서 제외된다. 다른 서버에서 clone만 하면
symlink, converted data, prior, checkpoint, render 및 log가 생기지 않는다.
필요한 산출물은 `rsync`, 공유 스토리지 또는 object storage로 별도 전송한다.

## 6. W&B 로깅

- Entity: `CamoSplat_ICLR_2027`
- Run name: scene ID
- Scalar 기본 주기: 10 iteration
- 콘솔 heartbeat 기본 주기: 100 iteration
- Evaluation 기본 주기: 500 iteration
- 고정 front camera: camera ID 0
- 기본 preview 수: held-out view 3개
- 허용한 유일한 image media key:
  `Eval Images/fixed_front_gt_render`

해당 media는 각 row의 왼쪽에 GT, 오른쪽에 render를 놓은 grid이다. L1, PSNR,
SSIM, LPIPS, render FPS 및 학습 scalar는 별도 scalar key로 기록한다.

Waymo 마지막 run은 한 scene에서 W&B가 약 1.69 GiB, media file 120개를
동기화했다고 보고했다. media 용량을 더 줄여야 하면 실행 전에
`WANDB_EVAL_INTERVAL`을 늘리거나 `WANDB_EVAL_IMAGE_COUNT`를 줄인다. media key를
추가하지 않는다.

## 7. 2026-08-14 실행 상태

### Waymo

- tmux session: `adgs_waymo_gpu2` (종료됨)
- 실제 실행 GPU: 2
- 10/10 scene의 60,000 iteration 학습 완료
- 10/10 scene의 validation render 완료
- 각 `launcher.log`에서 `TRAIN DONE`, `RENDER DONE`, `SCENE DONE` 확인
- 모든 scene에 최종 Gaussian point cloud 존재

이 설정은 별도의 `chkpnt*.pth`를 저장하지 않는다. 최종 학습 산출물은 다음
경로의 PLY이다.

```text
output/waymo_splatad/<scene>/point_cloud/iteration_60000/point_cloud.ply
```

완료된 `<scene>`은 아래 10개 전부이다.

```text
4986495627634617319_2980_000_3000_000
4672649953433758614_2700_000_2720_000
6791933003490312185_2607_000_2627_000
17364342162691622478_780_000_800_000
3385534893506316900_4252_000_4272_000
9747453753779078631_940_000_960_000
14940138913070850675_5755_330_5775_330
204421859195625800_1080_000_1100_000
7566697458525030390_1440_000_1460_000
17159836069183024120_640_000_660_000
```

각 output directory에는 `cfg_args`, `cameras.json`, `launcher.log`, W&B local
run 자료 및 render 결과도 있다.

주의: `scripts/train_waymo_splatad.sh`와
`scripts/run_waymo_preprocess_then_train.sh`의 코드상 GPU 기본값은 현재 0이다.
완료된 batch는 환경 변수로 GPU 2를 지정해 실행했다. 다시 GPU 2에서 실행할
때는 아래처럼 명시한다.

```shell
WAYMO_PIPELINE_GPU=2 bash scripts/run_waymo_preprocess_then_train.sh
# 또는 prior가 모두 ready인 경우
WAYMO_TRAIN_GPU=2 bash scripts/train_waymo_splatad.sh
```

완료 output이 이미 있으므로 현재 output root에 전체 Waymo launcher를 다시
실행하지 않는다. 공통 trainer는 기본 `ALLOW_EXISTING_OUTPUT=0`이며 기존 output을
발견하면 중단한다.

### nuScenes

- tmux session: `adgs_nuscenes_gpu4` (실행 중)
- GPU 4에서 scene 순차 전처리 후 순차 학습하도록 실행됨
- 현재 `PRIOR 1/10`, `scene-0101`의 optical flow 단계
- 아직 학습이 시작되지 않았고 `iteration_60000` 산출물도 없음
- pipeline log: `output/nuscenes_splatad/pipeline.log`

### AV2

- tmux session: `adgs_av2_gpu5` (실행 중)
- GPU 5에서 scene 순차 전처리 후 순차 학습하도록 실행됨
- `a7bcdabb-f9b7-3c16-806d-3ddf1c2d49a2`는 strict-ready로 skip됨
- 현재 `PRIOR 2/10`, `76c3f58f-9003-3bdb-90a3-b87cfbfa1c3b`의 optical
  flow를 `RESUME=1`로 처리 중
- 이전의 반복 `LOCK WAITING`은 GPU별 lock 적용 후 해소되었고 현재 실제
  `flow.py` 프로세스가 GPU 5에서 동작 중
- 아직 학습이 시작되지 않았고 `iteration_60000` 산출물도 없음
- pipeline log: `output/av2_splatad/pipeline.log`

상태 확인 시 GPU 4와 5 모두 100% utilization이었다. filesystem은 3.0 TiB 중
약 286 GiB가 남아 91% 사용 상태였으며, 100 GiB guard 위에 있지만 여유가
크지 않다.

## 8. 현재 서버 모니터링

실행 중인 세션에 attach하려면 다음을 사용한다.

```shell
tmux attach -t adgs_nuscenes_gpu4
tmux attach -t adgs_av2_gpu5
```

attach 없이 log를 확인하려면 다음을 사용한다.

```shell
tail -f output/nuscenes_splatad/pipeline.log
tail -f output/av2_splatad/pipeline.log
nvidia-smi
df -h .
```

scene별 완료 표식은 다음처럼 확인한다.

```shell
rg '\[(TRAIN DONE|RENDER DONE|SCENE DONE|FAILED)' \
  output/waymo_splatad/*/launcher.log

find output -path '*/point_cloud/iteration_60000/point_cloud.ply' -print
```

## 9. 새 서버에서 복구 및 재개

1. 저장소를 clone하고 `README.md`에 따라 AD-GS, DPT, SAM 환경과 checkpoint를
   설치한다.
2. raw dataset symlink를 다시 만들거나 공유 스토리지를 mount한다.
3. `data/processed/`와 `output/`이 필요하면 현재 서버에서 별도 전송한다.
4. strict validator로 scene을 확인한다.

```shell
/venv/ad-gs/bin/python scripts/validate_splatad_scene.py \
  data/processed/nuscenes/scene-0101 --dataset nuscenes
```

metadata와 train-only LiDAR까지만 확인하려면 `--metadata-only`를 추가한다.

중단된 end-to-end pipeline은 같은 명령을 다시 실행한다. launcher가 ready scene을
strict validation 후 skip하고, 유효한 `.adgs-priors-work`만 자동으로
`RESUME=1` 처리한다.

```shell
bash scripts/run_nuscenes_preprocess_then_train.sh
bash scripts/run_av2_preprocess_then_train.sh
```

쓰기 없이 실행 계획만 확인할 수 있다.

```shell
PIPELINE_DRY_RUN=1 bash scripts/run_nuscenes_preprocess_then_train.sh
PIPELINE_DRY_RUN=1 bash scripts/run_av2_preprocess_then_train.sh
```

모든 prior가 strict-ready인 경우에만 학습 launcher를 직접 실행한다.

```shell
bash scripts/train_nuscenes_splatad.sh
WAYMO_TRAIN_GPU=2 bash scripts/train_waymo_splatad.sh
bash scripts/train_av2_splatad.sh
```

`train.py` 기반 학습은 중간 iteration checkpoint 자동 재개를 구현하지 않았다.
불완전 output directory가 있으면 삭제/overwrite하기 전에 반드시 수동 점검한다.
완료된 scene을 보호하기 위해 `ALLOW_EXISTING_OUTPUT=1`을 무심코 사용하지 않는다.

## 10. 검증 및 다음 작업

설정 구현 당시 repository test 47개와 shell syntax 검사를 통과했다. 새 서버에서
환경을 복구한 뒤 다음을 다시 실행한다.

```shell
/venv/ad-gs/bin/python -m pytest -q tests
bash -n scripts/*.sh scripts/nuscene/*.sh scripts/waymo/*.sh scripts/av2/*.sh
```

우선순위는 다음과 같다.

1. 현재 서버의 nuScenes/AV2 전처리가 계속 진행되는지 log와 GPU utilization을
   확인한다.
2. free disk가 100 GiB guard에 근접하기 전에 output 보존/이관 정책을 정한다.
3. 각 dataset의 학습이 시작되면 scene별 `launcher.log`, W&B URL,
   `iteration_60000/point_cloud.ply`, render 완료 표식을 확인한다.
4. 다른 서버로 넘어갈 때 Git clone과 별개로 필요한 `data/processed/` 및
   `output/`을 전송한다.

이 문서가 Codex 대화 원문을 재생하거나 세션을 복원하는 것은 아니다. 새
세션에서는 이 파일과 `README.md`, 현재 pipeline log를 먼저 읽으면 동일한 작업
맥락에서 이어갈 수 있다.
