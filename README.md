# AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving
[ICCV 2025] Official implementation of "AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving"

![](./assets/pipeline.png)

## Preparation

### Install

```shell
git clone https://github.com/Dororo99/AD-GS.git
cd AD-GS

conda env create -f environment.yaml
conda activate AD-GS
pip install "git+https://github.com/facebookresearch/pytorch3d.git"  # install pytorch3d

# require CUDA 11.X
pip install -e ./submodules/simple-knn
pip install -e ./submodules/depth-diff-gaussian-rasterization
```

*If you have already installed colmap. You can remove the colmap installization in environment.yaml.*

### Monocular Depth Prior

We use [DPTv2](https://github.com/DepthAnything/Depth-Anything-V2) (Depth-Anything-V2-Large model) to get the monocular depth prior. In our paper, we create a new environment and follow the instructions from DPTv2 to prepare this model.

```shell
git clone https://github.com/DepthAnything/Depth-Anything-V2
cp ./scripts/run-dpt.py ./Depth-Anything-V2/
cd Depth-Anything-V2
conda create -n dpt python=3.11.0
conda activate dpt
pip install -r requirements.txt

# download checkpoints: Depth-Anything-V2-Large
mkdir checkpoints
cd checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true

# back to the original folder and environment
cd ../../
conda activate AD-GS
```

### Semantic Segmentation

We use [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file) to get the semantic segmentation prior. The segmentation provides the position of each possible dynamic objects on the image. In our paper, we create a new environment and run the following instructions to prepare this model.
```shell
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cp ./scripts/semantic.py ./Grounded-SAM-2/
cd Grounded-SAM-2
conda create -n sam python=3.10
conda activate sam
pip install torch torchvision torchaudio
export CUDA_HOME=/path/to/cuda-12.1/
pip install -e .
pip install --no-build-isolation -e grounding_dino

# download checkpoints
cd checkpoints
bash download_ckpts.sh
cd ../gdino_checkpoints
bash download_ckpts.sh

# back to the original folder and environment
cd ../../
conda activate AD-GS
```

*Notice: You may need to add ".png" and ".PNG" to Line 315 in Grounded-SAM-2/sam/utils/misc.py*

### Others

+ We use [Co-Tracker3](https://cotracker3.github.io/) to generate optical flow pseudo labels, and we load the pretrained model through ```torch.hub```.
+ We use ColMap to generate SfM points. If you have several problems in installing ColMap, just try to use conda ```conda install colmap=3.7 -c conda-forge```.

### Datasets

<details>

<summary>KITTI-MOT</summary>

#### Preprocess

Download the dataset [Here](https://www.cvlibs.net/datasets/kitti/eval_tracking.php), including Left/Right images, GPS/IMU data, Camera calibration files, Velodyne Point Clouds. The data structure should be like

```
kitti
|-- data_tracking_calib
|-- data_tracking_image_2
|-- data_tracking_image_3
|-- data_tracking_label_2
|-- data_tracking_oxts
`-- data_tracking_velodyne
```

Use the following instruction to preprocess the dataset.

```shell
bash scripts/kitti/prepare-kitti.sh <path to kitti>
```

#### Pseudo Labels

Generate priors in depth, object&sky mask, optical flow, and SfM. Segment pointcloud based on the object masks.

```shell
# monocular depth prior.
conda activate dpt
cd Depth-Anything-V2
python run-dpt.py --img-path ../data/kitti/<0001, 0002, 0006>/image --outdir ../data/kitti/<0001, 0002, 0006>/depth
cd ..

# object & sky mask.
conda activate sam
cd Grounded-SAM-2
python semantic.py ../data/kitti/<0001, 0002, 0006> --text sky. --name sky
python semantic.py ../data/kitti/<0001, 0002, 0006> --text car.bus.truck.van.human. --name semantic
cd ..

# segment pcd based on the object masks
conda activate AD-GS
bash scripts/kitti/segment-pcd.sh

# optical flow
bash scripts/kitti/prepare-flow.sh

# colmap
bash scripts/kitti/prepare-colmap.sh
```

</details>

<details>

<summary>Waymo</summary>

Download the dataset [Here](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_1/individual_files?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))), and the data structure should be like

```
waymo
|-- individual_files_validation_segment-10448102132863604198_472_000_492_000_with_camera_labels.tfrecord  # scene006
|-- individual_files_validation_segment-12374656037744638388_1412_711_1432_711_with_camera_labels.tfrecord  # scene026
|-- individual_files_validation_segment-17612470202990834368_2800_000_2820_000_with_camera_labels.tfrecord  # scene090
|-- individual_files_validation_segment-1906113358876584689_1359_560_1379_560_with_camera_labels.tfrecord  # scene105
|-- individual_files_validation_segment-2094681306939952000_2972_300_2992_300_with_camera_labels.tfrecord  # scene108
|-- individual_files_validation_segment-4246537812751004276_1560_000_1580_000_with_camera_labels.tfrecord  # scene134
|-- individual_files_validation_segment-5372281728627437618_2005_000_2025_000_with_camera_labels.tfrecord  # scene150
`-- individual_files_validation_segment-8398516118967750070_3958_000_3978_000_with_camera_labels.tfrecord  # scene181
```

We use the eight scenes selected by [StreetGS](https://github.com/zju3dv/street_gaussians). Use the following instruction to preprocess the dataset.

```shell
 # install waymo utils
pip install tensorflow==2.11.0
pip install waymo-open-dataset-tf-2-11-0==1.6.1 --no-dependencies

# preprocess
bash scripts/waymo/prepare-waymo.sh <path to waymo>
```

#### Pseudo Labels

Generate priors in depth, object&sky mask, optical flow, and SfM. Segment pointcloud based on the object masks.

```shell
# monocular depth prior.
conda activate dpt
cd Depth-Anything-V2
python run-dpt.py --img-path ../data/waymo/sceneXXX/image --outdir ../data/waymo/sceneXXX/depth
cd ..

# object & sky mask.
conda activate sam
cd Grounded-SAM-2
python semantic.py ../data/waymo/sceneXXX --text sky. --name sky
python semantic.py ../data/waymo/sceneXXX --text car.bus.truck.van.human. --name semantic
cd ..

# segment pcd based on the object masks
conda activate AD-GS
bash scripts/waymo/segment-pcd.sh

# optical flow
bash scripts/waymo/prepare-flow.sh

# colmap
bash scripts/waymo/prepare-colmap.sh
```

</details>


<details>

<summary>nuScenes</summary>

#### Preprocess

Download the dataset [Here](https://www.nuscenes.org/nuscenes#data-collection), and the data structure should be like

```
nuScenes
|-- can_bus
|-- info
|-- lidarseg
|-- maps
|-- mini
|-- nuscenes_test
|-- samples
|-- sweeps
|-- tar
|-- test
|-- v1.0-test
`-- v1.0-trainval
```

We use the 10 to 69(inclusive) frames of scene 0230, 0242, 0255, 0295, 0518 and 0749. Use the following instruction to preprocess the dataset.

```shell
bash scripts/nuscene/prepare-nuscenes.sh <path to nuScenes>
```

#### Pseudo Labels

Generate priors in depth, object&sky mask, optical flow, and SfM. Segment pointcloud based on the object masks.

```shell
# monocular depth prior.
conda activate dpt
cd Depth-Anything-V2
python run-dpt.py --img-path ../data/nuscenes/sceneXXX/image --outdir ../data/nuscenes/sceneXXX/depth
cd ..

# object & sky mask.
conda activate sam
cd Grounded-SAM-2
python semantic.py ../data/nuscenes/sceneXXX --text sky. --name sky
python semantic.py ../data/nuscenes/sceneXXX --text car.bus.truck.van.human.bike. --name semantic
cd ..

# segment pcd based on the object masks
conda activate AD-GS
bash scripts/nuscene/segment-pcd.sh

# optical flow
bash scripts/nuscene/prepare-flow.sh

# colmap
bash scripts/nuscene/prepare-colmap.sh
```

</details>

## SplatAD 50% LINSPACE protocol (Waymo, nuScenes, AV2)

For this selected-scene experiment, this section supersedes the older upstream
Waymo and nuScenes examples above. The launchers do not use the old eight Waymo
validation scenes or the six cropped nuScenes examples.

The launchers below contain the selected 10 scenes for each dataset and reproduce
the SplatAD/NeurAD **data protocol**:

- Waymo: `FRONT`, `FRONT_LEFT`, `FRONT_RIGHT`, and `TOP` LiDAR.
- nuScenes: all six cameras and `LIDAR_TOP`, including each full asynchronous
  `sample_data` chain.
- AV2: all seven ring cameras in SplatAD's camera-major order. Each physical
  LiDAR sweep contributes independent `lidar_up` and `lidar_down` observation
  entries, matching the parser's two-sensor metadata layout; the physical sweep
  itself is read only once when constructing the initialization cloud.

The 50% LINSPACE split is applied independently to every camera and LiDAR
sensor. For a sensor with `N` observations, train contains `ceil(N * 0.5)`
indices from `np.linspace(0, N - 1, ..., dtype=int64)` and val is the exact
complement. This is intentionally not a global even/odd split. Initialization
`points3d.ply` is built only from independently selected train LiDAR sweeps;
held-out LiDAR is rejected by both the loader and preflight validator. The
Waymo converter also uses SplatAD's exposure-center pose and timestamp computed
from each `CameraImage` pose, velocity, trigger time, and readout-done time.
All selected camera and LiDAR timestamps share SplatAD's common sensor-time
origin. AD-GS stores that same common interval normalized linearly to `[0, 1]`;
`frame_gap` is the median train-camera spacing divided by the common duration.

Raw symlinks remain under `data/{nuscenes,waymo,av2}`. Converted scenes are
written separately under `data/processed/<dataset>`:

```shell
# nuScenes: six cameras, full asynchronous sample_data chains
VALIDATE_ONLY=1 bash scripts/nuscene/prepare-nuscenes.sh
bash scripts/nuscene/prepare-nuscenes.sh

# Waymo training split: FRONT, FRONT_LEFT, FRONT_RIGHT and TOP LiDAR
bash scripts/waymo/prepare-waymo.sh

# AV2 physical train split: all seven ring cameras
bash scripts/av2/prepare-av2.sh --validate_only
bash scripts/av2/prepare-av2.sh
```

Immediately after conversion, metadata and train-only LiDAR can be checked with
`--metadata-only`:

```shell
/venv/ad-gs/bin/python scripts/validate_splatad_scene.py \
    data/processed/nuscenes/scene-0101 --dataset nuscenes --metadata-only
```

Then prepare all required AD-GS priors for each converted scene. The integrated
launcher runs DPT, camera-isolated Grounded-SAM-2, flow, point segmentation, and
headless CPU COLMAP in a private work area. It verifies the complete result
before installing it, preserves the unsegmented PLY, and refuses existing priors
unless `OVERWRITE=1` is explicitly set.

Priors are streamed one scene and one camera at a time. Depth is stored as
compressed `float16` and restored to `float32` (maximum normalized quantization
error `5e-4`), semantic/sky preserve their nonzero binary masks losslessly by
bit-packing, and flow remains exact in compressed NPZ form. Legacy NPY/NPZ
priors remain readable. A conservative planning estimate for all 30 scenes is
about 354 GiB, but actual flow compression is data-dependent. The end-to-end
launchers use a scene-scoped lock per physical GPU. Duplicate builders on the
same GPU are serialized, while nuScenes on GPU 4 and AV2 on GPU 5 can preprocess
concurrently in isolated scene workspaces. They check a 100 GiB free-space floor
before every prior and train/render scene
and stop without deleting staging if the floor is crossed. This estimate
excludes converted images/PLY/COLMAP data and training checkpoints or renders,
so continue monitoring `df -h` and apply an explicit output-retention policy
instead of automatic deletion.

```shell
# Inspect one scene without writing or running inference.
DRY_RUN=1 bash scripts/prepare_splatad_priors.sh \
    nuscenes data/processed/nuscenes/scene-0101 4

# Run one scene per dataset; repeat for every scene in its dataset launcher.
bash scripts/prepare_splatad_priors.sh \
    nuscenes data/processed/nuscenes/scene-0101 4
bash scripts/prepare_splatad_priors.sh \
    waymo data/processed/waymo/4986495627634617319_2980_000_3000_000 6
bash scripts/prepare_splatad_priors.sh \
    av2 data/processed/av2/a7bcdabb-f9b7-3c16-806d-3ddf1c2d49a2 5
```

The direct training launchers validate every scene before creating a worker.
nuScenes uses physical GPU 4 and AV2 uses physical GPU 5; each launcher passes
the same GPU twice to run one worker in scene-list order. Waymo likewise uses
one configurable GPU. On completion, `render.py --skip_train` evaluates the
complete validation complement.

Use the end-to-end launchers when priors are incomplete. They strictly skip
ready scenes, apply `RESUME=1` only to manifest-validated staging, build all
remaining priors, and then invoke the single-GPU training batch.

```shell
# Inspect both end-to-end pipelines without acquiring locks or writing.
PIPELINE_DRY_RUN=1 bash scripts/run_nuscenes_preprocess_then_train.sh
PIPELINE_DRY_RUN=1 bash scripts/run_av2_preprocess_then_train.sh

# End-to-end: nuScenes -> GPU 4, AV2 -> GPU 5.
bash scripts/run_nuscenes_preprocess_then_train.sh
bash scripts/run_av2_preprocess_then_train.sh

# Direct training only, when every prior is already ready.
bash scripts/train_nuscenes_splatad.sh
bash scripts/train_waymo_splatad.sh
bash scripts/train_av2_splatad.sh
```

These launchers enable live W&B logging by default and mirror the existing
TensorBoard scalars/images into one run per scene. The entity is
`CamoSplat_ICLR_2027`; the convention `[SplitType]_[DatasetType]_[Model]`
produces these projects for the AD-GS baseline:

- `SplatAD_nuScenes_AD-GS`
- `SplatAD_Waymo_AD-GS`
- `SplatAD_Argoverse2_AD-GS`

Following the reference SplatAD logger, every 500 iterations AD-GS selects
three deterministic held-out views from the front camera (camera ID 0). It
logs one image grid with ground truth on the left and the current render on
the right for each row. The same views also report L1, PSNR, SSIM, LPIPS, and
render FPS. Training logs additionally include all native loss components,
dynamic learning rates, iteration speed/ETA, GPU memory, active SH degree,
and total/scene/object Gaussian counts.

The terminal prints a plain-text heartbeat every 100 iterations, plus
scene-level `TRAIN START`, `TRAIN DONE/FAILED`, `RENDER START`,
`RENDER DONE/FAILED`, elapsed time, and the W&B URL. The complete stdout and
stderr stream remains visible in the terminal and is also saved to
`<output-root>/<scene>/launcher.log`.

Before launching any worker, the batch validates every configured scene and
prints `READY` or `NEEDS_PRIORS` for each one. If a scene is incomplete, no
training starts and the summary prints the exact full-prior preparation command.
An interrupted `.adgs-priors-work` is first validated and resumed explicitly
with `RESUME=1`. The launchers never enable `OVERWRITE=1`, delete staging, or
fabricate a missing point-cloud `obj` field automatically; invalid staging
requires manual inspection.

Each run name is the scene ID and runs are grouped by project. Defaults can be
overridden without editing the scripts, for example
`WANDB_MODE=offline`, `WANDB_ENABLED=0`, `WANDB_PROJECT=...`, or
`WANDB_RUN_NAME_PREFIX=debug-`. Logging cadence can be adjusted with
`WANDB_EVAL_INTERVAL`, `WANDB_EVAL_IMAGE_COUNT`,
`WANDB_SCALAR_LOG_INTERVAL`, and `ADGS_CONSOLE_LOG_INTERVAL`. Set
`WANDB_EVAL_LPIPS=0` to omit the optional LPIPS calculation.

This parity statement covers the selected sensors, observation ordering,
per-sensor split, camera calibration/crops, Waymo exposure-center pose, and
train-only LiDAR seed. AD-GS is still a different model and uses a single
center-time pinhole pose; it does not reproduce SplatAD's per-column Waymo
rolling-shutter renderer (about 54 ms readout in the checked segment). It also
does not reproduce SplatAD's point-level rolling-LiDAR timing/return supervision
or synthetic missing-return augmentation (`add_missing_points`): AD-GS has no
LiDAR-ray loss and uses measured first-return points for initialization.

The setup step only prepares code, environments, checkpoints, and launchers. It
does not start full conversion, prior generation, or training; those jobs begin
only when the commands above are invoked.

## Run

Use the scripts to train and evaluate our model.

```shell
# kitti
bash scripts/kitti/run-kitti.sh cuda:0

# waymo
bash scripts/waymo/run-waymo.sh cuda:0

# nuscenes
bash scripts/nuscene/run-nuscenes.sh cuda:0

# The first argument means the device ID.
```
The results can be found in ```./output```.

## Acknowledgments

This framework is adapted from [Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). We also thank [DPTv2](https://github.com/DepthAnything/Depth-Anything-V2), [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2?tab=readme-ov-file) and [Co-Tracker3](https://cotracker3.github.io/) for their great works.

## BibTex

```
@article{xu2025adgs,
    title={{AD-GS}: Object-Aware {B-Spline} {Gaussian} Splatting for Self-Supervised Autonomous Driving},
    author={Jiawei, Xu and Kai, Deng and Zexin, Fan and Shenlong, Wang and Jin, Xie and Jian, Yang},
    journal={International Conference on Computer Vision},
    year={2025},
}
```
