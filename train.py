#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import time
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, get_flow_loss, get_depth_loss
from utils.depth_utils import get_scaled_shifted_depth
from utils.flow_utils import flow_to_img, get_img_flow
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_config
from scene.env import EnvironmentMap
from scene.cameras import Camera
from torch.utils.tensorboard import SummaryWriter


_LPIPS_METRIC = None
_LPIPS_UNAVAILABLE = False
WANDB_EVAL_MEDIA_KEY = 'Eval Images/fixed_front_gt_render'
WANDB_EVAL_STEP_KEY = 'Eval Images/iteration'


class _SelectiveWandbMediaWriter:
    """Allow exactly one W&B media key while preserving scalar sync.

    The selected image bypasses TensorBoard to avoid storing and uploading it
    twice. All other image calls are dropped; non-media methods continue
    through the real SummaryWriter.
    """

    def __init__(self, writer, wandb_run):
        self._writer = writer
        self._wandb_run = wandb_run

    def __getattr__(self, name):
        return getattr(self._writer, name)

    def add_image(
        self, tag, img_tensor, global_step=None, walltime=None,
        dataformats='CHW',
    ):
        del walltime
        if tag != WANDB_EVAL_MEDIA_KEY:
            return
        if dataformats != 'CHW':
            raise ValueError(
                'Selected W&B preview must use CHW, got {}'.format(dataformats)
            )
        import wandb
        image = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        self._wandb_run.log({
            WANDB_EVAL_STEP_KEY: int(global_step or 0),
            WANDB_EVAL_MEDIA_KEY: wandb.Image(image),
        })



def training(dataset, opt, pipe, testing_iterations, saving_iterations, debug_from):
    first_iter = 0
    tb_writer, wandb_run = prepare_output_and_logger(dataset, opt, pipe)

    gaussians = GaussianModel(dataset.sh_degree, dataset.order_args)
    env_map = EnvironmentMap(**dataset.env_args)
    scene = Scene(dataset, gaussians, env_map)

    gaussians.training_setup(opt)
    env_map.training_setup(opt)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    training_start_time = time.time()
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iteration_start_time = time.time()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        if opt.data_sample == 'order':
            viewpoint_cam: Camera = viewpoint_stack.pop(0)
        elif opt.data_sample == 'stack':
            choice = randint(0, len(viewpoint_stack) - 1)
            viewpoint_cam: Camera = viewpoint_stack.pop(choice)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if opt.lambda_flow > 0.0 and viewpoint_cam.flow is not None:
            flow_choice = randint(0, len(viewpoint_cam.flow) - 1)
            flow_pkg = viewpoint_cam.flow[flow_choice]
            flow_pkg = [a.cuda() if torch.is_tensor(a) else a for a in flow_pkg]
        else:
            flow_pkg = None

        render_pkg = render(viewpoint_cam, gaussians, env_map, pipe, flow_pkg=flow_pkg, render_objmask=opt.lambda_obj > 0.0)
        image, visibility_filter, radii = render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        dssim_loss = (1.0 - ssim(image, gt_image))
        depth_loss, flow_loss, obj_loss, sky_loss, sigma_loss, reg_loss, reg_sigma_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if opt.lambda_depth > 0.0:
            assert pipe.inv_depth, 'Depth-Any-Thing V2 monocular depth supervision should only support 1/d.'
            gt_depth = viewpoint_cam.depth.cuda()
            depth_loss = get_depth_loss(render_pkg['depth'], gt_depth)

        if opt.lambda_flow > 0.0 and flow_pkg is not None:
            flow_loss = get_flow_loss(render_pkg['img_flow'], flow_pkg, render_pkg['img_opacity'], dist=gaussians.scene_extent * 1e-3)
        
        if opt.lambda_obj > 0.0:
            gt_semantic = viewpoint_cam.semantic.cuda()
            pred_semantic = torch.clip(render_pkg['img_semantic'], 1e-3, 1.0 - 1e-3)
            obj_loss = torch.nn.functional.binary_cross_entropy(pred_semantic[0], (gt_semantic > 0).float())

        if opt.lambda_sky > 0.0:
            gt_sky = viewpoint_cam.sky.cuda().float()
            pred_sky = torch.clip(render_pkg['img_opacity'], 1e-3, 1.0 - 1e-3)
            sky_loss = torch.nn.functional.binary_cross_entropy(1.0 - pred_sky, gt_sky)
        
        if opt.lambda_reg > 0.0:
            deform_param = gaussians.xyz_deform_param[gaussians.obj_near_idx]  # P, K, 3, C
            reg_loss = torch.mean(torch.sum(torch.var(deform_param, dim=1), dim=-1))

        if opt.lambda_sigma > 0.0:
            time_sigma = torch.exp(gaussians.gs_time_sigma)
            sigma_loss = torch.mean(torch.abs(gaussians.frame_gap / torch.mean(time_sigma, dim=-1)))
            if opt.lambda_sigma_reg > 0.0:
                time_sigma = gaussians.gs_time_sigma[gaussians.obj_near_idx]  # P, K, 2
                reg_sigma_loss = torch.mean(torch.sum(torch.var(time_sigma, dim=1), dim=-1))

        loss = (1.0 - opt.lambda_dssim) * opt.lambda_l1 * Ll1 + opt.lambda_dssim * dssim_loss
        loss += depth_loss * opt.lambda_depth + flow_loss * opt.lambda_flow
        loss += sky_loss * opt.lambda_sky + obj_loss * opt.lambda_obj
        loss += sigma_loss * opt.lambda_sigma + reg_loss * opt.lambda_reg + reg_sigma_loss * opt.lambda_sigma_reg
        loss.backward()

        log_losses = {
            'total_loss': loss,
            'depth_loss': depth_loss,
            'l1_loss': Ll1,
            'dssim_loss': dssim_loss,
            'flow_loss': flow_loss,
            'obj_loss': obj_loss,
            'sky_loss': sky_loss,
            'sigma_loss': sigma_loss,
            'reg_loss': reg_loss,
            'reg_sigma_loss': reg_sigma_loss,
        }

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "pts": gaussians.get_pts_num,
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, log_losses, opt, testing_iterations, scene, render, (pipe,))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(render_pkg)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(opt.densify_scene_grad_threshold, opt.densify_obj_grad_threshold, 0.005, iteration > opt.opacity_reset_interval)
                elif gaussians.use_near_idx and iteration % opt.near_idx_reset_interval == 0:
                    gaussians.set_obj_near_idx()
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                env_map.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                env_map.optimizer.zero_grad(set_to_none=True)

            iteration_seconds = time.time() - iteration_start_time
            _report_training_runtime(
                tb_writer,
                iteration,
                opt.iterations,
                iteration_seconds,
                training_start_time,
                ema_loss_for_log,
                scene,
            )

    tb_writer.flush()
    tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()


def _wandb_is_enabled():
    value = os.getenv('WANDB_ENABLED', '0').strip().lower()
    if value in ('1', 'true', 'yes', 'on'):
        return True
    if value in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(
        "WANDB_ENABLED must be one of 1/0, true/false, yes/no, on/off"
    )


def _read_nonnegative_int_env(name, default):
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError("{} must be an integer".format(name)) from error
    if value < 0:
        raise ValueError(
            "{} must be greater than or equal to zero".format(name)
        )
    return value


def _read_bool_env(name, default):
    raw_value = os.getenv(name, '1' if default else '0').strip().lower()
    if raw_value in ('1', 'true', 'yes', 'on'):
        return True
    if raw_value in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(
        "{} must be one of 1/0, true/false, yes/no, on/off".format(name)
    )


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def _should_log_wandb_preview(iteration, total_iterations):
    if not _wandb_is_enabled():
        return False
    interval = _read_nonnegative_int_env('WANDB_EVAL_INTERVAL', 500)
    if interval == 0:
        return False
    return iteration % interval == 0 or iteration == total_iterations


def _should_log_wandb_scalars(iteration, total_iterations):
    interval = _read_nonnegative_int_env(
        'WANDB_SCALAR_LOG_INTERVAL', 10
    )
    if interval == 0:
        return False
    return (
        iteration == 1
        or iteration % interval == 0
        or iteration == total_iterations
    )


def _select_fixed_eval_cameras(cameras, camera_id, count):
    """Pick stable, time-spaced held-out views for GT/render comparisons."""
    if count <= 0:
        return []
    matching = [
        camera for camera in cameras if int(camera.cam_id) == camera_id
    ]
    candidates = matching if matching else list(cameras)
    candidates = sorted(
        candidates,
        key=lambda camera: (
            float(camera.time),
            str(camera.image_name),
            int(camera.uid),
        ),
    )
    if len(candidates) <= count:
        return candidates
    if count == 1:
        return [candidates[len(candidates) // 2]]
    indices = [
        int(round(position * (len(candidates) - 1) / float(count - 1)))
        for position in range(count)
    ]
    return [candidates[index] for index in indices]


def _build_gt_render_grid(image_pairs):
    """Build a CHW grid whose rows are [ground truth | rendered image]."""
    rows = []
    for ground_truth, rendered in image_pairs:
        height = min(ground_truth.shape[1], rendered.shape[1])
        width = min(ground_truth.shape[2], rendered.shape[2])
        ground_truth = ground_truth[:, :height, :width]
        rendered = rendered[:, :height, :width]
        rows.append(torch.cat((ground_truth, rendered), dim=2))
    if not rows:
        raise ValueError("At least one GT/render image pair is required")
    max_width = max(row.shape[2] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[2] < max_width:
            padding = torch.zeros(
                row.shape[0],
                row.shape[1],
                max_width - row.shape[2],
                dtype=row.dtype,
                device=row.device,
            )
            row = torch.cat((row, padding), dim=2)
        padded_rows.append(row)
    return torch.cat(padded_rows, dim=1)


def _get_lpips_metric(device):
    global _LPIPS_METRIC, _LPIPS_UNAVAILABLE
    if _LPIPS_UNAVAILABLE or not _read_bool_env('WANDB_EVAL_LPIPS', True):
        return None
    if _LPIPS_METRIC is None:
        try:
            from lpipsPyTorch import LPIPS
            _LPIPS_METRIC = LPIPS(net_type='alex').eval().to(device)
        except Exception as error:
            _LPIPS_UNAVAILABLE = True
            print(
                "[W&B][WARNING] LPIPS initialization failed; continuing "
                "without LPIPS: {}".format(error),
                flush=True,
            )
            return None
    return _LPIPS_METRIC


def _fixed_eval_cameras(scene):
    cache_name = '_wandb_fixed_eval_cameras'
    if not hasattr(scene, cache_name):
        camera_id = _read_nonnegative_int_env('WANDB_EVAL_CAMERA_ID', 0)
        count = _read_nonnegative_int_env('WANDB_EVAL_IMAGE_COUNT', 3)
        cameras = _select_fixed_eval_cameras(
            scene.getTestCameras(), camera_id, count
        )
        if not cameras:
            raise RuntimeError(
                "W&B preview logging requires at least one validation camera"
            )
        setattr(scene, cache_name, cameras)
        print(
            "[W&B] fixed eval camera_id={} views={}".format(
                camera_id,
                ','.join(camera.image_name for camera in cameras),
            ),
            flush=True,
        )
    return getattr(scene, cache_name)


@torch.no_grad()
def _report_wandb_preview(
    tb_writer, iteration, total_iterations, scene, render_func, render_args
):
    if not _should_log_wandb_preview(iteration, total_iterations):
        return

    cameras = _fixed_eval_cameras(scene)
    image_pairs = []
    l1_values = []
    psnr_values = []
    ssim_values = []
    lpips_values = []
    lpips_metric = None
    render_seconds = 0.0
    used_cuda = False

    for viewpoint in cameras:
        if viewpoint.original_image.is_cuda:
            torch.cuda.synchronize(viewpoint.original_image.device)
        render_start_time = time.time()
        render_pkg = render_func(
            viewpoint, scene.gaussians, scene.env_map, *render_args
        )
        image = torch.clamp(render_pkg['render'], 0.0, 1.0)
        if image.is_cuda:
            torch.cuda.synchronize(image.device)
            used_cuda = True
        render_seconds += time.time() - render_start_time
        gt_image = torch.clamp(
            viewpoint.original_image.to(image.device), 0.0, 1.0
        )
        height = min(gt_image.shape[1], image.shape[1])
        width = min(gt_image.shape[2], image.shape[2])
        gt_image = gt_image[:, :height, :width]
        image = image[:, :height, :width]
        l1_values.append(l1_loss(image, gt_image).double())
        psnr_values.append(
            psnr(image[None], gt_image[None]).mean().double()
        )
        ssim_values.append(ssim(image, gt_image).double())

        if lpips_metric is None:
            lpips_metric = _get_lpips_metric(image.device)
        if lpips_metric is not None:
            lpips_values.append(
                lpips_metric(
                    image[None] * 2.0 - 1.0,
                    gt_image[None] * 2.0 - 1.0,
                ).mean().double()
            )
        image_pairs.append(
            (
                gt_image.detach().cpu(),
                image.detach().cpu(),
            )
        )

    render_seconds = max(render_seconds, 1e-9)
    comparison_grid = _build_gt_render_grid(image_pairs)
    tb_writer.add_image(
        'Eval Images/fixed_front_gt_render',
        comparison_grid,
        global_step=iteration,
    )
    tb_writer.add_scalar(
        'Eval Images Metrics/fixed_front_l1',
        torch.stack(l1_values).mean(),
        iteration,
    )
    tb_writer.add_scalar(
        'Eval Images Metrics/fixed_front_psnr',
        torch.stack(psnr_values).mean(),
        iteration,
    )
    tb_writer.add_scalar(
        'Eval Images Metrics/fixed_front_ssim',
        torch.stack(ssim_values).mean(),
        iteration,
    )
    if lpips_values:
        tb_writer.add_scalar(
            'Eval Images Metrics/fixed_front_lpips',
            torch.stack(lpips_values).mean(),
            iteration,
        )
    tb_writer.add_scalar(
        'Eval Images Metrics/fixed_front_render_fps',
        len(cameras) / render_seconds,
        iteration,
    )
    tb_writer.flush()
    print(
        "[W&B][PREVIEW] iter={} logged {} fixed GT|Render pairs "
        "(PSNR={:.3f}, SSIM={:.4f})".format(
            iteration,
            len(cameras),
            torch.stack(psnr_values).mean().item(),
            torch.stack(ssim_values).mean().item(),
        ),
        flush=True,
    )
    if used_cuda:
        torch.cuda.empty_cache()


def _report_training_runtime(
    tb_writer,
    iteration,
    total_iterations,
    iteration_seconds,
    training_start_time,
    ema_loss,
    scene,
):
    elapsed_seconds = max(time.time() - training_start_time, 1e-9)
    iterations_per_second = iteration / elapsed_seconds
    eta_seconds = (total_iterations - iteration) / max(
        iterations_per_second, 1e-9
    )
    if _should_log_wandb_scalars(iteration, total_iterations):
        tb_writer.add_scalar(
            'Train Performance/iteration_wall_time_ms',
            iteration_seconds * 1000.0,
            iteration,
        )
        tb_writer.add_scalar(
            'Train Performance/iterations_per_second',
            iterations_per_second,
            iteration,
        )
        tb_writer.add_scalar(
            'Train Performance/eta_seconds', eta_seconds, iteration
        )
        tb_writer.add_scalar(
            'GPU Memory/allocated_mb',
            torch.cuda.memory_allocated() / (1024.0 ** 2),
            iteration,
        )
        tb_writer.add_scalar(
            'GPU Memory/reserved_mb',
            torch.cuda.memory_reserved() / (1024.0 ** 2),
            iteration,
        )
        tb_writer.add_scalar(
            'GPU Memory/peak_allocated_mb',
            torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            iteration,
        )
        tb_writer.add_scalar(
            'points/total_points', scene.gaussians.get_pts_num, iteration
        )
        tb_writer.add_scalar(
            'points/scene_points',
            scene.gaussians.get_scene_pts_num,
            iteration,
        )
        tb_writer.add_scalar(
            'points/obj_points', scene.gaussians.get_obj_pts_num, iteration
        )
        tb_writer.add_scalar(
            'model/active_sh_degree',
            scene.gaussians.active_sh_degree,
            iteration,
        )
        dynamic_lr_names = {
            'scene_xyz',
            'obj_xyz',
            'deform_xyz',
            'deform_background',
        }
        for param_group in scene.gaussians.optimizer.param_groups:
            if param_group.get('name') in dynamic_lr_names:
                tb_writer.add_scalar(
                    'learning_rate/{}'.format(param_group['name']),
                    param_group['lr'],
                    iteration,
                )
        for param_group in scene.env_map.optimizer.param_groups:
            tb_writer.add_scalar(
                'learning_rate/{}'.format(
                    param_group.get('name', 'env')
                ),
                param_group['lr'],
                iteration,
            )

    console_interval = _read_nonnegative_int_env(
        'ADGS_CONSOLE_LOG_INTERVAL', 100
    )
    should_log_console = console_interval > 0 and (
        iteration == 1
        or iteration % console_interval == 0
        or iteration == total_iterations
    )
    if should_log_console:
        scene_name = os.getenv('ADGS_SCENE_NAME', 'unknown-scene')
        physical_gpu = os.getenv('ADGS_PHYSICAL_GPU', '?')
        print(
            "\n[TRAIN] scene={} gpu={} iter={}/{} ({:.1f}%) "
            "loss={:.6f} points={} speed={:.2f}it/s elapsed={} "
            "eta={} gpu_mem={:.0f}MB".format(
                scene_name,
                physical_gpu,
                iteration,
                total_iterations,
                100.0 * iteration / total_iterations,
                ema_loss,
                scene.gaussians.get_pts_num,
                iterations_per_second,
                _format_duration(elapsed_seconds),
                _format_duration(eta_seconds),
                torch.cuda.memory_allocated() / (1024.0 ** 2),
            ),
            flush=True,
        )
        tb_writer.flush()


def _initialize_wandb(args, opt, pipe):
    """Start one scene-level W&B run and mirror the TensorBoard stream."""
    if not _wandb_is_enabled():
        return None

    required = ('WANDB_ENTITY', 'WANDB_PROJECT', 'WANDB_NAME')
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "W&B logging is enabled but variables are missing: {}".format(
                ', '.join(missing)
            )
        )
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B logging is enabled but wandb is not installed"
        ) from error

    split_type = os.getenv('WANDB_SPLIT_TYPE', 'SplatAD')
    dataset_type = os.getenv('WANDB_DATASET_TYPE', '')
    model_name = os.getenv('WANDB_MODEL_NAME', 'AD-GS')
    config = {
        'model': dict(vars(args)),
        'optimization': dict(vars(opt)),
        'pipeline': dict(vars(pipe)),
        'logging': {
            'eval_interval': _read_nonnegative_int_env(
                'WANDB_EVAL_INTERVAL', 500
            ),
            'eval_image_count': _read_nonnegative_int_env(
                'WANDB_EVAL_IMAGE_COUNT', 3
            ),
            'eval_camera_id': _read_nonnegative_int_env(
                'WANDB_EVAL_CAMERA_ID', 0
            ),
            'eval_lpips': _read_bool_env('WANDB_EVAL_LPIPS', True),
            'scalar_interval': _read_nonnegative_int_env(
                'WANDB_SCALAR_LOG_INTERVAL', 10
            ),
            'console_interval': _read_nonnegative_int_env(
                'ADGS_CONSOLE_LOG_INTERVAL', 100
            ),
        },
        'experiment': {
            'split_type': split_type,
            'dataset_type': dataset_type,
            'model_name': model_name,
            'scene': os.path.basename(os.path.normpath(args.source_path)),
        },
    }
    tags = [
        value for value in (split_type, dataset_type, model_name) if value
    ]
    run = wandb.init(
        entity=os.environ['WANDB_ENTITY'],
        project=os.environ['WANDB_PROJECT'],
        name=os.environ['WANDB_NAME'],
        group=os.getenv('WANDB_RUN_GROUP') or None,
        job_type='train',
        dir=args.model_path,
        config=config,
        tags=tags,
        sync_tensorboard=True,
        save_code=False,
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    run.define_metric(WANDB_EVAL_STEP_KEY)
    run.define_metric(
        WANDB_EVAL_MEDIA_KEY, step_metric=WANDB_EVAL_STEP_KEY
    )
    print(
        "[W&B] run={} mode={} url={}".format(
            run.name,
            os.getenv('WANDB_MODE', 'online'),
            getattr(run, 'url', None) or '(offline/no URL)',
        ),
        flush=True,
    )
    return run


def prepare_output_and_logger(args, opt, pipe):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Initialize W&B before constructing SummaryWriter so TensorBoard events
    # are mirrored into the scene-level W&B run from the first scalar onward.
    wandb_run = _initialize_wandb(args, opt, pipe)
    tb_writer = SummaryWriter(args.model_path)
    if wandb_run is not None:
        tb_writer = _SelectiveWandbMediaWriter(tb_writer, wandb_run)
    return tb_writer, wandb_run

def training_report(tb_writer, iteration, losses, opt, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if not _wandb_is_enabled() or _should_log_wandb_scalars(
        iteration, opt.iterations
    ):
        for l_name, l_value in losses.items():
            tb_writer.add_scalar(
                'train_loss_patches/{}'.format(l_name), l_value, iteration
            )

    _report_wandb_preview(
        tb_writer,
        iteration,
        opt.iterations,
        scene,
        renderFunc,
        renderArgs,
    )

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()}, 
            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if idx < 5:
                        if viewpoint.flow is not None:
                            gt_flow_pkg = viewpoint.flow[0]
                            gt_flow_pkg = [a.cuda() if torch.is_tensor(a) else a for a in gt_flow_pkg]
                            flow_time, K, R, T, gt_flow, gt_flow_vis = gt_flow_pkg
                        else:
                            gt_flow_pkg = None

                        render_pkg = renderFunc(viewpoint, scene.gaussians, scene.env_map, *renderArgs, flow_pkg=gt_flow_pkg, render_objmask=True)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_depth = viewpoint.depth.to('cuda')
                        gt_obj = (viewpoint.semantic.to('cuda') > 0).float()
                        error_map = torch.abs((image - gt_image))
                        
                        background = render_pkg['background']
                        foreground = render_pkg['foreground']
                        obj_map = render_pkg['img_semantic'].repeat(3, 1, 1)

                        if opt.lambda_depth > 0.0:
                            depthmap = get_scaled_shifted_depth(render_pkg['depth'], gt_depth)
                        else:
                            depthmap = render_pkg['depth']
                            depthmap = (depthmap - torch.min(depthmap)) / (torch.max(depthmap) - torch.min(depthmap))
                        depthmap = torch.clamp(depthmap, 0.0, 1.0)[None, ...]
                        gt_depth = torch.clamp(gt_depth, 0.0, 1.0)[None, ...]
                        
                        # flow = flow_to_img(render_pkg['img_flow'], gt_flow_vis) if gt_flow_pkg is not None else None
                        flow = flow_to_img(get_img_flow(render_pkg['img_flow'], gt_flow_pkg, dist=scene.gaussians.scene_extent * 1e-3), gt_flow_vis) if gt_flow_pkg is not None else None

                        if iteration == min(testing_iterations):
                            tb_writer.add_image(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image, global_step=iteration)
                            tb_writer.add_image(config['name'] + "_view_{}/depth_gt".format(viewpoint.image_name), gt_depth.repeat(3, 1, 1), global_step=iteration)
                            tb_writer.add_image(config['name'] + "_view_{}/sky_gt".format(viewpoint.image_name), viewpoint.sky[None].repeat(3, 1, 1).float(), global_step=iteration)
                            tb_writer.add_image(config['name'] + "_view_{}/obj_gt".format(viewpoint.image_name), gt_obj[None].repeat(3, 1, 1), global_step=iteration)
                            if gt_flow_pkg is not None:
                                tb_writer.add_image(config['name'] + "_view_{}/flow_gt".format(viewpoint.image_name), flow_to_img(gt_flow, gt_flow_vis), global_step=iteration)

                        tb_writer.add_image(
                            config['name'] + "_view_{}/gt_render".format(
                                viewpoint.image_name
                            ),
                            _build_gt_render_grid(((gt_image, image),)),
                            global_step=iteration,
                        )
                        tb_writer.add_image(config['name'] + "_view_{}/render".format(viewpoint.image_name), image, global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/opacity".format(viewpoint.image_name), render_pkg['img_opacity'].repeat(3, 1, 1), global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depthmap.repeat(3, 1, 1), global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/foreground".format(viewpoint.image_name), foreground, global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/background".format(viewpoint.image_name), background, global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/error_map".format(viewpoint.image_name), error_map, global_step=iteration)
                        tb_writer.add_image(config['name'] + "_view_{}/obj".format(viewpoint.image_name), obj_map, global_step=iteration)
                        if flow is not None:
                            tb_writer.add_image(config['name'] + "_view_{}/flow".format(viewpoint.image_name), flow, global_step=iteration)
                    else:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, scene.env_map, *renderArgs)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        error_map = torch.abs((image - gt_image))
                    
                    l1_test += error_map.mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_scalar('points/total_points', scene.gaussians.get_pts_num, iteration)
            tb_writer.add_scalar('points/scene_points', scene.gaussians.get_scene_pts_num, iteration)
            tb_writer.add_scalar('points/obj_points', scene.gaussians.get_obj_pts_num, iteration)

        print("\n[ITER {}] Points: Total {} Scene {} Object {}".format(iteration, scene.gaussians.get_pts_num, scene.gaussians.get_scene_pts_num, scene.gaussians.get_obj_pts_num))

        torch.cuda.empty_cache()

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', '-c', type=str, default=None)
    config_path = parser.parse_known_args()[0].config
    if config_path is not None:
        assert os.path.exists(config_path)
        print("Find Config:", config_path)
        config = get_config(config_path)
    else:
        config = None
    lp = ModelParams(parser, config)
    op = OptimizationParams(parser, config)
    pp = PipelineParams(parser, config)
    parser.add_argument('--ip', type=str, default="localhost")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[60_000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.test_iterations += list(range(10_000, args.iterations, 10_000))
    args.test_iterations.append(args.iterations)

    args.data_device = "cuda:0" if args.data_device == 'cuda' else args.data_device
    torch.cuda.set_device(args.data_device)
    
    if not args.quiet:
        print(vars(args))

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.debug_from)

    # All done
    print("\nTraining complete.")
