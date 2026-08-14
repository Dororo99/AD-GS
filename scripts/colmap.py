import os
import struct
import shutil
import sqlite3
import argparse
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from plyfile import PlyElement, PlyData
from tqdm import tqdm

try:
    from scripts.prior_storage import load_mask_prior
except ImportError:  # Direct ``python scripts/colmap.py`` execution.
    from prior_storage import load_mask_prior

# try "conda install colmap -c conda-forge" if you have several problem in installing colmap.
# this script is mainly borrowed from StreetGS. https://github.com/zju3dv/street_gaussians

def print_notice(text):
    print("\033[32m{}\033[0m".format(text))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    bundled_colmap = '/venv/ad-gs/bin/colmap'
    default_colmap = os.environ.get(
        'COLMAP_BIN', bundled_colmap if os.path.isfile(bundled_colmap) else 'colmap'
    )
    parser.add_argument('--cmd', default=default_colmap, help='command for colmap')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--split_mode', default='nvs-75')
    parser.add_argument(
        '--cam', type=int, default=None,
        help='expected camera count (auto-inferred from camera_ids when omitted)',
    )
    args = parser.parse_args()
    return args

def storePly(path, xyz, rgb, t=None, dynamic=None):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if t is not None:
        dtype.append(('t', 'f4'))
    if dynamic is not None:
        dtype.append(('dy', 'f4'))
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    if t is not None:
        attributes = np.concatenate([attributes, t], axis=-1)
    if dynamic is not None:
        attributes = np.concatenate([attributes, dynamic], axis=-1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def get_val_frames(num_frames, test_every=None, train_every=None):
    assert train_every is None or test_every is None
    if train_every is None:
        val_frames = set(np.arange(test_every, num_frames, test_every))
    else:
        train_frames = set(np.arange(0, num_frames, train_every))
        val_frames = (set(np.arange(num_frames)) - train_frames) if train_every > 1 else train_frames

    return list(val_frames)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors


def _metadata_camera_ids(meta, num_images, num_cam, legacy_num_cam, dataset_name):
    if 'camera_ids' in meta.files:
        camera_ids = np.asarray(meta['camera_ids'], dtype=np.int64).reshape(-1)
        if len(camera_ids) != num_images:
            raise ValueError(
                f'{dataset_name}: camera_ids has {len(camera_ids)} entries, '
                f'but image/ has {num_images} files'
            )
        inferred = len(dict.fromkeys(camera_ids.tolist()))
        if num_cam is not None and num_cam != inferred:
            raise ValueError(
                f'{dataset_name}: --cam {num_cam} disagrees with metadata '
                f'camera_ids ({inferred} cameras)'
            )
        return camera_ids

    resolved = legacy_num_cam if num_cam is None else num_cam
    if resolved is None or resolved <= 0:
        raise ValueError(f'{dataset_name}: a positive camera count is required')
    if num_images % resolved != 0:
        raise ValueError(
            f'{dataset_name}: {num_images} images are not divisible by the '
            f'legacy camera count {resolved}; regenerate metadata with camera_ids'
        )
    return np.arange(num_images, dtype=np.int64) % resolved


def _extract_intrinsics(K):
    K = np.asarray(K)
    if K.ndim == 2 and K.shape[1] >= 4:
        return K[:, 0], K[:, 1], K[:, 2], K[:, 3]
    if K.ndim == 3 and K.shape[1:] == (3, 3):
        return K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]
    raise ValueError(f'Unsupported intrinsic array shape: {K.shape}')


def _prepare_colmap_meta_ad(path, colmap_path, metadata_name, num_cam,
                            legacy_num_cam, dataset_name):
    colmap_image_dir = os.path.join(colmap_path, 'images')
    colmap_mask_dir = os.path.join(colmap_path, 'masks')
    os.makedirs(colmap_image_dir, exist_ok=True)
    os.makedirs(colmap_mask_dir, exist_ok=True)

    image_dir = os.path.join(path, 'image')
    image_names = sorted(os.listdir(image_dir))
    meta = np.load(os.path.join(path, metadata_name), allow_pickle=True)
    K = np.asarray(meta['K'])
    R = np.asarray(meta['R'])
    T = np.asarray(meta['T'])
    is_val_list = np.asarray(meta['is_val_list'], dtype=np.bool_).reshape(-1)
    num_images = len(image_names)
    for field_name, values in (
        ('K', K), ('R', R), ('T', T), ('is_val_list', is_val_list),
    ):
        if len(values) != num_images:
            raise ValueError(
                f'{dataset_name}: {field_name} has {len(values)} entries, '
                f'but image/ has {num_images} files'
            )

    raw_camera_ids = _metadata_camera_ids(
        meta, num_images, num_cam, legacy_num_cam, dataset_name
    )
    raw_camera_order = list(dict.fromkeys(raw_camera_ids.tolist()))
    camera_id_map = {
        int(raw_id): contiguous_id
        for contiguous_id, raw_id in enumerate(raw_camera_order)
    }
    camera_ids = np.asarray(
        [camera_id_map[int(raw_id)] for raw_id in raw_camera_ids],
        dtype=np.int64,
    )
    camera_count = len(camera_id_map)
    for camera_id in range(camera_count):
        os.makedirs(os.path.join(colmap_image_dir, str(camera_id)), exist_ok=True)
        os.makedirs(os.path.join(colmap_mask_dir, str(camera_id)), exist_ok=True)

    fx, fy, cx, cy = _extract_intrinsics(K)
    indices = []
    selected_camera_ids = []
    widths = []
    heights = []
    image_index_by_name = {}
    for global_idx, image_name in enumerate(tqdm(image_names, desc='Reading')):
        if is_val_list[global_idx]:
            continue
        camera_id = int(camera_ids[global_idx])
        relative_name = f'{camera_id}/{image_name}'
        source_image = os.path.join(image_dir, image_name)
        shutil.copy(source_image, os.path.join(colmap_image_dir, relative_name))
        with Image.open(source_image) as image:
            width, height = image.size

        stem = os.path.splitext(image_name)[0]
        semantic_path = os.path.join(path, 'semantic', f'mask_{stem}.npy')
        sky_path = os.path.join(path, 'sky', f'mask_{stem}.npy')
        semantic_mask = load_mask_prior(semantic_path, "semantic") == 0
        sky_mask = load_mask_prior(sky_path, "sky") == 0
        if semantic_mask.shape != (height, width) or sky_mask.shape != (height, width):
            raise ValueError(
                f'{dataset_name}: mask/image size mismatch for {image_name}: '
                f'image={(height, width)}, semantic={semantic_mask.shape}, '
                f'sky={sky_mask.shape}'
            )
        mask = np.logical_and(semantic_mask, sky_mask)[..., None]
        mask = np.uint8(np.repeat(mask, 3, axis=-1) * 255)
        # COLMAP resolves masks by appending ``.png`` to the complete image
        # filename (for example ``0/000000.jpg.png``), not by replacing the
        # image suffix.  Using the image filename directly silently disables
        # semantic/sky masking during feature extraction.
        Image.fromarray(mask).save(
            os.path.join(colmap_mask_dir, relative_name + '.png')
        )

        local_idx = len(indices)
        indices.append(global_idx)
        selected_camera_ids.append(camera_id)
        widths.append(width)
        heights.append(height)
        image_index_by_name[relative_name] = local_idx

    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError(f'{dataset_name}: the training split has no images')
    return {
        'cx': np.asarray(cx)[indices],
        'cy': np.asarray(cy)[indices],
        'fx': np.asarray(fx)[indices],
        'fy': np.asarray(fy)[indices],
        'R': R[indices],
        'T': T[indices],
        'width': np.asarray(widths, dtype=np.int64),
        'height': np.asarray(heights, dtype=np.int64),
        'camera_ids': np.asarray(selected_camera_ids, dtype=np.int64),
        'num_cameras': camera_count,
        'image_index_by_name': image_index_by_name,
    }

def run_colmap(colmap_path, camera_meta, colmap_cmd='colmap', use_gpu=False,
               cam_num=None):
    mask_dir = os.path.join(colmap_path, 'masks')
    image_dir = os.path.join(colmap_path, 'images')
    assert os.path.exists(image_dir), 'Cannot find ' + image_dir
    assert os.path.exists(mask_dir), 'Cannot find ' + mask_dir
    print(image_dir)

    print_notice('Feature Extraction')
    gpu_option = '' if use_gpu else ' --SiftExtraction.use_gpu 0'
    ret = os.system(
        f'{colmap_cmd} feature_extractor \
        --ImageReader.mask_path {mask_dir} \
        --ImageReader.camera_model PINHOLE  \
        --ImageReader.single_camera_per_folder 1 \
        --database_path {colmap_path}/database.db \
        --image_path {image_dir}' + gpu_option
    )
    assert ret == 0, 'There might be several mistakes in feature extraction.'
    print_notice('Feature Extraction Done')

    print_notice('Process camera meta')
    model_dir = os.path.join(colmap_path, 'created/sparse/model')
    os.makedirs(model_dir, exist_ok=True)

    db_connect = sqlite3.connect(os.path.join(colmap_path, 'database.db'))
    c = db_connect.cursor()
    c.execute('SELECT image_id, name, camera_id FROM images')
    image_rows = c.fetchall()
    expected_camera_count = int(camera_meta.get('num_cameras', cam_num or 1))
    if cam_num is not None and cam_num != expected_camera_count:
        raise ValueError(
            f'--cam {cam_num} disagrees with prepared metadata '
            f'({expected_camera_count} cameras)'
        )
    camera_ids = np.asarray(
        camera_meta.get(
            'camera_ids',
            np.arange(len(camera_meta['R']), dtype=np.int64)
            % expected_camera_count,
        ),
        dtype=np.int64,
    )
    if len(camera_ids) != len(camera_meta['R']):
        raise ValueError('camera_ids and camera poses have different lengths')
    image_index_by_name = camera_meta.get('image_index_by_name', {})
    db_camera_by_folder = {}
    row_indices = {}
    for image_id, image_name, db_camera_id in image_rows:
        normalized_name = image_name.replace('\\', '/')
        folder_name = normalized_name.split('/', 1)[0]
        try:
            folder_camera_id = int(folder_name)
        except ValueError as exc:
            raise ValueError(
                f'COLMAP image {image_name!r} is not inside a numeric camera folder'
            ) from exc
        if folder_camera_id in db_camera_by_folder:
            if db_camera_by_folder[folder_camera_id] != db_camera_id:
                raise ValueError(
                    f'COLMAP assigned multiple DB cameras to folder {folder_name}'
                )
        else:
            db_camera_by_folder[folder_camera_id] = db_camera_id

        if image_index_by_name:
            if normalized_name not in image_index_by_name:
                raise ValueError(
                    f'COLMAP returned unknown prepared image {normalized_name!r}'
                )
            idx = int(image_index_by_name[normalized_name])
        else:
            idx = int(os.path.splitext(normalized_name.rsplit('/', 1)[-1])[0])
        if int(camera_ids[idx]) != folder_camera_id:
            raise ValueError(
                f'Camera folder {folder_camera_id} disagrees with metadata '
                f'camera_id {camera_ids[idx]} for {normalized_name}'
            )
        row_indices[image_id] = idx

    if set(db_camera_by_folder) != set(range(expected_camera_count)):
        raise ValueError(
            f'Expected camera folders 0..{expected_camera_count - 1}, got '
            f'{sorted(db_camera_by_folder)}'
        )
    with open(os.path.join(model_dir, 'images.txt'), 'w') as f:
        R = camera_meta['R']
        T = camera_meta['T']
        for img_id, img_name, db_camera_id in image_rows:
            idx = row_indices[img_id]
            R_quat = Rotation.from_matrix(R[idx]).as_quat()  # The returned value is in scalar-last (x, y, z, w) format.
            R_quat[0], R_quat[1], R_quat[2], R_quat[3] = R_quat[3], R_quat[0], R_quat[1], R_quat[2]
            rt = np.concatenate([R_quat, T[idx]], axis=0)
            # Use COLMAP's real positive database camera id. Folder ids are
            # zero-based only in the staging directory.
            f.write(
                f'{img_id} ' + ' '.join([str(a) for a in rt.tolist()])
                + f' {db_camera_id} {img_name}\n\n'
            )
    
    with open(os.path.join(model_dir, 'cameras.txt'), 'w') as f:
        for folder_camera_id in range(expected_camera_count):
            members = np.flatnonzero(camera_ids == folder_camera_id)
            if not len(members):
                raise ValueError(f'Camera {folder_camera_id} has no training images')
            idx = int(members[0])
            cx = float(camera_meta['cx'][idx])
            cy = float(camera_meta['cy'][idx])
            fx = float(camera_meta['fx'][idx])
            fy = float(camera_meta['fy'][idx])
            width = int(camera_meta.get('width', 2 * camera_meta['cx'])[idx])
            height = int(camera_meta.get('height', 2 * camera_meta['cy'])[idx])
            for field_name in ('cx', 'cy', 'fx', 'fy'):
                if not np.allclose(
                    np.asarray(camera_meta[field_name])[members],
                    camera_meta[field_name][idx],
                    rtol=1e-5, atol=1e-4,
                ):
                    raise ValueError(
                        f'Camera {folder_camera_id} has varying {field_name}'
                    )
            if 'width' in camera_meta and not np.all(
                np.asarray(camera_meta['width'])[members] == width
            ):
                raise ValueError(f'Camera {folder_camera_id} has varying width')
            if 'height' in camera_meta and not np.all(
                np.asarray(camera_meta['height'])[members] == height
            ):
                raise ValueError(f'Camera {folder_camera_id} has varying height')
            db_camera_id = int(db_camera_by_folder[folder_camera_id])
            if db_camera_id <= 0:
                raise ValueError(f'Invalid COLMAP camera id {db_camera_id}')
            f.write(
                f'{db_camera_id} PINHOLE {width} {height} '
                f'{fx} {fy} {cx} {cy}\n'
            )
            params = np.array([fx, fy, cx, cy]).astype(np.float64)
            c.execute(
                'UPDATE cameras SET model = ?, width = ?, height = ?, params = ? '
                'WHERE camera_id = ?',
                (1, width, height, params.tobytes(), db_camera_id),
            )
        
    db_connect.commit()
    db_connect.close()
    print_notice('Process camera meta Done')

    print_notice('Exhaustive Match')
    gpu_option = '' if use_gpu else ' --SiftMatching.use_gpu 0'
    ret = os.system(
        f'{colmap_cmd} exhaustive_matcher \
        --database_path {colmap_path}/database.db' + gpu_option
    )
    assert ret == 0, 'There might be several mistakes in exhaustive match'
    print_notice('Exhaustive Match Done')

    print_notice('Point Triangulate')
    triangulated_dir = os.path.join(colmap_path, 'triangulated/sparse/model')
    os.makedirs(triangulated_dir, exist_ok=True)
    os.system('touch {}'.format(os.path.join(model_dir, 'points3D.txt')))
    ret = os.system(f'{colmap_cmd} point_triangulator \
        --database_path {colmap_path}/database.db \
        --image_path {image_dir} \
        --input_path {model_dir} \
        --output_path {triangulated_dir} \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_principal_point 0 \
        --Mapper.max_extra_param 0 \
        --clear_points 0 \
        --Mapper.ba_global_max_num_iterations 30 \
        --Mapper.filter_max_reproj_error 4 \
        --Mapper.filter_min_tri_angle 0.5 \
        --Mapper.tri_min_angle 0.5 \
        --Mapper.tri_ignore_two_view_tracks 1 \
        --Mapper.tri_complete_max_reproj_error 4 \
        --Mapper.tri_continue_max_angle_error 4')
    assert ret == 0, 'There might be several mistakes in point triangulate'
    print_notice('Point Triangulate Done')

def _prepare_colmap_meta_waymo_legacy(path, colmap_path, num_cam=1):
    colmap_image_dir = os.path.join(colmap_path, 'images')
    os.makedirs(colmap_image_dir, exist_ok=True)
    colmap_mask_dir = os.path.join(colmap_path, 'masks')
    os.makedirs(colmap_mask_dir, exist_ok=True)

    for i in range(num_cam):
        os.makedirs(os.path.join(colmap_image_dir, f"{i}"), exist_ok=True)
        os.makedirs(os.path.join(colmap_mask_dir, f"{i}"), exist_ok=True)

    image_path = os.path.join(path, 'image')
    meta = np.load(os.path.join(path, "cameras.npz"), allow_pickle=True)
    K, R, T = meta['K'], meta['R'], meta['T']
    time_stamps = meta['time_stamps']
    is_val_list = meta['is_val_list']
    cur_idx = 0
    for idx, img_path in enumerate(tqdm(list(sorted(os.listdir(image_path))), desc='Reading')):
        if is_val_list[idx]:
            continue
        cam_id = idx % num_cam
        shutil.copy(os.path.join(image_path, img_path), os.path.join(colmap_image_dir, f"{cam_id}", '{:06d}.jpg'.format(cur_idx)))
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + img_path.split(".")[0] + ".npy")
        semantic_mask = load_mask_prior(semantic_path, "semantic") == 0
        sky_mask = load_mask_prior(sky_path, "sky") == 0
        mask = np.logical_and(semantic_mask, sky_mask)[..., None]
        # mask = semantic_mask[..., None]
        mask = np.uint8(np.repeat(mask, 3, axis=-1) * 255)
        Image.fromarray(mask).save(os.path.join(colmap_mask_dir, f"{cam_id}", '{:06d}.jpg'.format(cur_idx)))
        cur_idx += 1
    select_list = np.logical_not(is_val_list)
    return {
        'cx': K[select_list, 2],
        'cy': K[select_list, 3],
        'fx': K[select_list, 0],
        'fy': K[select_list, 1],
        'R': R[select_list],
        'T': T[select_list],
    }


def prepare_colmap_meta_waymo(path, colmap_path, num_cam=None):
    return _prepare_colmap_meta_ad(
        path=path,
        colmap_path=colmap_path,
        metadata_name='cameras.npz',
        num_cam=num_cam,
        legacy_num_cam=1,
        dataset_name='Waymo',
    )

def prepare_colmap_meta_kitti(path, colmap_path, split_mode='nvs-75', num_cam=2):
    colmap_image_dir = os.path.join(colmap_path, 'images')
    os.makedirs(colmap_image_dir, exist_ok=True)
    colmap_mask_dir = os.path.join(colmap_path, 'masks')
    os.makedirs(colmap_mask_dir, exist_ok=True)

    for i in range(num_cam):
        os.makedirs(os.path.join(colmap_image_dir, f"{i}"), exist_ok=True)
        os.makedirs(os.path.join(colmap_mask_dir, f"{i}"), exist_ok=True)

    image_path = os.path.join(path, 'image')
    meta = np.load(os.path.join(path, "poses.npz"), allow_pickle=True)
    R, T = meta['R'], meta['T']
    height = float(meta['height'])
    width = float(meta['width'])
    focal = meta['focal']

    if split_mode == 'nvs-25':
        i_test = get_val_frames(R.shape[0] // 2, train_every=4)
    elif split_mode == 'nvs-50':
        i_test = get_val_frames(R.shape[0] // 2, test_every=2)
    elif split_mode == 'nvs-75':
        i_test = get_val_frames(R.shape[0] // 2, test_every=4)
    else:
        raise ValueError("No such split method: " + split_mode)
    
    indices = []
    selected_camera_ids = []
    image_index_by_name = {}
    cur_idx = 0
    for idx, img_path in enumerate(tqdm(list(sorted(os.listdir(image_path))), desc='Reading')):
        if idx // 2 in i_test:
            continue
        cam_id = idx % num_cam
        shutil.copy(os.path.join(image_path, img_path), os.path.join(colmap_image_dir, f"{cam_id}", '{:06d}.png'.format(cur_idx)))
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + img_path.split(".")[0] + ".npy")
        semantic_mask = load_mask_prior(semantic_path, "semantic") == 0
        sky_mask = load_mask_prior(sky_path, "sky") == 0
        mask = np.logical_and(semantic_mask, sky_mask)[..., None]
        # mask = semantic_mask[..., None]
        mask = np.uint8(np.repeat(mask, 3, axis=-1) * 255)
        output_name = '{:06d}.png'.format(cur_idx)
        Image.fromarray(mask).save(os.path.join(colmap_mask_dir, f"{cam_id}", output_name))
        image_index_by_name[f'{cam_id}/{output_name}'] = len(indices)
        cur_idx += 1
        indices.append(idx)
        selected_camera_ids.append(cam_id)

    
    return {
        'cx': np.full((len(indices)), fill_value=width / 2),
        'cy': np.full((len(indices)), fill_value=height / 2),
        'fx': np.full((len(indices)), fill_value=focal),
        'fy': np.full((len(indices)), fill_value=focal),
        'R': R[indices],
        'T': T[indices],
        'width': np.full((len(indices)), fill_value=int(width), dtype=np.int64),
        'height': np.full((len(indices)), fill_value=int(height), dtype=np.int64),
        'camera_ids': np.asarray(selected_camera_ids, dtype=np.int64),
        'num_cameras': num_cam,
        'image_index_by_name': image_index_by_name,
    }

def _prepare_colmap_meta_nuscenes_legacy(path, colmap_path, num_cam=3):
    colmap_image_dir = os.path.join(colmap_path, 'images')
    os.makedirs(colmap_image_dir, exist_ok=True)
    colmap_mask_dir = os.path.join(colmap_path, 'masks')
    os.makedirs(colmap_mask_dir, exist_ok=True)

    for i in range(num_cam):
        os.makedirs(os.path.join(colmap_image_dir, f"{i}"), exist_ok=True)
        os.makedirs(os.path.join(colmap_mask_dir, f"{i}"), exist_ok=True)

    image_path = os.path.join(path, 'image')
    meta = np.load(os.path.join(path, "meta.npz"), allow_pickle=True)
    K, R, T = meta['K'], meta['R'], meta['T']
    time_stamps = meta['time_stamps']
    is_val_list = meta['is_val_list']
    cur_idx = 0
    for idx, img_path in enumerate(tqdm(list(sorted(os.listdir(image_path))), desc='Reading')):
        if is_val_list[idx]:
            continue
        cam_id = idx % num_cam
        shutil.copy(os.path.join(image_path, img_path), os.path.join(colmap_image_dir, f"{cam_id}", '{:06d}.png'.format(cur_idx)))
        semantic_path = os.path.join(path, "semantic", 'mask_' + img_path.split(".")[0] + ".npy")
        sky_path = os.path.join(path, "sky", 'mask_' + img_path.split(".")[0] + ".npy")
        semantic_mask = load_mask_prior(semantic_path, "semantic") == 0
        sky_mask = load_mask_prior(sky_path, "sky") == 0
        mask = np.logical_and(semantic_mask, sky_mask)[..., None]
        # mask = semantic_mask[..., None]
        mask = np.uint8(np.repeat(mask, 3, axis=-1) * 255)
        Image.fromarray(mask).save(os.path.join(colmap_mask_dir, f"{cam_id}", '{:06d}.png'.format(cur_idx)))
        cur_idx += 1
    select_list = np.logical_not(is_val_list)
    return {
        'cx': K[select_list, 0, 2],
        'cy': K[select_list, 1, 2],
        'fx': K[select_list, 0, 0],
        'fy': K[select_list, 1, 1],
        'R': R[select_list],
        'T': T[select_list],
    }


def prepare_colmap_meta_nuscenes(path, colmap_path, num_cam=None):
    return _prepare_colmap_meta_ad(
        path=path,
        colmap_path=colmap_path,
        metadata_name='meta.npz',
        num_cam=num_cam,
        legacy_num_cam=3,
        dataset_name='nuScenes/AV2',
    )

if __name__ == '__main__':
    args = get_args()
    colmap_dir = os.path.join(args.path, 'colmap')

    ply_path = os.path.join(args.path, 'colmap.ply')
    if os.path.exists(os.path.join(args.path, "cameras.npz")):
        os.makedirs(colmap_dir, exist_ok=True)
        print("Found cameras.npz file, assuming Waymo data set!")
        camera_meta = prepare_colmap_meta_waymo(args.path, colmap_dir, num_cam=args.cam)
    elif os.path.exists(os.path.join(args.path, 'poses.npz')):
        print('Found poses.npz file, assuming KITTI or vKITTI data set!')
        colmap_dir = os.path.join(args.path, 'colmap-{}'.format(args.split_mode.split('-')[-1]))
        os.makedirs(colmap_dir, exist_ok=True)
        kitti_num_cam = 2 if args.cam is None else args.cam
        camera_meta = prepare_colmap_meta_kitti(
            args.path, colmap_dir, split_mode=args.split_mode,
            num_cam=kitti_num_cam,
        )
        ply_path = os.path.join(args.path, 'colmap-{}.ply'.format(args.split_mode.split('-')[-1]))
    elif os.path.exists(os.path.join(args.path, "meta.npz")):
        os.makedirs(colmap_dir, exist_ok=True)
        print("Found meta.npz file, assuming nuScenes or AV2 data set!")
        camera_meta = prepare_colmap_meta_nuscenes(args.path, colmap_dir, num_cam=args.cam)
    else:
        assert False, 'Could not recognize scene type!'

    run_colmap(colmap_path=colmap_dir, camera_meta=camera_meta, colmap_cmd=args.cmd, use_gpu=args.use_gpu, cam_num=args.cam)
    xyz, rgb, _ = read_points3D_binary(os.path.join(colmap_dir, 'triangulated/sparse/model/points3D.bin'))
    storePly(ply_path, xyz=xyz, rgb=rgb)
    print('SfM pointcloud:', ply_path, 'pts:', xyz.shape[0])
