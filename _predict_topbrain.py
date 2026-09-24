import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import ants
import nibabel as nib
import numpy as np
import SimpleITK as sitk

# --- ENVIRONMENT SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
DUMMY_RAW_DIR = os.path.join(CURRENT_DIR, 'dummy_raw')
DUMMY_PREPROCESSED_DIR = os.path.join(CURRENT_DIR, 'dummy_preprocessed')

os.environ['nnUNet_results'] = MODELS_DIR
os.environ['nnUNet_raw'] = DUMMY_RAW_DIR
os.environ['nnUNet_preprocessed'] = DUMMY_PREPROCESSED_DIR

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

TEMPLATE_PATH = os.path.join(CURRENT_DIR, 'MNI', '1mm', 'ct_rorden_lps.nii')
DEFAULT_CONFIGURATION = '3d_fullres'
DEFAULT_PLANS = 'nnUNetResEncUNetMPlans'
DEFAULT_TRAINER = 'nnUNetTrainerNoMirroring'
DEFAULT_FOLD = '0'
DEFAULT_CHECKPOINT = 'checkpoint_final.pth'
MODEL_INPUT_ORIENTATION = 'LPS'
CASCADE_CROP_THRESHOLD_GIB = 8.0
CASCADE_CROP_MARGIN_MM = 20.0
CASCADE_CROP_MIN_COMPONENT_FRACTION = 0.01
CASCADE_TILE_TARGET_GIB = 8.0
CASCADE_TILE_CONTEXT_MM = 10.0
CASCADE_TILE_MAX_TILES = 64

MODEL_REGISTRY = {
    'base': {
        'output_kind': 'multiclass',
        'cta': {'dataset': '501', 'name': 'TopBrainCTA', 'crop_head': True},
        'mra': {'dataset': '502', 'name': 'TopBrainMRA', 'crop_head': False},
    },
    'binary': {
        'output_kind': 'binary',
        'cta': {'dataset': '511', 'name': 'TopBrainCTABinary', 'crop_head': True},
        'mra': {'dataset': '512', 'name': 'TopBrainMRABinary', 'crop_head': False},
    },
    'cascade': {
        'output_kind': 'multiclass',
        'cta': {
            'binary_dataset': '511',
            'cascade_dataset': '521',
            'name': 'TopBrainCTABinaryToMulticlass',
            'crop_head': True,
        },
        'mra': {
            'binary_dataset': '512',
            'cascade_dataset': '522',
            'name': 'TopBrainMRABinaryToMulticlass',
            'crop_head': False,
            'crop_cascade_to_binary': True,
        },
    },
}


def configure_environment():
    os.environ['nnUNet_results'] = MODELS_DIR
    os.environ['nnUNet_raw'] = DUMMY_RAW_DIR
    os.environ['nnUNet_preprocessed'] = DUMMY_PREPROCESSED_DIR
    os.environ.setdefault('nnUNet_compile', 'f')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.makedirs(DUMMY_RAW_DIR, exist_ok=True)
    os.makedirs(DUMMY_PREPROCESSED_DIR, exist_ok=True)


def _find_nnunet_predict_executable():
    """Find nnUNetv2_predict on PATH or beside the active Python interpreter."""
    executable = shutil.which("nnUNetv2_predict")
    if executable:
        return executable

    python_bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    for executable_name in ("nnUNetv2_predict", "nnUNetv2_predict.exe"):
        candidate = os.path.join(python_bin_dir, executable_name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "nnUNetv2_predict is not installed for the active Python interpreter "
        f"({sys.executable}). Install PyTorch first and then nnunetv2 in that "
        "same environment."
    )


def _get_nnunet_device():
    """Choose an nnU-Net device, with an environment-variable override."""
    device = os.environ.get("TOPBRAIN_NNUNET_DEVICE", "").strip().lower()
    if not device:
        device = "mps" if sys.platform == "darwin" else "cuda"
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            "TOPBRAIN_NNUNET_DEVICE must be one of: cpu, cuda, or mps"
        )
    return device


def _get_float_env(name, default):
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == '':
        return default
    try:
        return float(raw_value)
    except ValueError:
        print(f"  [Warning] Ignoring invalid {name}={raw_value!r}; using {default:g}.")
        return default


def _get_fixed_env():
    env = os.environ.copy()
    conda_prefix = sys.prefix
    conda_bin = os.path.join(conda_prefix, 'bin')
    current_path = env.get('PATH', '')
    env['PATH'] = f'{conda_bin}:{current_path}' if current_path else conda_bin

    lib_path = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6')
    if os.path.exists(lib_path):
        current_preload = env.get('LD_PRELOAD', '')
        env['LD_PRELOAD'] = f'{lib_path}:{current_preload}' if current_preload else lib_path
        lib_dir = os.path.dirname(lib_path)
        current_ld = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = f'{lib_dir}:{current_ld}' if current_ld else lib_dir
    else:
        print(f"  [Warning] Could not find Conda libstdc++ at {lib_path}")

    if _get_nnunet_device() == "mps":
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    return env


def _find_model_output_dir(dataset_id):
    dataset_prefix = f'Dataset{dataset_id}_'
    dataset_dirs = [
        os.path.join(MODELS_DIR, name)
        for name in sorted(os.listdir(MODELS_DIR))
        if name.startswith(dataset_prefix)
    ] if os.path.isdir(MODELS_DIR) else []
    if not dataset_dirs:
        return None

    model_dir = os.path.join(
        dataset_dirs[0],
        f'{DEFAULT_TRAINER}__{DEFAULT_PLANS}__{DEFAULT_CONFIGURATION}',
    )
    return model_dir if os.path.isdir(model_dir) else None


def _read_json_if_exists(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _get_dataset_num_labels(dataset_id):
    model_dir = _find_model_output_dir(dataset_id)
    dataset_json = _read_json_if_exists(
        os.path.join(model_dir, 'dataset.json') if model_dir else None
    )
    labels = dataset_json.get('labels', {}) if dataset_json else {}
    return max(1, len(labels))


def _get_prediction_array_spec(dataset_id):
    model_dir = _find_model_output_dir(dataset_id)
    plans_json = _read_json_if_exists(
        os.path.join(model_dir, 'plans.json') if model_dir else None
    )
    configuration = (
        plans_json.get('configurations', {}).get(DEFAULT_CONFIGURATION, {})
        if plans_json else {}
    )
    target_spacing_zyx = configuration.get('spacing')
    if not target_spacing_zyx:
        return None

    target_spacing_xyz = np.array(
        [target_spacing_zyx[2], target_spacing_zyx[1], target_spacing_zyx[0]],
        dtype=float,
    )
    patch_size_zyx = configuration.get('patch_size')
    patch_size_xyz = (
        np.array(
            [patch_size_zyx[2], patch_size_zyx[1], patch_size_zyx[0]],
            dtype=float,
        )
        if patch_size_zyx else None
    )
    return {
        'target_spacing_xyz': target_spacing_xyz,
        'patch_size_xyz': patch_size_xyz,
        'num_labels': _get_dataset_num_labels(dataset_id),
    }


def _prediction_array_gib_for_geometry(size_xyz, spacing_xyz, prediction_spec):
    size_xyz = np.array(size_xyz, dtype=float)
    spacing_xyz = np.array(spacing_xyz, dtype=float)
    target_spacing_xyz = prediction_spec['target_spacing_xyz']
    target_size_xyz = np.ceil((size_xyz * spacing_xyz) / target_spacing_xyz)
    patch_size_xyz = prediction_spec['patch_size_xyz']
    if patch_size_xyz is not None:
        target_size_xyz = np.maximum(target_size_xyz, patch_size_xyz)
    target_voxels = int(np.prod(target_size_xyz))
    num_labels = prediction_spec['num_labels']
    return target_voxels * num_labels * np.dtype(np.float32).itemsize / (1024 ** 3)


def _estimate_prediction_array_gib(input_image_path, dataset_id):
    """Estimate one dense float32 class-probability array in nnU-Net target space."""
    prediction_spec = _get_prediction_array_spec(dataset_id)
    if prediction_spec is None:
        return None

    reader = sitk.ImageFileReader()
    reader.SetFileName(input_image_path)
    reader.ReadImageInformation()
    return _prediction_array_gib_for_geometry(
        reader.GetSize(),
        reader.GetSpacing(),
        prediction_spec,
    )


def _should_crop_cascade_to_binary(binary_input_path, model_config):
    if not model_config.get('crop_cascade_to_binary'):
        return False

    threshold_gib = _get_float_env(
        'TOPBRAIN_CASCADE_CROP_THRESHOLD_GIB',
        CASCADE_CROP_THRESHOLD_GIB,
    )
    estimate_gib = _estimate_prediction_array_gib(
        binary_input_path,
        model_config['cascade_dataset'],
    )
    if estimate_gib is None:
        print('  [Cascade crop] Could not estimate Dataset '
              f'{model_config["cascade_dataset"]} memory; using full volume.',
              flush=True)
        return False

    print(
        f'  [Cascade crop] Estimated Dataset {model_config["cascade_dataset"]} '
        f'37-class array: {estimate_gib:0.1f} GiB '
        f'(threshold {threshold_gib:0.1f} GiB).',
        flush=True,
    )
    return estimate_gib >= threshold_gib


def _binary_component_crop(binary_path, margin_mm=None, min_component_fraction=None):
    if margin_mm is None:
        margin_mm = _get_float_env('TOPBRAIN_CASCADE_CROP_MARGIN_MM', CASCADE_CROP_MARGIN_MM)
    if min_component_fraction is None:
        min_component_fraction = _get_float_env(
            'TOPBRAIN_CASCADE_CROP_MIN_COMPONENT_FRACTION',
            CASCADE_CROP_MIN_COMPONENT_FRACTION,
        )

    binary_img = sitk.ReadImage(binary_path)
    binary_mask = sitk.Cast(binary_img > 0, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.ConnectedComponent(binary_mask))
    component_labels = list(stats.GetLabels())
    if not component_labels:
        print('  [Cascade crop] Binary mask is empty; using full volume.', flush=True)
        return None

    component_sizes = {
        label: int(stats.GetNumberOfPixels(label))
        for label in component_labels
    }
    largest_component = max(component_sizes.values())
    min_component_size = max(1, int(round(largest_component * min_component_fraction)))
    selected_labels = [
        label
        for label, size in component_sizes.items()
        if size >= min_component_size
    ]

    full_size = np.array(binary_img.GetSize(), dtype=int)
    min_index = full_size.copy()
    max_index = np.zeros(3, dtype=int)
    selected_voxels = 0
    for label in selected_labels:
        x, y, z, sx, sy, sz = stats.GetBoundingBox(label)
        component_min = np.array([x, y, z], dtype=int)
        component_max = component_min + np.array([sx, sy, sz], dtype=int)
        min_index = np.minimum(min_index, component_min)
        max_index = np.maximum(max_index, component_max)
        selected_voxels += component_sizes[label]

    spacing = np.array(binary_img.GetSpacing(), dtype=float)
    margin_voxels = np.ceil(margin_mm / spacing).astype(int)
    min_index = np.maximum(0, min_index - margin_voxels)
    max_index = np.minimum(full_size, max_index + margin_voxels)
    crop_size = max_index - min_index
    if np.any(crop_size <= 0):
        print('  [Cascade crop] Invalid binary-mask crop; using full volume.', flush=True)
        return None

    crop_voxels = int(np.prod(crop_size))
    full_voxels = int(np.prod(full_size))
    if crop_voxels >= 0.98 * full_voxels:
        print('  [Cascade crop] Binary-mask crop is nearly full size; using full volume.', flush=True)
        return None

    total_foreground = sum(component_sizes.values())
    retained_percent = 100.0 * selected_voxels / max(1, total_foreground)
    print(
        '  [Cascade crop] Cropping Dataset 522 input to binary-mask region: '
        f'index={tuple(int(v) for v in min_index)}, '
        f'size={tuple(int(v) for v in crop_size)}, '
        f'{crop_voxels / full_voxels:0.1%} of full volume, '
        f'{retained_percent:0.1f}% of binary foreground retained.',
        flush=True,
    )
    return {
        'index': [int(v) for v in min_index],
        'size': [int(v) for v in crop_size],
        'full_size': [int(v) for v in full_size],
    }


def _write_cropped_image(input_path, output_path, crop):
    image = sitk.ReadImage(input_path)
    cropped = sitk.RegionOfInterest(image, crop['size'], crop['index'])
    sitk.WriteImage(cropped, output_path, True)


def _restore_cascade_crop(cropped_prediction_path, reference_full_path, restored_path, crop):
    reference_img = sitk.ReadImage(reference_full_path)
    cropped_prediction = sitk.ReadImage(cropped_prediction_path)
    restored = sitk.Image(reference_img.GetSize(), cropped_prediction.GetPixelID())
    restored.CopyInformation(reference_img)
    restored = sitk.Paste(
        restored,
        cropped_prediction,
        cropped_prediction.GetSize(),
        [0, 0, 0],
        crop['index'],
    )
    sitk.WriteImage(restored, restored_path, True)


def _axis_tile_plan(
    full_size,
    spacing,
    prediction_spec,
    axis,
    tile_count,
    context_mm,
):
    boundaries = np.rint(
        np.linspace(0, int(full_size[axis]), tile_count + 1)
    ).astype(int)
    context_voxels = int(np.ceil(context_mm / spacing[axis]))
    tiles = []
    for tile_index in range(tile_count):
        core_start = int(boundaries[tile_index])
        core_end = int(boundaries[tile_index + 1])
        if core_end <= core_start:
            return None

        tile_start = max(0, core_start - context_voxels)
        tile_end = min(int(full_size[axis]), core_end + context_voxels)
        tile_size = np.array(full_size, dtype=int)
        tile_size[axis] = tile_end - tile_start
        estimate_gib = _prediction_array_gib_for_geometry(
            tile_size,
            spacing,
            prediction_spec,
        )
        tiles.append(
            {
                'number': tile_index + 1,
                'axis': axis,
                'core_start': core_start,
                'core_end': core_end,
                'tile_start': tile_start,
                'tile_end': tile_end,
                'index': [tile_start if dim == axis else 0 for dim in range(3)],
                'size': [int(value) for value in tile_size],
                'estimate_gib': estimate_gib,
            }
        )
    return tiles


def _plan_cascade_tiles(
    cascade_input_path,
    dataset_id,
    target_gib=None,
    context_mm=None,
):
    if target_gib is None:
        target_gib = _get_float_env(
            'TOPBRAIN_CASCADE_TILE_TARGET_GIB',
            CASCADE_TILE_TARGET_GIB,
        )
    if context_mm is None:
        context_mm = _get_float_env(
            'TOPBRAIN_CASCADE_TILE_CONTEXT_MM',
            CASCADE_TILE_CONTEXT_MM,
        )
    if target_gib <= 0:
        raise ValueError('TOPBRAIN_CASCADE_TILE_TARGET_GIB must be greater than zero')
    if context_mm < 0:
        raise ValueError('TOPBRAIN_CASCADE_TILE_CONTEXT_MM cannot be negative')

    prediction_spec = _get_prediction_array_spec(dataset_id)
    if prediction_spec is None:
        print(
            f'  [Cascade tile] Could not read Dataset {dataset_id} plans; '
            'using one full-volume prediction.',
            flush=True,
        )
        return None

    reader = sitk.ImageFileReader()
    reader.SetFileName(cascade_input_path)
    reader.ReadImageInformation()
    full_size = np.array(reader.GetSize(), dtype=int)
    spacing = np.array(reader.GetSpacing(), dtype=float)
    full_estimate_gib = _prediction_array_gib_for_geometry(
        full_size,
        spacing,
        prediction_spec,
    )
    print(
        f'  [Cascade tile] Estimated cropped Dataset {dataset_id} '
        f'{prediction_spec["num_labels"]}-class array: {full_estimate_gib:0.1f} GiB '
        f'(tile target {target_gib:0.1f} GiB).',
        flush=True,
    )
    if full_estimate_gib <= target_gib:
        return None

    candidates = []
    for axis in range(3):
        max_tiles = min(CASCADE_TILE_MAX_TILES, int(full_size[axis]))
        for tile_count in range(2, max_tiles + 1):
            tiles = _axis_tile_plan(
                full_size,
                spacing,
                prediction_spec,
                axis,
                tile_count,
                context_mm,
            )
            if tiles is None:
                continue
            max_estimate_gib = max(tile['estimate_gib'] for tile in tiles)
            if max_estimate_gib <= target_gib:
                total_estimate_gib = sum(tile['estimate_gib'] for tile in tiles)
                candidates.append(
                    (tile_count, total_estimate_gib, max_estimate_gib, axis, tiles)
                )
                break

    if not candidates:
        raise RuntimeError(
            f'Unable to split Dataset {dataset_id} input into tiles at or below '
            f'{target_gib:0.1f} GiB. Increase TOPBRAIN_CASCADE_TILE_TARGET_GIB '
            'or reduce TOPBRAIN_CASCADE_TILE_CONTEXT_MM.'
        )

    _, _, max_estimate_gib, axis, tiles = min(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1], candidate[2]),
    )
    print(
        f'  [Cascade tile] Splitting axis {"xyz"[axis]} into {len(tiles)} tiles '
        f'with {context_mm:g} mm context per internal edge; '
        f'maximum estimate {max_estimate_gib:0.1f} GiB.',
        flush=True,
    )
    return {
        'axis': axis,
        'full_size': [int(value) for value in full_size],
        'tiles': tiles,
    }


def _stitch_cascade_tile(stitched, tile_prediction, tile):
    axis = tile['axis']
    expected_size = tuple(tile['size'])
    if tile_prediction.GetSize() != expected_size:
        raise RuntimeError(
            'Dataset 522 tile output geometry does not match its input: '
            f'expected size {expected_size}, got {tile_prediction.GetSize()}'
        )

    source_index = [0, 0, 0]
    source_index[axis] = tile['core_start'] - tile['tile_start']
    source_size = list(tile_prediction.GetSize())
    source_size[axis] = tile['core_end'] - tile['core_start']
    destination_index = [0, 0, 0]
    destination_index[axis] = tile['core_start']
    return sitk.Paste(
        stitched,
        tile_prediction,
        source_size,
        source_index,
        destination_index,
    )


def _run_tiled_cascade(
    cascade_input_image,
    cascade_input_binary,
    cascade_output_path,
    temp_dir,
    dataset_id,
    disable_tta,
    tile_plan,
):
    reader = sitk.ImageFileReader()
    reader.SetFileName(cascade_input_image)
    reader.ReadImageInformation()
    stitched = None
    tiles_root = os.path.join(temp_dir, 'cascade_tiles')
    os.makedirs(tiles_root)

    for tile in tile_plan['tiles']:
        tile_number = tile['number']
        tile_root = os.path.join(tiles_root, f'tile_{tile_number:03d}')
        tile_input_dir = os.path.join(tile_root, 'input')
        tile_output_dir = os.path.join(tile_root, 'output')
        os.makedirs(tile_input_dir)
        os.makedirs(tile_output_dir)
        tile_input_image = os.path.join(tile_input_dir, 'case_0000.nii.gz')
        tile_input_binary = os.path.join(tile_input_dir, 'case_0001.nii.gz')
        print(
            f'  [Cascade tile] Tile {tile_number}/{len(tile_plan["tiles"])}: '
            f'index={tuple(tile["index"])}, size={tuple(tile["size"])}, '
            f'estimate={tile["estimate_gib"]:0.1f} GiB.',
            flush=True,
        )
        _write_cropped_image(cascade_input_image, tile_input_image, tile)
        _write_cropped_image(cascade_input_binary, tile_input_binary, tile)
        _run_nnunet_dir(
            tile_input_dir,
            tile_output_dir,
            dataset_id,
            disable_tta=disable_tta,
        )

        tile_result_path = os.path.join(tile_output_dir, 'case.nii.gz')
        if not os.path.exists(tile_result_path):
            raise FileNotFoundError(
                f'Dataset {dataset_id} did not create tile output: {tile_result_path}'
            )
        tile_prediction = sitk.ReadImage(tile_result_path)
        if stitched is None:
            stitched = sitk.Image(tuple(tile_plan['full_size']), tile_prediction.GetPixelID())
            stitched.SetSpacing(reader.GetSpacing())
            stitched.SetOrigin(reader.GetOrigin())
            stitched.SetDirection(reader.GetDirection())
        stitched = _stitch_cascade_tile(stitched, tile_prediction, tile)
        del tile_prediction
        shutil.rmtree(tile_root, ignore_errors=True)

    if stitched is None:
        raise RuntimeError('Cascade tile plan contained no tiles')
    sitk.WriteImage(stitched, cascade_output_path, True)


def get_orientation_and_closest_plane(img: sitk.Image):
    """
    Determines (1) a best-effort orientation string (e.g., 'LPS', 'RAS')
    and (2) the closest plane (axial/coronal/sagittal) even if oblique,
    based on the physical direction of the slice axis (k axis).

    Returns dict with:
      - orientation: str (e.g. 'LPS')
      - closest_plane: 'axial'|'coronal'|'sagittal'
      - tilt_degrees: float (0 means perfectly in that plane)
      - axis_alignment: dict with alignment to LR/AP/SI in [0..1]
    """
    f = sitk.DICOMOrientImageFilter()
    orientation = f.GetOrientationFromDirectionCosines(img.GetDirection())
    D = np.array(img.GetDirection(), dtype=float).reshape(3, 3)

    k_dir = D[:, 2]
    k_dir = k_dir / (np.linalg.norm(k_dir) + 1e-12)

    lr = abs(k_dir[0])  
    ap = abs(k_dir[1])  
    si = abs(k_dir[2])  

    axis_alignment = {"LR": float(lr), "AP": float(ap), "SI": float(si)}

    m = max(lr, ap, si)
    if m == si:
        closest_plane = "axial"
        cos_theta = si
    elif m == ap:
        closest_plane = "coronal"
        cos_theta = ap
    else:
        closest_plane = "sagittal"
        cos_theta = lr

    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    tilt_degrees = float(np.degrees(np.arccos(cos_theta)))

    return {
        "orientation": orientation,
        "closest_plane": closest_plane,
        "tilt_degrees": tilt_degrees,
        "axis_alignment": axis_alignment,
    }

def reorient_sitk_image(img: sitk.Image, target_orientation: str):
    """
    Reorient a SimpleITK image to a specified DICOM-style orientation code.
    """
    target = target_orientation.strip().upper()

    # Validation: Check for one char from each pair (LR, AP, SI).
    # The order can be permuted (e.g., 'RIA' is valid), as sitk.DICOMOrient handles it.
    has_lr = sum(c in target for c in "LR") == 1
    has_ap = sum(c in target for c in "AP") == 1
    has_si = sum(c in target for c in "SI") == 1
    
    if not (len(target) == 3 and has_lr and has_ap and has_si):
        raise ValueError(f"Invalid target_orientation='{target_orientation}'. Must be a 3-character string with one char from each of [LR], [AP], [SI].")

    f = sitk.DICOMOrientImageFilter()
    current = f.GetOrientationFromDirectionCosines(img.GetDirection())

    if current == target:
        return img

    out = sitk.DICOMOrient(img, target)
    return out

# --- HELPER: CROP TO HEAD (CTA ONLY) ---
def _crop_to_head(temp_input_path):
    print(f"  [Preprocessing] Analyzing {os.path.basename(temp_input_path)} for head cropping...")
    img = nib.load(temp_input_path)
    data = img.get_fdata()
    affine = img.affine
    orig_shape = data.shape

    # 1. Heuristic Z-Crop
    mask = data > -200 
    z_presence = np.any(mask, axis=(0, 1))
    z_indices = np.where(z_presence)[0]

    if len(z_indices) == 0:
        print("  [Warning] Image seems empty. Skipping crop.")
        return None

    z_top = z_indices[-1]
    z_bottom_actual = z_indices[0]
    z_spacing = abs(affine[2, 2])
    slices_to_keep = int(200 / z_spacing)
    z_crop_start = max(z_bottom_actual, z_top - slices_to_keep)
    
    # 2. Prep for ANTs
    heuristic_data = data[:, :, z_crop_start:z_top].copy()
    heuristic_data[heuristic_data < -100] = 0
    
    new_affine = affine.copy()
    z_shift = np.array([0, 0, z_crop_start])
    new_origin = nib.affines.apply_affine(affine, z_shift)
    new_affine[:3, 3] = new_origin
    heuristic_img = nib.Nifti1Image(heuristic_data, new_affine, img.header)

    nib.save(heuristic_img, temp_input_path)

    # 3. ANTs Registration
    print("  [Preprocessing] Registering MNI template TO Native CTA...")
    try:
        ants_template = ants.image_read(TEMPLATE_PATH)
    except Exception as e:
        print(f"  [Warning] Could not load template. Falling back to heuristic crop.")
        return (0, data.shape[0], 0, data.shape[1], z_crop_start, z_top, orig_shape, affine, img.header)

    ants_native = ants.image_read(temp_input_path)
    
    tx = ants.registration(
        fixed=ants_native,
        moving=ants_template,
        type_of_transform="Translation",
        reg_iterations=(100, 50, 0),
        metric="MI"
    )
        
    # 4. Define Bounding Box
    template_mask = ants_template * 0 + 1
    native_mask = ants.apply_transforms(
        fixed=ants_native, 
        moving=template_mask, 
        transformlist=tx['fwdtransforms'], 
        interpolator='nearestNeighbor'
    )
    
    native_mask_np = native_mask.numpy() > 0.5
    coords = np.array(np.nonzero(native_mask_np))
    
    if coords.size == 0:
        print("  [Warning] Registration resulted in empty mask. Falling back to heuristic crop.")
        nib.save(heuristic_img, temp_input_path)
        return (0, data.shape[0], 0, data.shape[1], z_crop_start, z_top, orig_shape, affine, img.header)
        
    x_min, y_min, z_min = np.min(coords, axis=1)
    x_max, y_max, z_max = np.max(coords, axis=1) + 1
    
    # 5. Crop Original Data
    abs_x_min, abs_x_max = x_min, x_max
    abs_y_min, abs_y_max = y_min, y_max
    abs_z_min, abs_z_max = z_crop_start + z_min, z_crop_start + z_max
    
    print(f"  [Crop] Native Bounds Mapped to Template - X: {abs_x_min}:{abs_x_max}, Y: {abs_y_min}:{abs_y_max}, Z: {abs_z_min}:{abs_z_max}")
    
    final_cropped_data = data[abs_x_min:abs_x_max, abs_y_min:abs_y_max, abs_z_min:abs_z_max]
    
    final_affine = affine.copy()
    shift = np.array([abs_x_min, abs_y_min, abs_z_min])
    final_origin = nib.affines.apply_affine(affine, shift)
    final_affine[:3, 3] = final_origin
    
    final_img = nib.Nifti1Image(final_cropped_data, final_affine, img.header)
    nib.save(final_img, temp_input_path)
        
    return (abs_x_min, abs_x_max, abs_y_min, abs_y_max, abs_z_min, abs_z_max, orig_shape, affine, img.header)


# --- HELPER: RESTORE CROP (PADDING) ---
def _restore_crop_space(prediction_path, final_output_path, crop_info):
    x_start, x_end, y_start, y_end, z_start, z_end, orig_shape, orig_affine, orig_header = crop_info
    
    if not os.path.exists(prediction_path):
        print("  [Error] No prediction file found to restore.")
        return

    pred_img = nib.load(prediction_path)
    pred_data = pred_img.get_fdata()
    
    restored_data = np.zeros(orig_shape, dtype=pred_data.dtype)
    dx, dy, dz = pred_data.shape
    
    x_slice = slice(x_start, x_start + min(dx, x_end - x_start))
    y_slice = slice(y_start, y_start + min(dy, y_end - y_start))
    z_slice = slice(z_start, z_start + min(dz, z_end - z_start))
    
    pred_x_slice = slice(0, min(dx, x_end - x_start))
    pred_y_slice = slice(0, min(dy, y_end - y_start))
    pred_z_slice = slice(0, min(dz, z_end - z_start))
    
    restored_data[x_slice, y_slice, z_slice] = pred_data[pred_x_slice, pred_y_slice, pred_z_slice]
    
    print("  [Postprocessing] Restoring 3D crop back to full uncropped space...")
    restored_img = nib.Nifti1Image(restored_data, orig_affine, orig_header)
    nib.save(restored_img, final_output_path)


def _apply_probability_threshold(
    probability_path,
    reference_segmentation_path,
    thresholded_segmentation_path,
    foreground_background_ratio,
    sum_foreground_probabilities,
):
    """Create a label map using either strongest-class or unified foreground probability."""
    if not np.isfinite(foreground_background_ratio) or foreground_background_ratio < 0:
        raise ValueError("foreground_background_ratio must be a finite number >= 0")

    if not os.path.exists(probability_path):
        raise FileNotFoundError(f"nnU-Net probability file not found: {probability_path}")

    with np.load(probability_path) as probability_archive:
        if "probabilities" not in probability_archive:
            raise KeyError(f"No 'probabilities' array found in {probability_path}")
        probabilities = probability_archive["probabilities"]

    if probabilities.ndim != 4 or probabilities.shape[0] < 2:
        raise ValueError(
            "Expected nnU-Net probabilities with shape "
            f"(classes, z, y, x); got {probabilities.shape}"
        )

    reference_image = sitk.ReadImage(reference_segmentation_path)
    reference_array_shape = tuple(reversed(reference_image.GetSize()))
    if probabilities.shape[1:] != reference_array_shape:
        raise ValueError(
            "Probability-map shape does not match the nnU-Net segmentation: "
            f"{probabilities.shape[1:]} versus {reference_array_shape}"
        )

    background_probability = probabilities[0]
    foreground_probabilities = probabilities[1:]
    if sum_foreground_probabilities:
        foreground_decision_probability = np.sum(foreground_probabilities, axis=0)
    else:
        foreground_decision_probability = np.max(foreground_probabilities, axis=0)

    # TopBrain labels are consecutive (background=0, artery labels=1..N), so
    # the foreground probability channel index maps directly to its label value.
    label_dtype = np.uint8 if probabilities.shape[0] <= 256 else np.uint16
    strongest_foreground_label = (
        np.argmax(foreground_probabilities, axis=0).astype(label_dtype) + 1
    )
    foreground_wins = foreground_decision_probability > (
        foreground_background_ratio * background_probability
    )
    thresholded_segmentation = np.where(
        foreground_wins, strongest_foreground_label, 0
    ).astype(label_dtype)

    thresholded_image = sitk.GetImageFromArray(thresholded_segmentation)
    thresholded_image.CopyInformation(reference_image)
    sitk.WriteImage(thresholded_image, thresholded_segmentation_path)


def _apply_foreground_background_threshold(
    probability_path,
    reference_segmentation_path,
    thresholded_segmentation_path,
    foreground_background_ratio,
):
    """Compare the strongest individual foreground class with background.

    nnU-Net normally assigns the class with the largest probability. For these
    multiclass TopBrain models, that means foreground wins when its strongest
    artery probability is greater than the background probability. This helper
    replaces that decision with:

        strongest_foreground > foreground_background_ratio * background

    Knob direction:
      - 1.0: nnU-Net's normal argmax decision
      - below 1.0 (for example 0.75): more generous / more foreground
      - above 1.0 (for example 1.25): more conservative / less foreground
      - 0.0: maximally generous within the area evaluated by nnU-Net

    The winning foreground artery class is unchanged; only the decision between
    that foreground class and background is adjusted.
    """
    _apply_probability_threshold(
        probability_path,
        reference_segmentation_path,
        thresholded_segmentation_path,
        foreground_background_ratio,
        sum_foreground_probabilities=False,
    )


def _apply_unified_probability_threshold(
    probability_path,
    reference_segmentation_path,
    thresholded_segmentation_path,
    foreground_background_ratio,
):
    """Compare total arterial probability with background, then assign an artery.

    The decision rule is:

        sum(all foreground artery probabilities) > ratio * background

    A voxel that passes is assigned the highest-probability individual artery
    class. Lower ratios are more generous; higher ratios are more conservative.
    Unlike the strongest-class method, this can retain a voxel whose arterial
    probability is divided among several neighboring artery classes.
    """
    _apply_probability_threshold(
        probability_path,
        reference_segmentation_path,
        thresholded_segmentation_path,
        foreground_background_ratio,
        sum_foreground_probabilities=True,
    )


def _run_nnunet_dir(input_dir, output_dir, dataset_id, disable_tta=True):
    cmd = [
        _find_nnunet_predict_executable(),
        "-i",
        input_dir,
        "-o",
        output_dir,
        "-d",
        str(dataset_id),
        "-c",
        DEFAULT_CONFIGURATION,
        "-p",
        DEFAULT_PLANS,
        "-tr",
        DEFAULT_TRAINER,
        "-f",
        DEFAULT_FOLD,
        "-chk",
        DEFAULT_CHECKPOINT,
        "-nps",
        "1",
        "-npp",
        "1",
        "-device",
        _get_nnunet_device(),
    ]
    if disable_tta:
        cmd.append("--disable_tta")

    print(f"  [nnU-Net] Running Dataset {dataset_id}...", flush=True)
    subprocess.run(cmd, check=True, env=_get_fixed_env())


def _copy_and_preprocess_input(input_path, temp_input_path, crop_head):
    shutil.copy2(input_path, temp_input_path)
    temp_input_image = sitk.ReadImage(temp_input_path)
    temp_input_image = reorient_sitk_image(temp_input_image, MODEL_INPUT_ORIENTATION)
    sitk.WriteImage(temp_input_image, temp_input_path)

    crop_info = None
    if crop_head:
        crop_info = _crop_to_head(temp_input_path)
    return crop_info


def _finalize_prediction(temp_prediction_path, output_path, crop_info, original_orientation):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if crop_info:
        _restore_crop_space(temp_prediction_path, output_path, crop_info)
    else:
        shutil.copy2(temp_prediction_path, output_path)

    output_image = sitk.ReadImage(output_path)
    output_geometry = get_orientation_and_closest_plane(output_image)
    if output_geometry['orientation'] != original_orientation:
        output_image = reorient_sitk_image(output_image, original_orientation)
        sitk.WriteImage(output_image, output_path)


def _predict_single_stage(input_path, output_path, model_config, disable_tta=True):
    original_orientation = get_orientation_and_closest_plane(sitk.ReadImage(input_path))['orientation']
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix='temp_topbrain_nnunet_', dir=output_dir)
    try:
        temp_in = os.path.join(temp_dir, 'input')
        temp_out = os.path.join(temp_dir, 'output')
        os.makedirs(temp_in)
        os.makedirs(temp_out)
        temp_input_path = os.path.join(temp_in, 'case_0000.nii.gz')
        crop_info = _copy_and_preprocess_input(input_path, temp_input_path, model_config['crop_head'])
        _run_nnunet_dir(temp_in, temp_out, model_config['dataset'], disable_tta=disable_tta)

        temp_result = os.path.join(temp_out, 'case.nii.gz')
        if not os.path.exists(temp_result):
            raise FileNotFoundError(f'nnU-Net did not create expected output: {temp_result}')
        _finalize_prediction(temp_result, output_path, crop_info, original_orientation)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _predict_cascade(input_path, output_path, model_config, binary_output_path=None, disable_tta=True):
    original_orientation = get_orientation_and_closest_plane(sitk.ReadImage(input_path))['orientation']
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix='temp_topbrain_cascade_', dir=output_dir)
    try:
        binary_in = os.path.join(temp_dir, 'binary_input')
        binary_out = os.path.join(temp_dir, 'binary_output')
        cascade_in = os.path.join(temp_dir, 'cascade_input')
        cascade_out = os.path.join(temp_dir, 'cascade_output')
        for path in (binary_in, binary_out, cascade_in, cascade_out):
            os.makedirs(path)

        binary_input_path = os.path.join(binary_in, 'case_0000.nii.gz')
        crop_info = _copy_and_preprocess_input(input_path, binary_input_path, model_config['crop_head'])
        _run_nnunet_dir(binary_in, binary_out, model_config['binary_dataset'], disable_tta=disable_tta)

        binary_result = os.path.join(binary_out, 'case.nii.gz')
        if not os.path.exists(binary_result):
            raise FileNotFoundError(f'Binary nnU-Net did not create expected output: {binary_result}')

        cascade_input_image = os.path.join(cascade_in, 'case_0000.nii.gz')
        cascade_input_binary = os.path.join(cascade_in, 'case_0001.nii.gz')
        cascade_crop = None
        if _should_crop_cascade_to_binary(binary_input_path, model_config):
            cascade_crop = _binary_component_crop(binary_result)

        if cascade_crop:
            _write_cropped_image(binary_input_path, cascade_input_image, cascade_crop)
            _write_cropped_image(binary_result, cascade_input_binary, cascade_crop)
        else:
            shutil.copy2(binary_input_path, cascade_input_image)
            shutil.copy2(binary_result, cascade_input_binary)

        cascade_result = os.path.join(cascade_out, 'case.nii.gz')
        tile_plan = _plan_cascade_tiles(
            cascade_input_image,
            model_config['cascade_dataset'],
        )
        if tile_plan:
            _run_tiled_cascade(
                cascade_input_image,
                cascade_input_binary,
                cascade_result,
                temp_dir,
                model_config['cascade_dataset'],
                disable_tta,
                tile_plan,
            )
        else:
            _run_nnunet_dir(
                cascade_in,
                cascade_out,
                model_config['cascade_dataset'],
                disable_tta=disable_tta,
            )
            if not os.path.exists(cascade_result):
                raise FileNotFoundError(
                    f'Cascade nnU-Net did not create expected output: {cascade_result}'
                )

        if cascade_crop:
            restored_cascade_dir = os.path.join(temp_dir, 'cascade_restored')
            os.makedirs(restored_cascade_dir)
            restored_cascade_result = os.path.join(restored_cascade_dir, 'case.nii.gz')
            _restore_cascade_crop(
                cascade_result,
                binary_input_path,
                restored_cascade_result,
                cascade_crop,
            )
            cascade_result = restored_cascade_result

        _finalize_prediction(cascade_result, output_path, crop_info, original_orientation)
        if binary_output_path:
            _finalize_prediction(binary_result, binary_output_path, crop_info, original_orientation)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# --- CORE RUNNER ---
def _run_nnunet(
    input_file,
    output_file,
    dataset_id,
    crop_head=False,
    foreground_background_ratio=None,
    use_unified_arterial_probability=False,
    disable_tta=True,
):
    if use_unified_arterial_probability and foreground_background_ratio is None:
        raise ValueError(
            "use_unified_arterial_probability requires foreground_background_ratio"
        )
    if foreground_background_ratio is not None:
        if not np.isfinite(foreground_background_ratio) or foreground_background_ratio < 0:
            raise ValueError("foreground_background_ratio must be a finite number >= 0")

    nnunet_predict_executable = _find_nnunet_predict_executable()
    nnunet_device = _get_nnunet_device()

    orig_geometries = get_orientation_and_closest_plane(sitk.ReadImage(input_file))
    
    base_dir = os.path.dirname(os.path.abspath(output_file))
    temp_dir = os.path.join(base_dir, 'temp_nnunet_inference')
    
    temp_in = os.path.join(temp_dir, 'input')
    temp_out = os.path.join(temp_dir, 'output')
    
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_in)
    os.makedirs(temp_out)

    temp_input_filename = 'case_0000.nii.gz'
    temp_input_path = os.path.join(temp_in, temp_input_filename)
    shutil.copy2(input_file, temp_input_path)
    temp_input_image = sitk.ReadImage(temp_input_path)
    temp_input_image = reorient_sitk_image(temp_input_image, 'LPS')
    sitk.WriteImage(temp_input_image, temp_input_path)

    crop_info = None
    if crop_head:
        crop_info = _crop_to_head(temp_input_path)
    
    cmd = [
        nnunet_predict_executable,
        "-i", temp_in, 
        "-o", temp_out, 
        "-d", str(dataset_id),
        "-c", "3d_fullres", 
        "-p", "nnUNetResEncUNetMPlans", 
        "-tr", "nnUNetTrainerNoMirroring", 
        "-f", "0", 
        "-chk", "checkpoint_final.pth",
        "-nps", "1", 
        "-npp", "1",
        "-device", nnunet_device,
    ]

    if foreground_background_ratio is not None:
        cmd.append("--save_probabilities")
    if disable_tta:
        cmd.append("--disable_tta")
    
    print(f"  [nnU-Net] Running Dataset {dataset_id}...")
    
    try:
        subprocess.run(cmd, check=True, env=_get_fixed_env())
        temp_result = os.path.join(temp_out, 'case.nii.gz')

        if foreground_background_ratio is not None:
            probability_path = os.path.join(temp_out, 'case.npz')
            threshold_method = (
                _apply_unified_probability_threshold
                if use_unified_arterial_probability
                else _apply_foreground_background_threshold
            )
            probability_description = (
                "summed arterial probability"
                if use_unified_arterial_probability
                else "strongest arterial-class probability"
            )
            print(
                f"  [Postprocessing] Comparing {probability_description} with "
                f"background using ratio {foreground_background_ratio:g}..."
            )
            threshold_method(
                probability_path,
                temp_result,
                temp_result,
                foreground_background_ratio,
            )
        
        if crop_info:
            _restore_crop_space(temp_result, output_file, crop_info)
        else:
            if os.path.exists(temp_result):
                shutil.copy2(temp_result, output_file)

        # Restore original orientation
        output_result_image = sitk.ReadImage(output_file)
        final_geometries = get_orientation_and_closest_plane(output_result_image)
        if final_geometries['orientation'] != orig_geometries['orientation']:
            output_result_image = reorient_sitk_image(output_result_image, orig_geometries['orientation'])
            sitk.WriteImage(output_result_image, output_file)

        print(f"  [Success] Saved to {output_file}\n")

    except subprocess.CalledProcessError as e:
        print(f"  [Error] nnU-Net failed with error: {e}")
        raise
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

# --- EXPORTED FUNCTIONS ---
def predict(input_path, output_path, modality, model='base', binary_output_path=None, disable_tta=True):
    """Predict TopBrain artery labels for one NIfTI file.

    ``model='base'`` uses Dataset501/502, ``model='binary'`` uses Dataset511/512,
    and ``model='cascade'`` runs Dataset511/512 first, then Dataset521/522 with
    image + binary mask channels.
    """
    configure_environment()
    modality = modality.strip().lower()
    model = model.strip().lower()
    if modality not in {'cta', 'mra'}:
        raise ValueError("modality must be 'cta' or 'mra'")
    if model not in MODEL_REGISTRY:
        raise ValueError(f"model must be one of: {', '.join(sorted(MODEL_REGISTRY))}")

    input_path = os.path.abspath(os.path.expanduser(input_path))
    output_path = os.path.abspath(os.path.expanduser(output_path))
    if binary_output_path:
        binary_output_path = os.path.abspath(os.path.expanduser(binary_output_path))

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f'Missing input NIfTI: {input_path}')

    print(
        f'Running TopBrain {model} {modality.upper()} prediction on '
        f'{os.path.basename(input_path)}...',
        flush=True,
    )
    if model == 'cascade':
        _predict_cascade(
            input_path,
            output_path,
            MODEL_REGISTRY[model][modality],
            binary_output_path=binary_output_path,
            disable_tta=disable_tta,
        )
    else:
        _predict_single_stage(
            input_path,
            output_path,
            MODEL_REGISTRY[model][modality],
            disable_tta=disable_tta,
        )
    print(f'  [Success] Saved to {output_path}')


def predict_cta(input_path, output_path, model='base', binary_output_path=None, disable_tta=True):
    predict(
        input_path,
        output_path,
        modality='cta',
        model=model,
        binary_output_path=binary_output_path,
        disable_tta=disable_tta,
    )


def predict_mra(input_path, output_path, model='base', binary_output_path=None, disable_tta=True):
    predict(
        input_path,
        output_path,
        modality='mra',
        model=model,
        binary_output_path=binary_output_path,
        disable_tta=disable_tta,
    )


def predict_cta_thresholded(input_path, output_path, foreground_background_ratio=1.0):
    """Predict CTA labels with an adjustable foreground/background decision.

    Use 1.0 for normal nnU-Net behavior. Lower the ratio for a more generous
    segmentation (try 0.75), or raise it for a more conservative segmentation
    (try 1.25). The ratio must be zero or greater.
    """
    print(
        f"Running thresholded CTA prediction on {os.path.basename(input_path)} "
        f"with foreground/background ratio {foreground_background_ratio:g}..."
    )
    _run_nnunet(
        input_path,
        output_path,
        "Dataset501_TopBrainCTA2026",
        crop_head=True,
        foreground_background_ratio=foreground_background_ratio,
        disable_tta=True,
    )


def predict_mra_thresholded(input_path, output_path, foreground_background_ratio=1.0):
    """Predict MRA labels with an adjustable foreground/background decision.

    Use 1.0 for normal nnU-Net behavior. Lower the ratio for a more generous
    segmentation (try 0.75), or raise it for a more conservative segmentation
    (try 1.25). The ratio must be zero or greater.
    """
    print(
        f"Running thresholded MRA prediction on {os.path.basename(input_path)} "
        f"with foreground/background ratio {foreground_background_ratio:g}..."
    )
    _run_nnunet(
        input_path,
        output_path,
        "502",
        crop_head=False,
        foreground_background_ratio=foreground_background_ratio,
        disable_tta=True,
    )


def predict_cta_uniprob(input_path, output_path, foreground_background_ratio=1.0):
    """Predict CTA labels using total arterial probability versus background.

    For each voxel, probabilities for every non-background artery class are
    summed. If that total exceeds ``foreground_background_ratio`` times the
    background probability, the voxel receives its highest-probability artery
    label. Lower ratios are more generous; higher ratios are more conservative.
    """
    print(
        f"Running unified-probability CTA prediction on {os.path.basename(input_path)} "
        f"with arterial/background ratio {foreground_background_ratio:g}..."
    )
    _run_nnunet(
        input_path,
        output_path,
        "Dataset501_TopBrainCTA2026",
        crop_head=True,
        foreground_background_ratio=foreground_background_ratio,
        use_unified_arterial_probability=True,
        disable_tta=True,
    )


def predict_mra_uniprob(input_path, output_path, foreground_background_ratio=1.0):
    """Predict MRA labels using total arterial probability versus background.

    For each voxel, probabilities for every non-background artery class are
    summed. If that total exceeds ``foreground_background_ratio`` times the
    background probability, the voxel receives its highest-probability artery
    label. Lower ratios are more generous; higher ratios are more conservative.
    """
    print(
        f"Running unified-probability MRA prediction on {os.path.basename(input_path)} "
        f"with arterial/background ratio {foreground_background_ratio:g}..."
    )
    _run_nnunet(
        input_path,
        output_path,
        "502",
        crop_head=False,
        foreground_background_ratio=foreground_background_ratio,
        use_unified_arterial_probability=True,
        disable_tta=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(description='Predict TopBrain artery labels for one CTA/MRA NIfTI.')
    parser.add_argument('-i', '--input', required=True, help='Input CTA/MRA NIfTI file.')
    parser.add_argument('-o', '--output', required=True, help='Output label NIfTI file.')
    parser.add_argument('--modality', choices=('cta', 'mra'), required=True)
    parser.add_argument('--model', choices=tuple(sorted(MODEL_REGISTRY)), default='base')
    parser.add_argument(
        '--binary-output',
        default=None,
        help='Optional restored binary output path for cascade predictions.',
    )
    parser.add_argument('--tta', action='store_true', help='Allow nnU-Net TTA. Default disables TTA/mirroring.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    predict(
        args.input,
        args.output,
        modality=args.modality,
        model=args.model,
        binary_output_path=args.binary_output,
        disable_tta=not args.tta,
    )


if __name__ == '__main__':
    main()
