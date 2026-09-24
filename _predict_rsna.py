import argparse
import os
import shutil
import subprocess
import sys
import tempfile


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
DUMMY_RAW_DIR = os.path.join(CURRENT_DIR, 'dummy_raw')
DUMMY_PREPROCESSED_DIR = os.path.join(CURRENT_DIR, 'dummy_preprocessed')
TEMPLATE_PATH = os.path.join(CURRENT_DIR, 'MNI', '1mm', 'ct_rorden_lps.nii')

DEFAULT_CONFIGURATION = '3d_fullres'
DEFAULT_PLANS = 'nnUNetResEncUNetMPlans'
DEFAULT_TRAINER = 'nnUNetTrainerNoMirroring'
DEFAULT_FOLD = '0'
DEFAULT_CHECKPOINT = 'checkpoint_final.pth'
MODEL_INPUT_ORIENTATION = 'RAS'

MODEL_REGISTRY = {
    'base': {
        'output_kind': 'multiclass',
        'cta': {'dataset': '601', 'name': 'RSNACTA', 'crop_head': True},
        'mra': {'dataset': '602', 'name': 'RSNAMRA', 'crop_head': False},
    },
    'binary': {
        'output_kind': 'binary',
        'cta': {'dataset': '611', 'name': 'RSNACTABinary', 'crop_head': True},
        'mra': {'dataset': '612', 'name': 'RSNAMRABinary', 'crop_head': False},
    },
    'cascade': {
        'output_kind': 'multiclass',
        'cta': {
            'binary_dataset': '611',
            'cascade_dataset': '621',
            'name': 'RSNACTABinaryToMulticlass',
            'crop_head': True,
        },
        'mra': {
            'binary_dataset': '612',
            'cascade_dataset': '622',
            'name': 'RSNAMRABinaryToMulticlass',
            'crop_head': False,
        },
    },
}


def _require_numpy():
    import numpy as np
    return np


def _require_nibabel():
    import nibabel as nib
    return nib


def _require_sitk():
    import SimpleITK as sitk
    return sitk


def _require_ants():
    import ants
    return ants


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
    executable = shutil.which('nnUNetv2_predict')
    if executable:
        return executable

    python_bin_dir = os.path.dirname(os.path.abspath(sys.executable))
    for executable_name in ('nnUNetv2_predict', 'nnUNetv2_predict.exe'):
        candidate = os.path.join(python_bin_dir, executable_name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        'nnUNetv2_predict is not installed for the active Python interpreter '
        f'({sys.executable}).'
    )


def _get_nnunet_device():
    device = os.environ.get('RSNA_NNUNET_DEVICE', '').strip().lower()
    if not device:
        device = 'mps' if sys.platform == 'darwin' else 'cuda'
    if device not in {'cpu', 'cuda', 'mps'}:
        raise ValueError('RSNA_NNUNET_DEVICE must be one of: cpu, cuda, or mps')
    return device


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
        print(f'  [Warning] Could not find Conda libstdc++ at {lib_path}')

    if _get_nnunet_device() == 'mps':
        env.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

    return env


def get_orientation_and_closest_plane(img):
    np = _require_numpy()
    sitk = _require_sitk()
    orienter = sitk.DICOMOrientImageFilter()
    orientation = orienter.GetOrientationFromDirectionCosines(img.GetDirection())
    direction = np.array(img.GetDirection(), dtype=float).reshape(3, 3)
    k_dir = direction[:, 2]
    k_dir = k_dir / (np.linalg.norm(k_dir) + 1e-12)

    lr = abs(k_dir[0])
    ap = abs(k_dir[1])
    si = abs(k_dir[2])
    axis_alignment = {'LR': float(lr), 'AP': float(ap), 'SI': float(si)}

    if max(lr, ap, si) == si:
        closest_plane = 'axial'
        cos_theta = si
    elif max(lr, ap, si) == ap:
        closest_plane = 'coronal'
        cos_theta = ap
    else:
        closest_plane = 'sagittal'
        cos_theta = lr

    tilt_degrees = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    return {
        'orientation': orientation,
        'closest_plane': closest_plane,
        'tilt_degrees': tilt_degrees,
        'axis_alignment': axis_alignment,
    }


def reorient_sitk_image(img, target_orientation):
    sitk = _require_sitk()
    target = target_orientation.strip().upper()
    has_lr = sum(c in target for c in 'LR') == 1
    has_ap = sum(c in target for c in 'AP') == 1
    has_si = sum(c in target for c in 'SI') == 1
    if not (len(target) == 3 and has_lr and has_ap and has_si):
        raise ValueError(
            f"Invalid target_orientation='{target_orientation}'. "
            'Must contain one char from each of [LR], [AP], [SI].'
        )

    orienter = sitk.DICOMOrientImageFilter()
    current = orienter.GetOrientationFromDirectionCosines(img.GetDirection())
    if current == target:
        return img
    return sitk.DICOMOrient(img, target)


def _crop_to_head(temp_input_path):
    np = _require_numpy()
    nib = _require_nibabel()

    print(f'  [Preprocessing] Cropping CTA head region for {os.path.basename(temp_input_path)}...')
    img = nib.load(temp_input_path)
    data = img.get_fdata()
    affine = img.affine
    orig_shape = data.shape

    mask = data > -200
    z_presence = np.any(mask, axis=(0, 1))
    z_indices = np.where(z_presence)[0]
    if len(z_indices) == 0:
        print('  [Warning] Image seems empty. Skipping crop.')
        return None

    z_top = z_indices[-1]
    z_bottom_actual = z_indices[0]
    z_spacing = abs(affine[2, 2]) if abs(affine[2, 2]) > 1e-6 else 1.0
    slices_to_keep = int(200 / z_spacing)
    z_crop_start = max(z_bottom_actual, z_top - slices_to_keep)

    heuristic_data = data[:, :, z_crop_start:z_top].copy()
    heuristic_data[heuristic_data < -100] = 0

    new_affine = affine.copy()
    new_origin = nib.affines.apply_affine(affine, np.array([0, 0, z_crop_start]))
    new_affine[:3, 3] = new_origin
    heuristic_img = nib.Nifti1Image(heuristic_data, new_affine, img.header)
    nib.save(heuristic_img, temp_input_path)

    try:
        ants = _require_ants()
        ants_template = ants.image_read(TEMPLATE_PATH)
        ants_native = ants.image_read(temp_input_path)
        print('  [Preprocessing] Registering CT template to native CTA crop...')
        tx = ants.registration(
            fixed=ants_native,
            moving=ants_template,
            type_of_transform='Translation',
            reg_iterations=(100, 50, 0),
            metric='MI',
        )
        template_mask = ants_template * 0 + 1
        native_mask = ants.apply_transforms(
            fixed=ants_native,
            moving=template_mask,
            transformlist=tx['fwdtransforms'],
            interpolator='nearestNeighbor',
        )
        native_mask_np = native_mask.numpy() > 0.5
        coords = np.array(np.nonzero(native_mask_np))
    except Exception as exc:
        print(f'  [Warning] Template crop failed ({exc}). Using heuristic z crop.')
        nib.save(heuristic_img, temp_input_path)
        return (0, data.shape[0], 0, data.shape[1], z_crop_start, z_top, orig_shape, affine, img.header)

    if coords.size == 0:
        print('  [Warning] Template crop produced empty mask. Using heuristic z crop.')
        nib.save(heuristic_img, temp_input_path)
        return (0, data.shape[0], 0, data.shape[1], z_crop_start, z_top, orig_shape, affine, img.header)

    x_min, y_min, z_min = np.min(coords, axis=1)
    x_max, y_max, z_max = np.max(coords, axis=1) + 1
    abs_x_min, abs_x_max = int(x_min), int(x_max)
    abs_y_min, abs_y_max = int(y_min), int(y_max)
    abs_z_min, abs_z_max = int(z_crop_start + z_min), int(z_crop_start + z_max)

    print(
        '  [Crop] Bounds '
        f'X:{abs_x_min}:{abs_x_max} Y:{abs_y_min}:{abs_y_max} Z:{abs_z_min}:{abs_z_max}'
    )

    final_cropped_data = data[abs_x_min:abs_x_max, abs_y_min:abs_y_max, abs_z_min:abs_z_max]
    final_affine = affine.copy()
    final_origin = nib.affines.apply_affine(affine, np.array([abs_x_min, abs_y_min, abs_z_min]))
    final_affine[:3, 3] = final_origin
    final_img = nib.Nifti1Image(final_cropped_data, final_affine, img.header)
    nib.save(final_img, temp_input_path)

    return (abs_x_min, abs_x_max, abs_y_min, abs_y_max, abs_z_min, abs_z_max, orig_shape, affine, img.header)


def _restore_crop_space(prediction_path, final_output_path, crop_info):
    np = _require_numpy()
    nib = _require_nibabel()

    x_start, x_end, y_start, y_end, z_start, z_end, orig_shape, orig_affine, orig_header = crop_info
    if not os.path.exists(prediction_path):
        raise FileNotFoundError(f'No prediction file found to restore: {prediction_path}')

    pred_img = nib.load(prediction_path)
    pred_data = np.asanyarray(pred_img.dataobj)
    restored_data = np.zeros(orig_shape, dtype=pred_data.dtype)
    dx, dy, dz = pred_data.shape

    x_slice = slice(x_start, x_start + min(dx, x_end - x_start))
    y_slice = slice(y_start, y_start + min(dy, y_end - y_start))
    z_slice = slice(z_start, z_start + min(dz, z_end - z_start))
    pred_x_slice = slice(0, min(dx, x_end - x_start))
    pred_y_slice = slice(0, min(dy, y_end - y_start))
    pred_z_slice = slice(0, min(dz, z_end - z_start))

    restored_data[x_slice, y_slice, z_slice] = pred_data[pred_x_slice, pred_y_slice, pred_z_slice]
    restored_img = nib.Nifti1Image(restored_data, orig_affine, orig_header)
    nib.save(restored_img, final_output_path)


def _run_nnunet_dir(input_dir, output_dir, dataset_id, disable_tta=True):
    cmd = [
        _find_nnunet_predict_executable(),
        '-i',
        input_dir,
        '-o',
        output_dir,
        '-d',
        str(dataset_id),
        '-c',
        DEFAULT_CONFIGURATION,
        '-p',
        DEFAULT_PLANS,
        '-tr',
        DEFAULT_TRAINER,
        '-f',
        DEFAULT_FOLD,
        '-chk',
        DEFAULT_CHECKPOINT,
        '-nps',
        '1',
        '-npp',
        '1',
        '-device',
        _get_nnunet_device(),
    ]
    if disable_tta:
        cmd.append('--disable_tta')

    print(f'  [nnU-Net] Running Dataset {dataset_id}...')
    subprocess.run(cmd, check=True, env=_get_fixed_env())


def _copy_and_preprocess_input(input_path, temp_input_path, crop_head):
    sitk = _require_sitk()
    shutil.copy2(input_path, temp_input_path)
    temp_input_image = sitk.ReadImage(temp_input_path)
    temp_input_image = reorient_sitk_image(temp_input_image, MODEL_INPUT_ORIENTATION)
    sitk.WriteImage(temp_input_image, temp_input_path)

    crop_info = None
    if crop_head:
        crop_info = _crop_to_head(temp_input_path)
    return crop_info


def _finalize_prediction(temp_prediction_path, output_path, crop_info, original_orientation):
    sitk = _require_sitk()
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
    sitk = _require_sitk()
    original_orientation = get_orientation_and_closest_plane(sitk.ReadImage(input_path))['orientation']
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix='temp_rsna_nnunet_', dir=output_dir)
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
    sitk = _require_sitk()
    original_orientation = get_orientation_and_closest_plane(sitk.ReadImage(input_path))['orientation']
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix='temp_rsna_cascade_', dir=output_dir)
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

        shutil.copy2(binary_input_path, os.path.join(cascade_in, 'case_0000.nii.gz'))
        shutil.copy2(binary_result, os.path.join(cascade_in, 'case_0001.nii.gz'))
        _run_nnunet_dir(cascade_in, cascade_out, model_config['cascade_dataset'], disable_tta=disable_tta)

        cascade_result = os.path.join(cascade_out, 'case.nii.gz')
        if not os.path.exists(cascade_result):
            raise FileNotFoundError(f'Cascade nnU-Net did not create expected output: {cascade_result}')

        _finalize_prediction(cascade_result, output_path, crop_info, original_orientation)
        if binary_output_path:
            _finalize_prediction(binary_result, binary_output_path, crop_info, original_orientation)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def predict(input_path, output_path, modality, model='base', binary_output_path=None, disable_tta=True):
    """Predict RSNA artery labels for one NIfTI file.

    Parameters
    ----------
    input_path:
        Source CTA/MRA NIfTI.
    output_path:
        Destination label map.
    modality:
        ``'cta'`` or ``'mra'``.
    model:
        ``'base'`` for Dataset601/602 multiclass models, ``'binary'`` for
        Dataset611/612 binary models, or ``'cascade'`` for binary then
        image+binary-mask multiclass prediction.
    binary_output_path:
        Optional binary label output for cascade predictions.
    disable_tta:
        Keep true by default to avoid left/right mirroring with side-specific labels.
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
        f'Running RSNA {model} {modality.upper()} prediction on '
        f'{os.path.basename(input_path)}...'
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


def build_parser():
    parser = argparse.ArgumentParser(description='Predict RSNA artery labels for one CTA/MRA NIfTI.')
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
