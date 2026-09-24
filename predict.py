#!/usr/bin/env python
"""Prediction-only interface for intracranial artery segmentation."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
CTA_TEMPLATE_PATH = REPO_ROOT / "MNI" / "1mm" / "ct_rorden_lps.nii"
DEFAULT_INPUT_PATH = REPO_ROOT / "working" / "input"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "working" / "output"

TRAINER_DIRECTORY = (
    "nnUNetTrainerNoMirroring__nnUNetResEncUNetMPlans__3d_fullres"
)
CHECKPOINT_NAME = "checkpoint_final.pth"

MODEL_SPECS = {
    "topbrain": {
        "family": "topbrain",
        "engine_model": "base",
        "datasets": {"cta": ("501",), "mra": ("502",)},
    },
    "topbrain_cascade": {
        "family": "topbrain",
        "engine_model": "cascade",
        "datasets": {"cta": ("511", "521"), "mra": ("512", "522")},
    },
    "rsna": {
        "family": "rsna",
        "engine_model": "base",
        "datasets": {"cta": ("601",), "mra": ("602",)},
    },
    "rsna_cascade": {
        "family": "rsna",
        "engine_model": "cascade",
        "datasets": {"cta": ("611", "621"), "mra": ("612", "622")},
    },
}

TEMP_DIRECTORY_PREFIXES = (
    "temp_topbrain_cascade_",
    "temp_topbrain_nnunet_",
    "temp_rsna_cascade_",
    "temp_rsna_nnunet_",
)

# Edit these values to run this file directly with VS Code's Python debugger.
DEBUG_MODALITY = "mra"  # "cta" or "mra"
DEBUG_MODEL = "topbrain_cascade"  # topbrain, topbrain_cascade, rsna, rsna_cascade
DEBUG_INPUT_PATH = DEFAULT_INPUT_PATH  # One NIfTI file or a directory of NIfTI files
DEBUG_OUTPUT_PATH = DEFAULT_OUTPUT_PATH
DEBUG_DEVICE = "auto"  # "auto", "cuda", "cpu", or "mps"
DEBUG_OVERWRITE = False
DEBUG_TTA = False
DEBUG_RECURSIVE = False
DEBUG_SAVE_BINARY = False
DEBUG_FAIL_FAST = False


def _is_nifti(path: Path) -> bool:
    return path.is_file() and (
        path.name.lower().endswith(".nii")
        or path.name.lower().endswith(".nii.gz")
    )


def _nifti_stem(path: Path) -> str:
    lower_name = path.name.lower()
    if lower_name.endswith(".nii.gz"):
        return path.name[:-7]
    if lower_name.endswith(".nii"):
        return path.name[:-4]
    raise ValueError(f"Not a NIfTI file: {path}")


def _is_generated_output(path: Path) -> bool:
    stem = _nifti_stem(path).lower()
    return stem.endswith("_label")


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _find_nnunet_predict_executable() -> Path:
    executable = shutil.which("nnUNetv2_predict")
    if executable:
        return Path(executable)

    python_directory = Path(sys.executable).resolve().parent
    for name in ("nnUNetv2_predict", "nnUNetv2_predict.exe"):
        candidate = python_directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "nnUNetv2_predict was not found. Activate the project environment or "
        "install the packages in requirements.txt."
    )


def _dataset_training_directory(dataset_id: str) -> Path:
    matches = sorted(MODELS_DIR.glob(f"Dataset{dataset_id}_*"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one models/Dataset{dataset_id}_* directory, found {len(matches)}."
        )
    return matches[0] / TRAINER_DIRECTORY


def _validate_selected_model(modality: str, model: str) -> None:
    if not MODELS_DIR.is_dir():
        raise FileNotFoundError(f"Model directory not found: {MODELS_DIR}")
    for dataset_id in MODEL_SPECS[model]["datasets"][modality]:
        training_directory = _dataset_training_directory(dataset_id)
        required_paths = (
            training_directory / "dataset.json",
            training_directory / "plans.json",
            training_directory / "fold_0" / CHECKPOINT_NAME,
        )
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Dataset {dataset_id} is incomplete; missing: " + ", ".join(missing)
            )
    if modality == "cta" and not CTA_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"CTA crop template not found: {CTA_TEMPLATE_PATH}")
    _find_nnunet_predict_executable()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in the active environment.") from exc

    if torch.cuda.is_available():
        return "cuda"
    if sys.platform == "darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _configure_device(device: str) -> str:
    resolved_device = _resolve_device(device)
    os.environ["TOPBRAIN_NNUNET_DEVICE"] = resolved_device
    os.environ["RSNA_NNUNET_DEVICE"] = resolved_device
    return resolved_device


def _discover_inputs(input_path: Path, output_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if not _is_nifti(input_path):
            raise ValueError(f"Input must be a .nii or .nii.gz file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    inputs = []
    for path in iterator:
        if not _is_nifti(path) or _is_generated_output(path):
            continue
        resolved_path = path.resolve()
        if output_path != input_path and resolved_path.is_relative_to(output_path):
            continue
        inputs.append(resolved_path)
    inputs.sort()
    if not inputs:
        scope = "recursively" if recursive else "in the top level"
        raise RuntimeError(f"No NIfTI inputs found {scope} of {input_path}")
    return inputs


def _output_paths(
    input_file: Path,
    input_root: Path | None,
    output_root: Path,
    model: str,
) -> tuple[Path, Path, Path]:
    relative_parent = (
        input_file.parent.relative_to(input_root)
        if input_root is not None
        else Path()
    )
    output_directory = output_root / relative_parent
    stem = _nifti_stem(input_file)
    return (
        output_directory / f"{stem}_{model}_multiclass_label.nii.gz",
        output_directory / f"{stem}_{model}_unified_label.nii.gz",
        output_directory / f"{stem}_{model}_binary_label.nii.gz",
    )


def _validate_input_image(input_path: Path) -> None:
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK is not installed in the active environment.") from exc

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(input_path))
    reader.ReadImageInformation()
    if reader.GetDimension() != 3:
        raise ValueError(
            f"Only 3D NIfTI images are supported; {input_path.name} is "
            f"{reader.GetDimension()}D."
        )
    if any(size <= 0 for size in reader.GetSize()):
        raise ValueError(f"Input image has an empty dimension: {input_path}")


def _create_unified_label(
    multiclass_path: Path,
    unified_path: Path,
    *,
    overwrite: bool,
) -> bool:
    if unified_path.exists() and not overwrite:
        print(f"  Reusing unified label: {unified_path.name}")
        return False

    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError(
            "NumPy and SimpleITK are required to create unified labels."
        ) from exc

    multiclass_image = sitk.ReadImage(str(multiclass_path))
    unified_array = (
        sitk.GetArrayFromImage(multiclass_image) > 0
    ).astype(np.uint8, copy=False)
    unified_image = sitk.GetImageFromArray(unified_array)
    unified_image.CopyInformation(multiclass_image)

    unified_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=unified_path.parent,
        prefix=f".{unified_path.name}.",
        suffix=".nii.gz",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        sitk.WriteImage(unified_image, str(temporary_path), True)
        os.replace(temporary_path, unified_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"  Wrote unified label: {unified_path.name}")
    return True


def _load_prediction_engine(model: str):
    family = MODEL_SPECS[model]["family"]
    if family == "topbrain":
        import _predict_topbrain as engine
    else:
        import _predict_rsna as engine
    return engine


def _cleanup_stale_temp_directories(output_directory: Path) -> int:
    if not output_directory.is_dir():
        return 0
    removed = 0
    for path in output_directory.iterdir():
        if not path.is_dir():
            continue
        if not path.name.startswith(TEMP_DIRECTORY_PREFIXES):
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


def _validate_no_output_collisions(
    inputs: list[Path],
    input_root: Path | None,
    output_root: Path,
    model: str,
) -> None:
    destinations: dict[Path, Path] = {}
    for input_file in inputs:
        for destination in _output_paths(input_file, input_root, output_root, model):
            previous_input = destinations.get(destination)
            if previous_input is not None:
                raise RuntimeError(
                    f"Inputs {previous_input} and {input_file} map to the same output "
                    f"path: {destination}"
                )
            destinations[destination] = input_file


def predict(
    input_path: str | Path,
    output_path: str | Path,
    *,
    modality: str,
    model: str,
    device: str = "auto",
    overwrite: bool = False,
    tta: bool = False,
    recursive: bool = False,
    save_binary: bool = False,
    fail_fast: bool = False,
) -> dict[str, int]:
    """Segment one NIfTI image or a directory of NIfTI images.

    Every successful case produces a multiclass label and a binary unified
    label. Cascade models can optionally retain their first-stage binary mask.
    """
    modality = modality.strip().lower()
    model = model.strip().lower()
    device = device.strip().lower()
    if modality not in {"cta", "mra"}:
        raise ValueError("modality must be 'cta' or 'mra'")
    if model not in MODEL_SPECS:
        raise ValueError(f"model must be one of: {', '.join(MODEL_SPECS)}")
    if device not in {"auto", "cuda", "cpu", "mps"}:
        raise ValueError("device must be 'auto', 'cuda', 'cpu', or 'mps'")
    if save_binary and MODEL_SPECS[model]["engine_model"] != "cascade":
        raise ValueError("save_binary is available only for cascade models")

    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(f"Output path must be a directory: {output_path}")

    input_root = input_path if input_path.is_dir() else None
    inputs = _discover_inputs(input_path, output_path, recursive)
    _validate_no_output_collisions(inputs, input_root, output_path, model)
    _validate_selected_model(modality, model)
    engine = _load_prediction_engine(model)
    resolved_device = _configure_device(device)
    engine_model = MODEL_SPECS[model]["engine_model"]

    print(f"Found {len(inputs)} input NIfTI file(s).")
    print(f"Modality: {modality.upper()}")
    print(f"Model: {model}")
    print(f"Device: {resolved_device}")
    print(f"Output directory: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    started = time.time()
    predicted = 0
    unified_created = 0
    skipped = 0
    failures: list[str] = []
    cleaned_directories: set[Path] = set()

    for index, input_file in enumerate(inputs, start=1):
        multiclass_path, unified_path, binary_path = _output_paths(
            input_file, input_root, output_path, model
        )
        multiclass_path.parent.mkdir(parents=True, exist_ok=True)
        if multiclass_path.parent not in cleaned_directories:
            removed = _cleanup_stale_temp_directories(multiclass_path.parent)
            if removed:
                print(f"Cleaned {removed} stale inference temp directorie(s).")
            cleaned_directories.add(multiclass_path.parent)

        elapsed = time.time() - started
        if index == 1:
            eta = "estimating after first case"
        else:
            average = elapsed / (index - 1)
            eta = f"about {_format_duration(average * (len(inputs) - index + 1))} remaining"
        print(
            f"\n[{index}/{len(inputs)}] {input_file.name} "
            f"({_format_duration(elapsed)} elapsed, {eta})"
        )

        binary_required = save_binary and engine_model == "cascade"
        required_outputs = [multiclass_path, unified_path]
        if binary_required:
            required_outputs.append(binary_path)
        if not overwrite and all(path.is_file() for path in required_outputs):
            print("  Skipping: all requested outputs already exist.")
            skipped += 1
            continue

        try:
            _validate_input_image(input_file)
            if not overwrite and not multiclass_path.exists() and unified_path.exists():
                raise FileExistsError(
                    f"Unified output exists without its multiclass output: {unified_path}. "
                    "Use --overwrite to regenerate the pair."
                )
            if (
                not overwrite
                and binary_required
                and multiclass_path.exists()
                and not binary_path.exists()
            ):
                raise FileExistsError(
                    f"Cascade outputs are incomplete for {input_file.name}. "
                    "Use --overwrite to regenerate them with --save-binary."
                )

            ran_inference = overwrite or not multiclass_path.exists()
            if ran_inference:
                print(f"  Running {model} {modality.upper()} prediction...")
                engine.predict(
                    str(input_file),
                    str(multiclass_path),
                    modality=modality,
                    model=engine_model,
                    binary_output_path=str(binary_path) if binary_required else None,
                    disable_tta=not tta,
                )
                if not multiclass_path.is_file():
                    raise RuntimeError(
                        f"Prediction did not create {multiclass_path}"
                    )
                if binary_required and not binary_path.is_file():
                    raise RuntimeError(f"Prediction did not create {binary_path}")
                predicted += 1
            else:
                print(f"  Reusing multiclass label: {multiclass_path.name}")

            if _create_unified_label(
                multiclass_path,
                unified_path,
                overwrite=overwrite,
            ):
                unified_created += 1
        except Exception as exc:
            message = f"{input_file}: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(f"  Failed: {type(exc).__name__}: {exc}")
            if fail_fast:
                raise

    elapsed = time.time() - started
    print(
        f"\nFinished in {_format_duration(elapsed)}: {predicted} predicted, "
        f"{unified_created} unified labels created, {skipped} skipped, "
        f"{len(failures)} failed."
    )
    if failures:
        raise RuntimeError("Some predictions failed:\n" + "\n".join(failures))
    return {
        "inputs": len(inputs),
        "predicted": predicted,
        "unified_created": unified_created,
        "skipped": skipped,
        "failed": len(failures),
    }


def check_setup() -> None:
    """Validate Python packages, nnU-Net executable, templates, and model files."""
    packages = (
        ("numpy", "numpy"),
        ("SimpleITK", "SimpleITK"),
        ("nibabel", "nibabel"),
        ("antspyx", "ants"),
        ("torch", "torch"),
        ("nnunetv2", "nnunetv2"),
    )
    errors: list[str] = []
    imported_packages = {}
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for distribution_name, module_name in packages:
        try:
            package_version = importlib.metadata.version(distribution_name)
            imported_packages[module_name] = importlib.import_module(module_name)
            print(f"{distribution_name}: {package_version}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Python package is missing: {distribution_name}")
        except Exception as exc:
            errors.append(f"Could not import {distribution_name}: {exc}")

    for engine_module in ("_predict_topbrain", "_predict_rsna"):
        try:
            importlib.import_module(engine_module)
        except Exception as exc:
            errors.append(f"Could not import {engine_module}: {exc}")

    try:
        print(f"nnUNetv2_predict: {_find_nnunet_predict_executable()}")
    except Exception as exc:
        errors.append(str(exc))
    if not CTA_TEMPLATE_PATH.is_file():
        errors.append(f"CTA crop template is missing: {CTA_TEMPLATE_PATH}")

    dataset_ids = sorted(
        {
            dataset_id
            for spec in MODEL_SPECS.values()
            for datasets in spec["datasets"].values()
            for dataset_id in datasets
        }
    )
    complete_datasets = 0
    for dataset_id in dataset_ids:
        try:
            training_directory = _dataset_training_directory(dataset_id)
            dataset_complete = True
            for relative_path in (
                Path("dataset.json"),
                Path("plans.json"),
                Path("fold_0") / CHECKPOINT_NAME,
            ):
                if not (training_directory / relative_path).is_file():
                    dataset_complete = False
                    errors.append(
                        f"Dataset {dataset_id} is missing {relative_path}"
                    )
            complete_datasets += int(dataset_complete)
        except Exception as exc:
            errors.append(str(exc))
    print(f"Model datasets found: {complete_datasets}/{len(dataset_ids)}")

    if not errors:
        torch = imported_packages["torch"]
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        elif sys.platform == "darwin":
            print(f"MPS available: {torch.backends.mps.is_available()}")
        print("Setup check passed.")
        return
    raise RuntimeError("Setup check failed:\n- " + "\n- ".join(errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment intracranial arteries in 3D CTA or MRA NIfTI images and "
            "write multiclass and unified binary labels."
        )
    )
    parser.add_argument("--input", help="Input .nii/.nii.gz file or directory.")
    parser.add_argument("--output", help="Directory in which labels will be written.")
    parser.add_argument("--modality", choices=("cta", "mra"))
    parser.add_argument("--model", choices=tuple(MODEL_SPECS))
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default="auto",
        help="Inference device (default: auto).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable nnU-Net test-time augmentation; disabled by default.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively and preserve subdirectories.",
    )
    parser.add_argument(
        "--save-binary",
        action="store_true",
        help="Also retain the first-stage binary mask from a cascade model.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed case instead of finishing the batch.",
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Validate dependencies and all model files without predicting.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.check_setup:
        check_setup()
        return

    missing = [
        option
        for option, value in (
            ("--input", args.input),
            ("--output", args.output),
            ("--modality", args.modality),
            ("--model", args.model),
        )
        if value is None
    ]
    if missing:
        parser.error("the following arguments are required: " + ", ".join(missing))

    predict(
        args.input,
        args.output,
        modality=args.modality,
        model=args.model,
        device=args.device,
        overwrite=args.overwrite,
        tta=args.tta,
        recursive=args.recursive,
        save_binary=args.save_binary,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        predict(
            DEBUG_INPUT_PATH,
            DEBUG_OUTPUT_PATH,
            modality=DEBUG_MODALITY,
            model=DEBUG_MODEL,
            device=DEBUG_DEVICE,
            overwrite=DEBUG_OVERWRITE,
            tta=DEBUG_TTA,
            recursive=DEBUG_RECURSIVE,
            save_binary=DEBUG_SAVE_BINARY,
            fail_fast=DEBUG_FAIL_FAST,
        )
