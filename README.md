# Intracranial Artery Segmentation

Prediction-only tools for segmenting intracranial arteries in 3D CTA and
time-of-flight MRA NIfTI images. Four trained model configurations are
available:

| Model | Description | Multiclass labels |
| --- | --- | --- |
| `topbrain` | Single-stage TopBrain model | 36 named arterial segments |
| `topbrain_cascade` | Binary artery localization followed by TopBrain classification | 36 named arterial segments |
| `rsna` | Single-stage RSNA model | 13 arterial/aneurysm-location regions |
| `rsna_cascade` | Binary artery localization followed by RSNA classification | 13 arterial/aneurysm-location regions |

Every prediction creates both a multiclass segmentation and a unified binary
artery segmentation. This software is intended for research use only and is
not a medical device or a substitute for clinical review.

## Inputs

Inputs must be three-dimensional `.nii` or `.nii.gz` files. The user must
specify the image modality (`cta` or `mra`); the program does not infer it from
the image. A single file or a directory can be processed.

CTA inputs are automatically cropped to the head for inference and restored to
the original image space. MRA inputs are processed in their supplied field of
view. Output segmentations retain the input image space.

## Installation

An NVIDIA CUDA GPU is strongly recommended. CPU inference is supported but is
much slower. Large cascade inputs can also require substantial system memory;
32 GB RAM should be considered a practical minimum, with 64 GB preferable for
large head-and-neck studies.

Create the supplied Conda environment:

```bash
cd /path/to/artery-segment2
conda env create -f environment.yml
conda activate artery-segment
python predict.py --check-setup
```

Alternatively, use an existing Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
python predict.py --check-setup
```

PyTorch packages are platform-specific. If the pinned PyTorch package is not
appropriate for the local CUDA or Apple environment, install the correct
PyTorch build first and then install the remaining requirements. The setup
check reports the selected Python executable, package versions, model files,
and available accelerator.

## Model Files

Model weights are large and are intentionally excluded from Git by
`models/.gitignore`. Distribute the model bundle separately and place it in
`models/` without changing its directory names. A valid inference model has
this structure:

```text
models/
  Dataset501_TopBrainCTA2026/
    nnUNetTrainerNoMirroring__nnUNetResEncUNetMPlans__3d_fullres/
      dataset.json
      plans.json
      fold_0/
        checkpoint_final.pth
  ...
```

The complete bundle contains datasets `501`, `502`, `511`, `512`, `521`,
`522`, `601`, `602`, `611`, `612`, `621`, and `622`. Do not rename these
directories. Run `python predict.py --check-setup` after installing or moving
the bundle.

Verify the transferred final checkpoints against the included manifest:

```bash
sha256sum --check model_checksums.sha256
```

## Command Line

Predict one MRA using the TopBrain cascade:

```bash
python predict.py \
  --input /data/case01.nii.gz \
  --output /data/segmentations \
  --modality mra \
  --model topbrain_cascade
```

Predict every top-level NIfTI file in a CTA directory using the RSNA model:

```bash
python predict.py \
  --input /data/cta \
  --output /data/segmentations \
  --modality cta \
  --model rsna
```

Process nested directories and preserve their relative layout:

```bash
python predict.py \
  --input /data/mra \
  --output /data/segmentations \
  --modality mra \
  --model rsna_cascade \
  --recursive
```

Required prediction arguments:

- `--input`: one NIfTI file or a directory containing NIfTI files
- `--output`: output directory
- `--modality`: `cta` or `mra`
- `--model`: `topbrain`, `topbrain_cascade`, `rsna`, or `rsna_cascade`

Useful options:

- `--device auto|cuda|cpu|mps`: execution device; `auto` prefers CUDA, then MPS
- `--overwrite`: regenerate existing outputs
- `--recursive`: search nested input directories
- `--save-binary`: retain the first-stage binary mask from a cascade
- `--tta`: enable nnU-Net test-time augmentation; disabled by default
- `--fail-fast`: stop a directory run after its first failed case

Complete help is available with `python predict.py --help`.

## VS Code Debugging

Open `predict.py` and edit the `DEBUG_*` variables near the top of the file:

```python
DEBUG_MODALITY = "mra"
DEBUG_MODEL = "topbrain_cascade"
DEBUG_INPUT_PATH = "/data/case01.nii.gz"
DEBUG_OUTPUT_PATH = "/data/segmentations"
DEBUG_DEVICE = "auto"
```

Then run **Python: Current File** in the VS Code debugger without supplying
command-line arguments. When any command-line argument is supplied, the normal
CLI parser is used instead.

## Outputs

For an input named `case01.nii.gz` and model `topbrain_cascade`, the default
outputs are:

```text
case01_topbrain_cascade_multiclass_label.nii.gz
case01_topbrain_cascade_unified_label.nii.gz
```

The multiclass label contains the model-specific artery class numbers. The
class names and values are stored in the corresponding model's `dataset.json`.
The unified label is `uint8`, with background `0` and every predicted artery
voxel set to `1`.

With `--save-binary`, cascade models additionally write:

```text
case01_topbrain_cascade_binary_label.nii.gz
```

Existing complete output sets are skipped unless `--overwrite` is supplied.
If a prior run stopped after creating only the multiclass label, rerunning will
create the missing unified label without repeating inference.

## Notes

- Test-time augmentation is disabled by default because the models were
  trained with no-mirroring trainers.
- TopBrain cascade inference includes memory-aware cropping and tiling for very
  large MRA volumes.
- Temporary inference directories are removed after each case. Recognized
  leftovers from interrupted runs are cleaned automatically on the next run.
- NIfTI input is deliberate: DICOM conversion and de-identification should be
  handled before using this repository.

## Citation

The inference pipeline uses nnU-Net. Please cite:

> Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a
> self-configuring method for deep learning-based biomedical image
> segmentation. *Nature Methods*. 2021;18:203-211.

## License

Code is provided under the [MIT License](LICENSE). Model weights and imaging
templates may have separate distribution or use terms; confirm those terms
before redistributing the complete model bundle.
