import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import SimpleITK as sitk

import predict


class PredictTests(unittest.TestCase):
    def test_parser_accepts_prediction_arguments(self):
        args = predict.build_parser().parse_args(
            [
                "--input",
                "scan.nii.gz",
                "--output",
                "outputs",
                "--modality",
                "mra",
                "--model",
                "topbrain_cascade",
            ]
        )
        self.assertEqual(args.modality, "mra")
        self.assertEqual(args.model, "topbrain_cascade")

    def test_unified_label_is_binary_and_preserves_geometry(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            multiclass_path = root / "multiclass.nii.gz"
            unified_path = root / "unified.nii.gz"

            array = np.array([[[0, 1], [5, 0]], [[2, 0], [0, 9]]], dtype=np.uint8)
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((0.4, 0.5, 0.7))
            image.SetOrigin((12.0, -8.0, 3.5))
            sitk.WriteImage(image, str(multiclass_path), True)

            created = predict._create_unified_label(
                multiclass_path,
                unified_path,
                overwrite=False,
            )
            unified = sitk.ReadImage(str(unified_path))
            unified_array = sitk.GetArrayFromImage(unified)

            self.assertTrue(created)
            self.assertEqual(set(np.unique(unified_array)), {0, 1})
            self.assertTrue(np.array_equal(unified_array, array > 0))
            self.assertEqual(unified.GetSize(), image.GetSize())
            self.assertTrue(
                np.allclose(unified.GetSpacing(), image.GetSpacing(), rtol=0, atol=1e-6)
            )
            self.assertTrue(
                np.allclose(unified.GetOrigin(), image.GetOrigin(), rtol=0, atol=1e-6)
            )

    def test_directory_discovery_excludes_generated_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "scan.nii.gz"
            generated = root / "scan_rsna_multiclass_label.nii.gz"
            source.touch()
            generated.touch()

            inputs = predict._discover_inputs(root, root / "outputs", recursive=False)

            self.assertEqual(inputs, [source.resolve()])

    def test_single_file_prediction_writes_pair_then_skips(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "scan.nii.gz"
            output_path = root / "outputs"
            source = sitk.GetImageFromArray(np.ones((3, 4, 5), dtype=np.int16))
            sitk.WriteImage(source, str(input_path), True)

            engine = mock.Mock()

            def fake_predict(input_file, multiclass_file, **kwargs):
                input_image = sitk.ReadImage(input_file)
                label = sitk.Image(input_image.GetSize(), sitk.sitkUInt8)
                label.CopyInformation(input_image)
                label[1, 1, 1] = 4
                sitk.WriteImage(label, multiclass_file, True)

            engine.predict.side_effect = fake_predict
            patches = (
                mock.patch.object(predict, "_validate_selected_model"),
                mock.patch.object(predict, "_configure_device", return_value="cpu"),
                mock.patch.object(predict, "_load_prediction_engine", return_value=engine),
            )
            with patches[0], patches[1], patches[2]:
                first = predict.predict(
                    input_path,
                    output_path,
                    modality="mra",
                    model="rsna",
                )
                second = predict.predict(
                    input_path,
                    output_path,
                    modality="mra",
                    model="rsna",
                )

            self.assertEqual(first["predicted"], 1)
            self.assertEqual(first["unified_created"], 1)
            self.assertEqual(second["skipped"], 1)
            self.assertEqual(engine.predict.call_count, 1)
            self.assertTrue((output_path / "scan_rsna_multiclass_label.nii.gz").is_file())
            self.assertTrue((output_path / "scan_rsna_unified_label.nii.gz").is_file())


if __name__ == "__main__":
    unittest.main()
