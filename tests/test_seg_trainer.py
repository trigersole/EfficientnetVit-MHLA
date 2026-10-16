import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from applications.efficientvit_seg.train_efficientvit_seg_model import (
    SegmentationDataset,
    compute_confusion_matrix,
    learning_rate_at_step,
    save_model_weights,
)


class SegmentationTrainerTest(unittest.TestCase):
    def test_learning_rate_warmup_and_cosine_bounds(self):
        values = [learning_rate_at_step(step, 100, 10, 1.0e-3, 0.01) for step in range(101)]
        self.assertAlmostEqual(values[0], 1.0e-4)
        self.assertAlmostEqual(values[9], 1.0e-3)
        self.assertAlmostEqual(values[10], 1.0e-3)
        self.assertAlmostEqual(values[-1], 1.0e-5)
        self.assertTrue(all(math.isfinite(value) and value > 0 for value in values))

    def test_confusion_matrix_ignores_void_pixels(self):
        target = torch.tensor([[0, 1, -1], [1, 2, 2]])
        prediction = torch.tensor([[0, 2, 1], [1, 0, 2]])
        actual = compute_confusion_matrix(prediction, target, 3)
        expected = torch.tensor([[1, 0, 0], [0, 1, 1], [1, 0, 1]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)

    def test_cityscapes_discovery_mapping_and_shapes(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "leftImg8bit" / "train" / "sample_city"
            label_dir = root / "gtFine" / "train" / "sample_city"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            image = np.full((12, 20, 3), 127, dtype=np.uint8)
            label = np.full((12, 20), 7, dtype=np.uint8)
            label[:, 10:] = 26
            label[0, 0] = 0
            Image.fromarray(image).save(image_dir / "sample_000000_000000_leftImg8bit.png")
            Image.fromarray(label).save(label_dir / "sample_000000_000000_gtFine_labelIds.png")

            dataset = SegmentationDataset(
                str(root), "cityscapes", "train", (8, 16), training=True, scale_range=(1.0, 1.0)
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["data"].shape), (3, 8, 16))
            self.assertEqual(tuple(sample["label"].shape), (8, 16))
            self.assertTrue(set(sample["label"].unique().tolist()).issubset({-1, 0, 13}))

    def test_best_weight_file_is_evaluator_compatible(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            path = Path(temporary_directory) / "best.pt"
            model = torch.nn.Conv2d(3, 2, 1)
            save_model_weights(path, model)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(set(checkpoint), {"state_dict"})
            self.assertEqual(set(checkpoint["state_dict"]), set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
