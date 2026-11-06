import math
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image

from applications.efficientvit_seg.train_efficientvit_seg_model import (
    SegmentationDataset,
    compute_confusion_matrix,
    learning_rate_at_step,
    save_model_weights,
    build_optimizer,
    load_compatible_weights,
    parse_args,
    evaluate,
    DistributedEvalSampler,
)


def evaluate_uneven_shard(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.BatchNorm2d(2))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.copy_(torch.tensor([1.0, 0.0]))
    model = torch.nn.parallel.DistributedDataParallel(model)
    data = [{"data": torch.ones(3, 4, 8), "label": torch.zeros(8, 16, dtype=torch.long)} for _ in range(3)]
    loader = torch.utils.data.DataLoader(data, sampler=DistributedEvalSampler(data, rank, 2))
    miou, _ = evaluate(model, loader, torch.device("cpu"), 2, False, 2)
    if rank == 0:
        torch.save(miou, output)
    dist.destroy_process_group()


class SegmentationTrainerTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
    def test_uneven_ddp_validation_and_native_masks(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            root = Path(temporary_directory)
            output = root / "result.pt"
            mp.spawn(evaluate_uneven_shard, args=((root / "rendezvous").as_uri(), str(output)), nprocs=2)
            self.assertEqual(torch.load(output, weights_only=True), 100.0)

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

            dataset.photometric = True
            dataset.cat_max_ratio = 0.75
            for seed in range(5):
                random.seed(seed)
                augmented = dataset[0]
                self.assertTrue(torch.isfinite(augmented["data"]).all())
                self.assertTrue(set(augmented["label"].unique().tolist()).issubset({-1, 0, 13}))
            dataset.training = False
            dataset.native_val_labels = True
            sample = dataset[0]
            self.assertEqual(tuple(sample["data"].shape), (3, 8, 16))
            self.assertEqual(tuple(sample["label"].shape), (12, 20))
            self.assertEqual(sample["label"][0, 0].item(), -1)
            self.assertTrue(torch.all(sample["label"][:, 10:] == 13))

    def test_recipe_can_be_overridden_and_legacy_stays_available(self):
        base = ["train", "--data-root", "data", "--output-dir", "run"]
        with patch("sys.argv", base):
            self.assertFalse(parse_args().photometric)
        with patch("sys.argv", base + ["--recipe", "cityscapes_v2"]):
            args = parse_args()
            self.assertTrue(args.photometric)
            self.assertTrue(args.native_val_labels)
            self.assertEqual(args.cat_max_ratio, 0.75)
            self.assertEqual(args.mhla_eps, 1e-6)
        with patch("sys.argv", base + ["--recipe", "cityscapes_v2", "--no-photometric"]):
            self.assertFalse(parse_args().photometric)

    def test_pretrained_backbone_lr_and_no_decay_groups(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Conv2d(3, 2, 1)
        model.head = torch.nn.Conv2d(2, 2, 1)
        optimizer = build_optimizer(model, 1e-3, 0.01, 0.1)
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                is_backbone = parameter is model.backbone.weight or parameter is model.backbone.bias
                self.assertAlmostEqual(group["lr"], 1e-4 if is_backbone else 1e-3)
                self.assertEqual(group["weight_decay"], 0 if parameter.ndim == 1 else 0.01)

    def test_fresh_backbone_parameters_keep_full_lr(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.old = torch.nn.Conv2d(3, 2, 1)
        model.backbone.new = torch.nn.Conv2d(2, 2, 1)
        model.head = torch.nn.Conv2d(2, 2, 1)
        fresh = {"backbone.new.weight", "backbone.new.bias"}
        optimizer = build_optimizer(model, 1e-3, 0.01, 0.1, fresh)
        slow = {id(model.backbone.old.weight), id(model.backbone.old.bias)}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.assertAlmostEqual(group["lr"], 1e-4 if id(parameter) in slow else 1e-3)

    def test_backbone_scope_and_sliced_projection_init(self):
        source = {
            "backbone.blk.proj.conv.weight": torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8, 1, 1),
            "backbone.blk.qkv.conv.weight": torch.ones(6, 2, 1, 1),
            "head.op_list.0.weight": torch.ones(2, 2, 1, 1),
        }
        model = torch.nn.Module()
        model.backbone = torch.nn.Module()
        model.backbone.blk = torch.nn.Module()
        model.backbone.blk.proj = torch.nn.Module()
        model.backbone.blk.proj.conv = torch.nn.Conv2d(4, 2, 1, bias=False)
        model.backbone.blk.qkv = torch.nn.Module()
        model.backbone.blk.qkv.conv = torch.nn.Conv2d(2, 6, 1, bias=False)
        model.head = torch.nn.Module()
        model.head.op_list = torch.nn.ModuleList([torch.nn.Conv2d(2, 2, 1, bias=False)])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            path = Path(temporary_directory) / "init.pt"
            torch.save({"state_dict": source}, path)
            report = load_compatible_weights(model, str(path), scope="backbone", inherit_sliced_proj=True)
        self.assertEqual(report["sliced"], ["backbone.blk.proj.conv.weight"])
        self.assertEqual(report["shape_mismatch"], [])
        self.assertIn("head.op_list.0.weight", report["missing"])
        torch.testing.assert_close(model.backbone.blk.proj.conv.weight, source["backbone.blk.proj.conv.weight"][:, :4])
        self.assertFalse(torch.equal(model.head.op_list[0].weight, torch.ones(2, 2, 1, 1)))

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
