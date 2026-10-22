"""Offline CPU smoke tests. Place in project root and run: python smoke_test_overlap.py.

Synthetic forward/backward steps here are tests, not research training. They do
not write model weights into OUTPUT_DIR and do not change the 50-epoch protocol.
"""
import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import albumentations as A
import numpy as np
from PIL import Image
import torch

from common import settings_patches_overlap as settings
from common import data_preparation_patches_overlap as data
from common import metrics_patches_overlap as metrics
from common.experiment_utils_overlap import (
    strict_rgb_to_labels, seed_everything, preflight_dataset,
    validate_patient_splits, fold_manifest,
)
from UNETPP_WCD.model_patches_overlap import ImprovedUNetPlusPlus, VesselAwareLoss

ROOT = Path(__file__).resolve().parent


class OverlapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        seed_everything(42)

    def test_python_syntax(self):
        for relative in (
            "common/settings_patches_overlap.py",
            "common/data_preparation_patches_overlap.py",
            "common/experiment_utils_overlap.py",
            "common/metrics_patches_overlap.py",
            "UNETPP_WCD/model_patches_overlap.py",
            "run_pipeline_UNET_WCD_patches_overlap.py",
        ):
            path = ROOT / relative
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_mask_decoder(self):
        rgb = np.array(list(settings.COLOR_MAP.values()), dtype=np.uint8)[None]
        np.testing.assert_array_equal(strict_rgb_to_labels(rgb), np.arange(5)[None])
        rgb[0, 0] = [17, 23, 41]
        with self.assertRaises(ValueError):
            strict_rgb_to_labels(rgb)

    def test_coords_cover_small_and_irregular_images(self):
        for h, w in ((13, 17), (32, 32), (61, 93)):
            cover = np.zeros((h, w), dtype=np.int32)
            for y, x in metrics.iter_patch_coords(h, w, 32, 16):
                cover[y:min(y+32, h), x:min(x+32, w)] += 1
            self.assertTrue(np.all(cover > 0))
        with self.assertRaises(ValueError):
            list(metrics.iter_patch_coords(64, 64, 32, 40))

    def test_loss_gradients_and_empty_foreground(self):
        criterion = VesselAwareLoss(torch.ones(5), 0.25, 0.1, 3)
        criterion.topology_scale = 1.0
        for empty in (True, False):
            logits = torch.randn(2, 5, 32, 32, requires_grad=True)
            targets = torch.zeros(2, 32, 32, dtype=torch.long)
            if not empty:
                targets[:, 8:24, 14:18] = 1
                targets[:, 14:18, 8:24] = 2
            loss = criterion(logits, targets)
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_perfect_probability_blending(self):
        class ConstantModel:
            def predict_proba(self, images, tta=False):
                probs = torch.zeros(images.shape[0], 5, images.shape[2], images.shape[3])
                probs[:, 1] = 1
                return probs
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            for source, shape in (("small", (13, 17)), ("irregular", (61, 93))):
                Image.fromarray(np.zeros((*shape, 3), dtype=np.uint8)).save(root / "images" / f"{source}.png")
                mask = np.empty((*shape, 3), dtype=np.uint8)
                mask[:] = settings.COLOR_MAP[1]
                Image.fromarray(mask).save(root / "masks" / f"{source}.png")
            rows = []
            result, images = metrics.evaluate_fold_overlap(
                ConstantModel(), ["small", "irregular"], root / "images", root / "masks",
                5, settings.COLOR_MAP, settings.CLASS_NAMES,
                [A.Normalize(mean=settings.MEAN, std=settings.STD)],
                patch_size=32, stride=16, batch_size=2, verbose=False,
                per_image_records=rows, compute_topology=True,
            )
            self.assertAlmostEqual(result["arteries_dice"], 1.0, places=6)
            for pred, gt in images.values():
                np.testing.assert_array_equal(pred, gt)
            self.assertEqual(rows[0]["vessel_union_hd95_px"], 0.0)
            intervals = metrics.bootstrap_per_image(__import__("pandas").DataFrame(rows), samples=10)
            self.assertFalse(intervals.empty)

    def test_fit_runs_exactly_50_epochs(self):
        # Exercise the real fit/scheduler/callback loop with a tiny network.
        # Full UNet++ forward/backward is covered separately below.
        with tempfile.TemporaryDirectory() as temporary:
            mask_path = Path(temporary) / "mask.png"
            rgb = np.zeros((32, 32, 3), dtype=np.uint8)
            rgb[8:24, 14:18] = settings.COLOR_MAP[1]
            Image.fromarray(rgb).save(mask_path)
            labels = torch.from_numpy(strict_rgb_to_labels(rgb)).long()
            dataset = torch.utils.data.TensorDataset(
                torch.rand(2, 3, 32, 32), labels.unsqueeze(0).repeat(2, 1, 1)
            )
            dataset.mask_paths = [mask_path, mask_path]
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            model = ImprovedUNetPlusPlus(encoder_weights=None, device="cpu",
                                         union_coef=0.25, topology_coef=0.10)
            model.model = torch.nn.Conv2d(3, 5, kernel_size=1)
            callbacks = []
            model.fit(loader, on_epoch_end=lambda history: callbacks.append(len(history)))
            self.assertEqual(model.trained_epochs, 50)
            self.assertEqual(model.selected_epoch, 50)
            self.assertEqual(len(model.history), 50)
            self.assertEqual(callbacks, list(range(1, 51)))
            self.assertEqual(model.scheduler.last_epoch, 50 * len(loader))

    def test_synthetic_data_model_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("patches/images", "patches/masks", "full/images", "full/masks"):
                (root / relative).mkdir(parents=True)
            rng = np.random.default_rng(42)
            for index in range(6):
                source = f"source{index}"
                image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                labels = np.zeros((32, 32), dtype=np.uint8)
                for cls in range(1, 5):
                    labels[cls*4:cls*4+3, 5:27] = cls
                rgb_mask = np.array(list(settings.COLOR_MAP.values()), dtype=np.uint8)[labels]
                Image.fromarray(image).save(root / "full/images" / f"{source}.png")
                Image.fromarray(rgb_mask).save(root / "full/masks" / f"{source}.png")
                Image.fromarray(image).save(root / "patches/images" / f"{source}_y0_x0.png")
                Image.fromarray(rgb_mask).save(root / "patches/masks" / f"{source}_y0_x0.png")
            with patch.multiple(settings,
                DATASET_IMG_DIR=root / "patches/images", DATASET_MASK_DIR=root / "patches/masks",
                NORMALIZED_IMG_DIR=root / "full/images", NORMALIZED_MASK_DIR=root / "full/masks",
                PATCH_SIZE=32, STRIDE=16, FOLDS_NUM=2, TEST_SIZE=2,
                BATCH_SIZE=2, NUM_WORKERS=0, PATIENT_GROUPS_CSV=None,
            ):
                audit = preflight_dataset(verbose=False)
                normalize = [A.Normalize(mean=settings.MEAN, std=settings.STD)]
                loaders = data.create_cross_val_loaders(normalize, normalize, verbose=False)
                validate_patient_splits(loaders, audit["patient_mapping"])
                train, val = loaders[0]
                manifest = fold_manifest(train, val, 1, audit)
                model = ImprovedUNetPlusPlus(encoder_weights=None, device="cpu")
                weights = model._build_class_weights(train)
                self.assertEqual(sum(model.class_counts), len(train.dataset) * 32 * 32)
                images, masks = next(iter(train))
                criterion = VesselAwareLoss(weights, topology_coef=0)
                model.model.train()
                loss = criterion(model.model(images), masks)
                loss.backward()
                optimizer = torch.optim.AdamW(model.model.parameters(), lr=1e-4)
                optimizer.step()
                probs = model.predict_proba(images, tta=True)
                self.assertEqual(tuple(probs.shape), (2, 5, 32, 32))
                self.assertTrue(torch.allclose(probs.sum(1), torch.ones(2, 32, 32), atol=1e-5))
                # Test serialization only, NOT claiming that this model trained 50 epochs.
                model.is_fitted = True
                model.metadata = manifest
                model.class_weights = weights
                checkpoint = root / "test_checkpoint.pth"
                model.save(checkpoint)
                restored = ImprovedUNetPlusPlus(encoder_weights=None, device="cpu")
                restored.load(checkpoint, expected_protocol=manifest["protocol_sha256"])
                torch.testing.assert_close(model.predict_proba(images), restored.predict_proba(images))
                with self.assertRaises(ValueError):
                    restored.load(checkpoint, expected_protocol="wrong_protocol")


if __name__ == "__main__":
    unittest.main(verbosity=2)
