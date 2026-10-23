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

    def _sampler_fixture(self, root, missing_uncertain=False, one_rare_patch=False):
        images, masks = [], []
        for source in range(4):
            for patch_index in range(10):
                name = f"source{source}_y0_x{patch_index * 8}.png"
                image_path, mask_path = root / "images" / name, root / "masks" / name
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                labels = np.zeros((8, 8), dtype=np.uint8)
                has_crossing = (source == 0 and patch_index == 0) if one_rare_patch else patch_index in (0, 1)
                if has_crossing:
                    labels[1:3, 1:3] = 3
                if not missing_uncertain and patch_index in (0, 2):
                    labels[4:6, 4:6] = 4
                palette = np.array([settings.COLOR_MAP[c] for c in range(5)], dtype=np.uint8)
                Image.fromarray(palette[labels]).save(mask_path)
                # Sampler must read masks only: intentionally do not write image files.
                images.append(image_path)
                masks.append(mask_path)
        return images, masks

    def test_rare_sampler_quotas_caps_determinism(self):
        with tempfile.TemporaryDirectory() as temporary:
            images, masks = self._sampler_fixture(Path(temporary))
            sampler = data.RareClassMixSampler(
                images, masks, seed=123, fractions={3: 0.25, 4: 0.15},
                min_pixels=1, max_patch_repeats=4, max_source_multiplier=2.0,
            )
            sampler.set_epoch(0)
            first = list(sampler)
            self.assertEqual(first, list(sampler))
            self.assertEqual(len(first), 40)
            self.assertTrue(all(0 <= index < 40 for index in first))
            draws = np.bincount(first, minlength=40)
            self.assertLessEqual(draws.max(), 4)
            sources = np.bincount(sampler.source_ids[first], minlength=4)
            self.assertTrue(np.all(sources <= sampler.source_caps))
            stats = sampler.last_epoch_stats
            self.assertEqual(stats["sampling_uniform_slots"], 24)
            self.assertEqual(stats["sampling_targeted_crossings"], 10)
            self.assertEqual(stats["sampling_targeted_uncertain_vessels"], 6)
            self.assertEqual(stats["sampling_fallback_slots"], 0)
            self.assertEqual(stats["sampling_unique_patches"], np.count_nonzero(draws))
            self.assertEqual(sum(row["draws"] for row in sampler.last_source_records), 40)
            self.assertEqual(len(sampler.audit_records()), 40)
            second = data.RareClassMixSampler(
                images, masks, seed=123, fractions={3: 0.25, 4: 0.15},
                min_pixels=1, max_patch_repeats=4, max_source_multiplier=2.0,
            )
            self.assertEqual(first, list(second))
            sampler.set_epoch(1)
            self.assertNotEqual(first, list(sampler))
            second.set_epoch(1)
            self.assertEqual(list(sampler), list(second))

    def test_rare_sampler_missing_class_and_exhausted_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            images, masks = self._sampler_fixture(
                Path(temporary), missing_uncertain=True, one_rare_patch=True,
            )
            with self.assertWarns(UserWarning):
                sampler = data.RareClassMixSampler(
                    images, masks, seed=7, fractions={3: 0.25, 4: 0.15},
                    min_pixels=1, max_patch_repeats=2, max_source_multiplier=1.0,
                )
            indices = list(sampler)
            self.assertEqual(len(indices), 40)
            self.assertLessEqual(np.bincount(indices, minlength=40).max(), 2)
            np.testing.assert_array_equal(
                np.bincount(sampler.source_ids[indices], minlength=4), [10, 10, 10, 10],
            )
            self.assertEqual(sampler.last_epoch_stats["sampling_targeted_uncertain_vessels"], 0)
            self.assertLessEqual(sampler.last_epoch_stats["sampling_targeted_crossings"], 2)
            self.assertGreaterEqual(sampler.last_epoch_stats["sampling_fallback_slots"], 14)

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
            class RecordingSampler(torch.utils.data.Sampler):
                def __init__(self):
                    self.epochs, self.last_epoch_stats = [], {}
                def set_epoch(self, epoch):
                    self.epochs.append(epoch)
                def __len__(self):
                    return 2
                def __iter__(self):
                    self.last_epoch_stats = {"sampling_draws": 2}
                    return iter([0, 1])
            sampler = RecordingSampler()
            loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler)
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
            self.assertEqual(sampler.epochs, list(range(50)))
            self.assertTrue(all(row["sampling_draws"] == 2 for row in model.history))

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
                self.assertEqual(manifest["training_sampler"], train.sampler.protocol_config)
                train_keys = set(manifest["train_sources"])
                self.assertTrue(set(train.sampler.source_keys).issubset(train_keys))
                self.assertTrue(set(train.sampler.source_keys).isdisjoint(manifest["val_sources"]))
                with patch.object(settings, "RARE_SAMPLER_ENABLED", False):
                    baseline_loaders = data.create_cross_val_loaders(normalize, normalize, verbose=False)
                    baseline_train, baseline_val = baseline_loaders[0]
                    baseline_manifest = fold_manifest(baseline_train, baseline_val, 1, audit)
                self.assertIsInstance(baseline_train.sampler, torch.utils.data.RandomSampler)
                self.assertEqual(manifest["train_sources"], baseline_manifest["train_sources"])
                self.assertEqual(manifest["val_sources"], baseline_manifest["val_sources"])
                self.assertEqual(len(train), len(baseline_train))
                self.assertNotEqual(manifest["protocol_sha256"], baseline_manifest["protocol_sha256"])
                self.assertNotIn("training_sampler", baseline_manifest)
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
