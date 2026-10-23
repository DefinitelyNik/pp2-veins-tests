"""Run from project root. Use --help for checks/evaluation-only mode."""
import argparse
import gc
from importlib import metadata
from pathlib import Path
import platform
import zipfile

import pandas as pd
import torch

from common import settings_patches_overlap as settings
from common import data_preparation_patches_overlap as data_prep
from common import metrics_patches_overlap as mpo
from common.experiment_utils_overlap import (
    seed_everything, write_json, preflight_dataset,
    validate_patient_splits, fold_manifest,
)
from UNETPP_WCD.model_patches_overlap import (
    ImprovedUNetPlusPlus, train_extra_transforms, val_extra_transforms,
)

ModelClass = ImprovedUNetPlusPlus


def versions():
    result = {"python": platform.python_version(), "torch": str(torch.__version__),
              "device": str(settings.DEVICE), "cuda": torch.version.cuda}
    for package in ("torchvision", "segmentation-models-pytorch", "albumentations",
                    "numpy", "pandas", "scipy", "scikit-image", "Pillow", "timm"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "not installed"
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def archive_csv_metrics_and_logs():
    """Copy all run CSVs to an additional ZIP; keep original outputs untouched.

    Rebuild after each CSV save, including each training epoch. Both evaluation
    variants are included if present. Checkpoints, images and other non-CSV
    files are deliberately excluded. Replace the previous ZIP only on success.
    """
    output = Path(settings.OUTPUT_DIR).resolve()
    csv_paths = sorted(path for path in output.rglob("*.csv") if path.is_file())
    if not csv_paths:
        return None

    model_dir = Path(__file__).resolve().parent / "UNETPP_WCD"
    model_dir.mkdir(parents=True, exist_ok=True)
    archive_path = model_dir / f"{output.name}_metrics_logs.zip"
    temporary = archive_path.with_name(archive_path.name + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
        ) as archive:
            for csv_path in csv_paths:
                # Preserve paths so evaluation and evaluation_tta cannot collide.
                member = (Path(output.name) / csv_path.relative_to(output)).as_posix()
                archive.write(csv_path, arcname=member)
        temporary.replace(archive_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return archive_path


def save_csv(frame, path, index=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)
    # Also captures fold_*_history.csv written by the epoch-end callback.
    archive_csv_metrics_and_logs()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="Audit ALL source/patch pairs and splits; do not train or download weights.")
    parser.add_argument("--use-fitted", action="store_true",
                        help="Evaluate matching new-format checkpoints without downloading ImageNet weights.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly allow replacing an existing run (no automatic resume).")
    parser.add_argument("--baseline-sampler", action="store_true",
                        help="Disable rare-class sampling; default to original vessel_aware_50ep output.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional separate run directory; never reuse baseline weights for rare sampling.")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--no-topology-metrics", action="store_true",
                        help="Skip CPU hard skeleton/HD95 evaluation; training loss is unchanged.")
    parser.add_argument("--tta", action="store_true", default=settings.USE_TTA,
                        help="Four-flip inference, separately marked in output.")
    args = parser.parse_args(argv)
    if args.use_fitted and args.overwrite:
        parser.error("--use-fitted and --overwrite are mutually exclusive.")
    if args.baseline_sampler:
        settings.RARE_SAMPLER_ENABLED = False
        if args.output_dir is None:
            settings.OUTPUT_DIR = settings.BASE_DIR / "UNETPP_WCD" / "runs" / "vessel_aware_50ep"
    if args.output_dir is not None:
        settings.OUTPUT_DIR = args.output_dir.expanduser().resolve()

    settings.validate_settings()
    seed_everything(settings.SEED)
    print("Environment:", versions())
    print("Auditing the complete dataset before any training...")
    audit = preflight_dataset(verbose=True)
    loaders = data_prep.create_cross_val_loaders(
        train_extra_transforms=train_extra_transforms,
        val_extra_transforms=val_extra_transforms,
    )
    validate_patient_splits(loaders, audit["patient_mapping"])
    manifests = [fold_manifest(tr, vl, i, audit)
                 for i, (tr, vl) in enumerate(loaders, 1)]
    if args.check_only:
        for manifest in manifests:
            print(f"Fold {manifest['fold']}: train={len(manifest['train_sources'])}, "
                  f"validation={len(manifest['val_sources'])}, "
                  f"batches/epoch={manifest['train_batches_per_epoch']}")
        print("DATA AUDIT PASSED. No training or weight download performed.")
        return

    output = settings.OUTPUT_DIR
    eval_dir = output / ("evaluation_tta" if args.tta else "evaluation")
    checkpoint_paths = [ModelClass.get_model_save_path(i) for i in range(1, len(loaders)+1)]
    if args.use_fitted:
        missing = [str(path) for path in checkpoint_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoints: {missing}")
    elif output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output}. "
            "Choose a new OUTPUT_DIR or explicitly pass --overwrite."
        )
    if args.use_fitted and (eval_dir / "per_image_metrics.csv").exists():
        print("Existing evaluation CSVs will be replaced; checkpoints are unchanged.")

    output.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    write_json(eval_dir / "environment.json", versions())
    write_json(eval_dir / "evaluation_config.json", {
        "tta": args.tta, "hard_topology_metrics": not args.no_topology_metrics,
        "blending": "softmax_weighted_hann_floor_0.001",
        "empty_class_dice": 0, "fov": "all_pixels_no_fov_mask",
        "hd95": "pooled_bidirectional_8_connected_surface_distances_pixels",
        "bootstrap_scope": "conditional_on_trained_models_not_training_seed_variability",
        "dataset_sha256": audit["dataset_sha256"],
    })
    write_json(eval_dir / "data_audit.json", audit)
    all_image_records, raw_metrics = [], []
    for fold, ((train_loader, val_loader), manifest) in enumerate(zip(loaders, manifests), 1):
        seed_everything(settings.SEED + fold)
        print(f"\n===== FOLD {fold}/{len(loaders)} =====")
        # Never allocate all fold models at once. For inference, do not download
        # pretrained encoder weights which are already part of the checkpoint.
        model = ModelClass(encoder_weights=None if args.use_fitted else settings.ENCODER_WEIGHTS)
        checkpoint = checkpoint_paths[fold-1]
        if args.use_fitted:
            model.load(checkpoint, expected_protocol=manifest["protocol_sha256"])
            if model.trained_epochs != 50 or model.selected_epoch != 50:
                raise ValueError("Expected fixed epoch-50 checkpoints for this pipeline.")
        else:
            model.metadata = manifest
            write_json(output / f"fold_{fold}_manifest.json", manifest)
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "audit_records"):
                save_csv(pd.DataFrame(sampler.audit_records()),
                         output / f"fold_{fold}_sampler_audit.csv")
                write_json(output / f"fold_{fold}_sampler_config.json", sampler.protocol_config)
            source_sampling_history = []
            def log_history(history, current_fold=fold, current_sampler=sampler,
                            source_rows=source_sampling_history):
                save_csv(pd.DataFrame(history), output / f"fold_{current_fold}_history.csv")
                records = getattr(current_sampler, "last_source_records", [])
                if records:
                    source_rows.extend(records)
                    save_csv(pd.DataFrame(source_rows),
                             output / f"fold_{current_fold}_sampling_by_source.csv")
            # Outer validation is NOT used for checkpoint selection.
            model.fit(train_loader, val_loader=None, save_best=False,
                      on_epoch_end=log_history)
            model.save(checkpoint)
            write_json(output / f"fold_{fold}_class_statistics.json", {
                "train_class_counts": model.class_counts,
                "ce_weights": model.class_weights.detach().cpu().tolist(),
            })
        fold_records = []
        metrics, _ = mpo.evaluate_fold_overlap(
            model=model, val_source_keys=manifest["val_sources"],
            normalized_img_dir=settings.NORMALIZED_IMG_DIR,
            normalized_msk_dir=settings.NORMALIZED_MASK_DIR,
            num_classes=settings.NUM_CLASSES, color_map=settings.COLOR_MAP,
            class_names=settings.CLASS_NAMES, val_extra_transforms=val_extra_transforms,
            patch_size=settings.PATCH_SIZE, stride=settings.STRIDE,
            batch_size=settings.BATCH_SIZE, tta=args.tta, verbose=True,
            per_image_records=fold_records, fold_num=fold,
            visualization_dir=None if args.no_visualize else eval_dir / "visualizations",
            keep_visualizations=False, compute_topology=not args.no_topology_metrics,
            patient_groups=audit["patient_mapping"],
            confusion_matrix_path=eval_dir / f"fold_{fold}_confusion.npy",
        )
        all_image_records.extend(fold_records)
        raw_metrics.append(metrics)
        save_csv(pd.DataFrame(raw_metrics), eval_dir / "raw_metrics_patches_overlap.csv")
        save_csv(pd.DataFrame(all_image_records), eval_dir / "per_image_metrics.csv")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_df = pd.DataFrame(raw_metrics)
    summary = raw_df.agg(["mean", "std"]).T
    summary.index.name = "metric_name"
    save_csv(summary, eval_dir / "fold_mean_std.csv", index=True)
    intervals = mpo.bootstrap_per_image(
        pd.DataFrame(all_image_records), confidence_level=settings.CONFIDENCE_LEVEL,
        samples=settings.BOOTSTRAP_SAMPLES, seed=settings.SEED,
    )
    save_csv(intervals, eval_dir / "interval_metrics_patches_overlap.csv")
    print("\nFold summary (pixel-pooled within each fold):")
    print(summary)
    print(f"Results saved in: {eval_dir}")
    archive_path = archive_csv_metrics_and_logs()
    if archive_path is not None:
        print(f"Additional CSV metrics/history archive saved in: {archive_path}")
    print("Bootstrap intervals describe mean per-image OOF metrics conditional on these models.")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()