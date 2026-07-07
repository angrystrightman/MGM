import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compass_biome_utils import (
    alignment_report,
    biome_enrichment_rows,
    bray_curtis_similarity,
    filter_and_normalize_taxa,
    mean_baseline_predictions,
    normalize_mgnify_metadata,
    program_diagnostics_rows,
    reconstruction_metric_rows,
    select_balanced_sample_ids,
    select_top_taxa_by_train,
    stratified_split_sample_ids,
    task_suitability,
    top_taxa_rows,
    top_k_recall,
)
from scripts.audit_microcorpus_260k import summarize_metadata
from scripts.build_compass_pilot_dataset import write_pilot_dataset
from scripts.extract_compass_embeddings import last_valid_token_embeddings, resolve_sample_positions
from scripts.run_compass_bottleneck_pilot import run_pilot, train_bottleneck_model


class CompassBiomeMetadataTests(unittest.TestCase):
    def test_normalize_mgnify_metadata_names_hierarchy_columns(self):
        raw = pd.DataFrame(
            {
                "Unnamed: 0": ["S1", "S2", "S3"],
                "1": ["Host-associated", "Environmental", "Host-associated"],
                "2": ["Human", "Aquatic", "Human"],
                "3": ["Digestive system", "Marine", "Skin"],
                "4": ["Large intestine", None, "Oral"],
                "5": ["Fecal", None, None],
            }
        )

        normalized = normalize_mgnify_metadata(raw)

        self.assertEqual(
            list(normalized.columns),
            ["sample_id", "biome_1", "biome_2", "biome_3", "biome_4", "biome_5"],
        )
        self.assertEqual(normalized["sample_id"].tolist(), ["S1", "S2", "S3"])
        self.assertEqual(normalized["biome_1"].tolist(), ["Host-associated", "Environmental", "Host-associated"])

    def test_alignment_report_counts_matches_unmatched_and_duplicates(self):
        report = alignment_report(
            corpus_ids=["S1", "S2", "S2", "S4"],
            metadata_ids=["S1", "S3", "S4", "S4"],
        )

        self.assertEqual(report["num_corpus_samples"], 4)
        self.assertEqual(report["num_metadata_rows"], 4)
        self.assertEqual(report["num_matched_samples"], 2)
        self.assertEqual(report["num_unmatched_in_corpus"], 1)
        self.assertEqual(report["num_unmatched_in_metadata"], 1)
        self.assertEqual(report["duplicate_corpus_sample_ids"], ["S2"])
        self.assertEqual(report["duplicate_metadata_sample_ids"], ["S4"])
        self.assertFalse(report["ordering_matches"])

    def test_select_balanced_sample_ids_round_robins_label_groups(self):
        metadata = pd.DataFrame(
            {
                "sample_id": ["A1", "A2", "A3", "B1", "B2", "C1"],
                "biome_1": ["A", "A", "A", "B", "B", "C"],
            }
        )

        selected = select_balanced_sample_ids(metadata, label_column="biome_1", max_samples=5, seed=0)

        self.assertEqual(len(selected), 5)
        self.assertEqual(set(selected), {"A1", "A2", "B1", "B2", "C1"})
        selected_labels = metadata.set_index("sample_id").loc[selected, "biome_1"]
        self.assertEqual(selected_labels.value_counts().to_dict(), {"A": 2, "B": 2, "C": 1})

    def test_task_suitability_is_conservative_for_mgnify_biome_only_metadata(self):
        metadata = pd.DataFrame(
            {
                "sample_id": ["S1", "S2"],
                "biome_1": ["Host-associated", "Environmental"],
                "biome_2": ["Human", "Aquatic"],
            }
        )

        suitability = task_suitability(metadata, has_corpus=True, has_unnorm_data=True)

        self.assertEqual(suitability["MGM teacher embedding extraction"]["support"], "yes")
        self.assertEqual(suitability["taxa reconstruction target"]["support"], "yes")
        self.assertEqual(suitability["biome/source classification"]["support"], "yes")
        self.assertEqual(suitability["disease prediction"]["support"], "no")
        self.assertEqual(suitability["pathway reconstruction"]["support"], "no")

    def test_summarize_metadata_reports_columns_and_task_labels(self):
        metadata = pd.DataFrame(
            {
                "Unnamed: 0": ["S1", "S2"],
                "1": ["Host-associated", "Environmental"],
                "2": ["Human", "Aquatic"],
            }
        )

        summary = summarize_metadata(metadata)

        self.assertEqual(summary["shape"], [2, 3])
        self.assertEqual(summary["columns"][0]["name"], "sample_id")
        self.assertEqual(summary["columns"][1]["name"], "biome_1")
        self.assertFalse(summary["detected_fields"]["disease_or_phenotype"])
        self.assertFalse(summary["detected_fields"]["pathway"])

    def test_write_pilot_dataset_outputs_aligned_npz_and_tables(self):
        metadata = pd.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3"],
                "biome_1": ["A", "B", "A"],
                "biome_2": ["AA", "BB", "AA"],
            }
        )
        teacher_ids = ["S1", "S2", "S3"]
        taxa_names = ["g__A", "g__B"]
        x_unnorm = pd.DataFrame(
            [[0.7, 0.3], [0.1, 0.9], [0.4, 0.6]],
            index=teacher_ids,
            columns=taxa_names,
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_pilot_dataset(
                sample_ids=["S3", "S1"],
                metadata=metadata,
                teacher_ids=teacher_ids,
                x_unnorm=x_unnorm,
                output_dir=Path(tmp),
            )

            self.assertEqual(manifest["num_samples"], 2)
            self.assertEqual(manifest["num_taxa"], 2)
            sample_ids = pd.read_csv(Path(tmp) / "sample_ids.csv")["sample_id"].tolist()
            self.assertEqual(sample_ids, ["S3", "S1"])
            with np.load(Path(tmp) / "X_unnorm.npz", allow_pickle=False) as payload:
                self.assertEqual(payload["X"].shape, (2, 2))
                self.assertTrue(np.allclose(payload["X"].sum(axis=1), 1.0))

    def test_write_pilot_dataset_outputs_splits_top_taxa_and_retained_mass(self):
        metadata = pd.DataFrame(
            {
                "sample_id": [f"S{i}" for i in range(1, 9)],
                "biome_1": ["A", "A", "A", "A", "B", "B", "B", "B"],
            }
        )
        teacher_ids = metadata["sample_id"].tolist()
        x_unnorm = pd.DataFrame(
            [
                [0.8, 0.2, 0.0, 0.0],
                [0.7, 0.3, 0.0, 0.0],
                [0.6, 0.0, 0.4, 0.0],
                [0.5, 0.0, 0.5, 0.0],
                [0.0, 0.1, 0.8, 0.1],
                [0.0, 0.2, 0.6, 0.2],
                [0.0, 0.6, 0.0, 0.4],
                [0.0, 0.7, 0.0, 0.3],
            ],
            index=teacher_ids,
            columns=["g__A", "g__B", "g__C", "g__D"],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_pilot_dataset(
                sample_ids=teacher_ids,
                metadata=metadata,
                teacher_ids=teacher_ids,
                x_unnorm=x_unnorm,
                output_dir=Path(tmp),
                num_taxa=2,
                split_label_column="biome_1",
                seed=0,
            )

            self.assertEqual(manifest["num_taxa"], 2)
            self.assertIn("split_counts", manifest)
            self.assertTrue((Path(tmp) / "sample_splits.csv").exists())
            self.assertTrue((Path(tmp) / "taxa_filter.csv").exists())
            with np.load(Path(tmp) / "X_taxa.npz", allow_pickle=False) as payload:
                self.assertEqual(payload["X"].shape, (8, 2))
                self.assertTrue(np.allclose(payload["X"].sum(axis=1), 1.0))
                self.assertTrue(np.all(payload["retained_mass"] > 0.0))
            taxa_filter = pd.read_csv(Path(tmp) / "taxa_filter.csv")
            self.assertEqual(set(taxa_filter["taxon"].tolist()), {"g__A", "g__B"})

    def test_write_pilot_dataset_drops_zero_retained_samples_after_taxa_filtering(self):
        metadata = pd.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3", "S4"],
                "biome_1": ["A", "A", "B", "B"],
            }
        )
        x_unnorm = pd.DataFrame(
            [
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 0.0, 1.0],
                [0.7, 0.3, 0.0],
            ],
            index=["S1", "S2", "S3", "S4"],
            columns=["g__A", "g__B", "g__C"],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_pilot_dataset(
                sample_ids=["S1", "S2", "S3", "S4"],
                metadata=metadata,
                teacher_ids=["S1", "S2", "S3", "S4"],
                x_unnorm=x_unnorm,
                output_dir=Path(tmp),
                num_taxa=2,
                split_label_column="biome_1",
                seed=0,
            )

            self.assertEqual(manifest["num_samples"], 3)
            self.assertEqual(manifest["dropped_zero_retained_samples"], 1)
            self.assertNotIn("S3", pd.read_csv(Path(tmp) / "sample_ids.csv")["sample_id"].tolist())
            with np.load(Path(tmp) / "X_taxa.npz", allow_pickle=False) as payload:
                self.assertTrue(np.all(payload["retained_mass"] > 0.0))
                self.assertTrue(np.allclose(payload["X"].sum(axis=1), 1.0))


class CompassBiomeDatasetTransformTests(unittest.TestCase):
    def test_stratified_split_sample_ids_preserves_ids_and_label_balance(self):
        sample_ids = [f"S{i}" for i in range(12)]
        metadata = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "biome_1": ["A"] * 6 + ["B"] * 6,
            }
        )

        splits = stratified_split_sample_ids(
            metadata,
            sample_ids=sample_ids,
            label_column="biome_1",
            seed=0,
        )

        self.assertEqual(splits["sample_id"].tolist(), sample_ids)
        self.assertEqual(set(splits["split"]), {"train", "valid", "test"})
        counts = splits.groupby(["biome_1", "split"]).size().unstack(fill_value=0)
        self.assertTrue((counts["train"] >= 4).all())
        self.assertTrue((counts["valid"] >= 1).all())
        self.assertTrue((counts["test"] >= 1).all())

    def test_select_top_taxa_by_train_uses_train_split_only(self):
        x = np.array(
            [
                [1.0, 0.5, 0.0],
                [0.8, 0.4, 0.0],
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 9.0],
            ],
            dtype=np.float32,
        )
        splits = ["train", "train", "test", "test"]

        indices, table = select_top_taxa_by_train(x, splits, ["g__A", "g__B", "g__C"], num_taxa=2)

        self.assertEqual(indices.tolist(), [0, 1])
        self.assertEqual(table["taxon"].tolist(), ["g__A", "g__B"])

    def test_filter_and_normalize_taxa_reports_retained_mass(self):
        x = np.array([[0.2, 0.3, 0.5], [0.0, 0.4, 0.6]], dtype=np.float32)

        filtered, retained = filter_and_normalize_taxa(x, np.array([0, 1]))

        self.assertTrue(np.allclose(retained, [0.5, 0.4]))
        self.assertTrue(np.allclose(filtered.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(filtered[0], [0.4, 0.6]))


class CompassBiomeReadoutTests(unittest.TestCase):
    def test_top_taxa_rows_reports_highest_program_weights(self):
        programs = np.array(
            [
                [0.1, 0.8, 0.1],
                [0.5, 0.2, 0.3],
            ],
            dtype=np.float32,
        )

        rows = top_taxa_rows(programs, taxa_names=["g__A", "g__B", "g__C"], top_k=2)

        self.assertEqual(
            rows,
            [
                {"program": 0, "rank": 1, "taxon": "g__B", "weight": 0.8},
                {"program": 0, "rank": 2, "taxon": "g__A", "weight": 0.1},
                {"program": 1, "rank": 1, "taxon": "g__A", "weight": 0.5},
                {"program": 1, "rank": 2, "taxon": "g__C", "weight": 0.3},
            ],
        )

    def test_biome_enrichment_rows_reports_mean_activation_by_label(self):
        activations = np.array(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.2, 0.8],
            ],
            dtype=np.float32,
        )
        metadata = pd.DataFrame({"sample_id": ["S1", "S2", "S3"], "biome_1": ["A", "A", "B"]})

        rows = biome_enrichment_rows(activations, metadata, label_column="biome_1")

        self.assertEqual(rows[0]["program"], 0)
        self.assertEqual(rows[0]["label"], "A")
        self.assertAlmostEqual(rows[0]["mean_activation"], 0.8, places=6)
        self.assertEqual(rows[1]["program"], 0)
        self.assertEqual(rows[1]["label"], "B")
        self.assertAlmostEqual(rows[1]["mean_activation"], 0.2, places=6)

    def test_top_k_recall_and_bray_curtis_similarity_score_reconstructions(self):
        true = np.array([[0.6, 0.3, 0.1], [0.0, 0.8, 0.2]], dtype=np.float32)
        pred = np.array([[0.5, 0.4, 0.1], [0.7, 0.2, 0.1]], dtype=np.float32)

        self.assertAlmostEqual(top_k_recall(true, pred, k=2), 0.75)
        self.assertAlmostEqual(bray_curtis_similarity(true, pred), 0.6, places=6)

    def test_reconstruction_metric_rows_include_controls_and_splits(self):
        true = np.array(
            [
                [0.8, 0.2, 0.0],
                [0.7, 0.3, 0.0],
                [0.0, 0.2, 0.8],
                [0.0, 0.1, 0.9],
            ],
            dtype=np.float32,
        )
        pred = true.copy()
        splits = ["train", "train", "test", "test"]

        rows = reconstruction_metric_rows(
            true,
            pred,
            splits,
            model_name="real",
            top_k=2,
        )

        self.assertEqual([row["model"] for row in rows], ["real", "real"])
        self.assertEqual([row["split"] for row in rows], ["train", "test"])
        self.assertTrue(all(row["top_2_recall"] == 1.0 for row in rows))
        self.assertTrue(all(row["bray_curtis_similarity"] == 1.0 for row in rows))

    def test_mean_baseline_predictions_use_train_distribution(self):
        x = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]], dtype=np.float32)
        splits = ["train", "train", "test"]

        pred = mean_baseline_predictions(x, splits)

        self.assertEqual(pred.shape, x.shape)
        self.assertTrue(np.allclose(pred[2], [0.7, 0.3]))

    def test_program_diagnostics_rows_report_collapse_signals(self):
        activations = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.7, 0.3, 0.0],
            ],
            dtype=np.float32,
        )
        taxa_programs = np.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        )

        rows = program_diagnostics_rows(activations, taxa_programs, dead_threshold=0.05)
        metrics = {row["metric"]: row["value"] for row in rows if row["program"] == "all"}

        self.assertAlmostEqual(metrics["dead_program_fraction"], 1 / 3, places=6)
        self.assertIn("effective_programs_mean", metrics)


class CompassBiomeEmbeddingTests(unittest.TestCase):
    def test_resolve_sample_positions_finds_ordered_indices(self):
        index = pd.Index(["S1", "S2", "S3"])

        positions = resolve_sample_positions(["S3", "S1"], index)

        self.assertEqual(positions, [2, 0])

    def test_last_valid_token_embeddings_uses_attention_mask_lengths(self):
        import torch

        hidden = torch.tensor(
            [
                [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]],
                [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
            ]
        )
        mask = torch.tensor(
            [
                [1, 1, 0],
                [1, 1, 1],
            ]
        )

        embeddings = last_valid_token_embeddings(hidden, mask)

        self.assertEqual(embeddings.tolist(), [[2.0, 2.0], [5.0, 5.0]])


class CompassBiomeBottleneckTests(unittest.TestCase):
    def test_train_bottleneck_model_learns_shapes_and_decreases_loss(self):
        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(12, 6)).astype(np.float32)
        taxa_targets = rng.random(size=(12, 5)).astype(np.float32)
        taxa_targets = taxa_targets / taxa_targets.sum(axis=1, keepdims=True)

        result = train_bottleneck_model(
            embeddings=embeddings,
            taxa_targets=taxa_targets,
            num_programs=3,
            epochs=25,
            batch_size=6,
            learning_rate=0.05,
            seed=0,
            device_name="cpu",
        )

        self.assertEqual(result["activations"].shape, (12, 3))
        self.assertEqual(result["taxa_programs"].shape, (3, 5))
        self.assertTrue(np.all(np.isfinite(result["activations"])))
        self.assertTrue(np.allclose(result["activations"].sum(axis=1), 1.0, atol=1e-5))
        self.assertTrue(np.allclose(result["taxa_programs"].sum(axis=1), 1.0, atol=1e-5))
        self.assertLess(result["metrics"][-1]["loss"], result["metrics"][0]["loss"])

    def test_run_pilot_writes_split_metrics_controls_and_diagnostics(self):
        sample_ids = [f"S{i}" for i in range(6)]
        x = np.array(
            [
                [0.8, 0.2, 0.0],
                [0.7, 0.3, 0.0],
                [0.1, 0.8, 0.1],
                [0.2, 0.7, 0.1],
                [0.0, 0.2, 0.8],
                [0.0, 0.1, 0.9],
            ],
            dtype=np.float32,
        )
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 0.9],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            pd.DataFrame({"sample_id": sample_ids}).to_csv(dataset_dir / "sample_ids.csv", index=False)
            pd.DataFrame({"taxon": ["g__A", "g__B", "g__C"]}).to_csv(dataset_dir / "taxa_names.csv", index=False)
            pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "biome_1": ["A", "A", "B", "B", "C", "C"],
                }
            ).to_csv(dataset_dir / "metadata.csv", index=False)
            pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "biome_1": ["A", "A", "B", "B", "C", "C"],
                    "split": ["train", "train", "train", "train", "valid", "test"],
                }
            ).to_csv(dataset_dir / "sample_splits.csv", index=False)
            np.savez_compressed(
                dataset_dir / "X_taxa.npz",
                sample_ids=np.array(sample_ids),
                taxa_names=np.array(["g__A", "g__B", "g__C"]),
                X=x,
                retained_mass=np.ones(len(sample_ids), dtype=np.float32),
            )
            embeddings_path = root / "embeddings.npz"
            np.savez_compressed(
                embeddings_path,
                sample_ids=np.array(sample_ids),
                embeddings=embeddings,
            )

            manifest = run_pilot(
                dataset_dir=dataset_dir,
                embeddings_path=embeddings_path,
                output_dir=root / "bottleneck",
                num_programs=2,
                epochs=5,
                batch_size=3,
                learning_rate=0.05,
                seed=0,
                device_name="cpu",
                top_k=2,
                label_column="biome_1",
                controls="all",
                patience=None,
                make_plots=False,
            )

            metrics = pd.read_csv(root / "bottleneck" / "reconstruction_metrics.csv")
            self.assertEqual(set(metrics["model"]), {"real", "shuffle", "mean_baseline"})
            self.assertEqual(set(metrics["split"]), {"train", "valid", "test"})
            self.assertTrue((root / "bottleneck" / "program_diagnostics.csv").exists())
            self.assertTrue((root / "bottleneck" / "top_taxa_per_program.csv").exists())
            self.assertEqual(manifest["controls"], "all")


if __name__ == "__main__":
    unittest.main()
