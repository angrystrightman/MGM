import json
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
from scripts.fit_compass_nmf_dictionary import fit_nmf_dictionary
from scripts.run_compass_bottleneck_pilot import (
    dictionary_diversity_loss,
    run_pilot,
    sample_entropy_target_loss,
    train_bottleneck_model,
    usage_balance_loss,
)
from scripts.summarize_compass_nmf_warm_start import summarize_nmf_warm_start_runs
from scripts.summarize_compass_k_resolution import (
    recommend_k_resolution,
    summarize_k_resolution_runs,
)


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
        self.assertIn("dictionary_cosine_max_offdiag", metrics)
        self.assertIn("dictionary_cosine_p90_offdiag", metrics)
        self.assertIn("dictionary_cosine_p95_offdiag", metrics)
        self.assertIn("dictionary_top20_overlap_mean", metrics)
        self.assertIn("dictionary_top20_overlap_max", metrics)
        self.assertIn("dictionary_top20_jaccard_mean", metrics)
        self.assertGreaterEqual(metrics["dictionary_cosine_max_offdiag"], metrics["dictionary_cosine_mean_offdiag"])


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
    def test_dictionary_diversity_loss_penalizes_similar_programs_more_than_dissimilar_programs(self):
        import torch

        similar = torch.tensor(
            [
                [0.70, 0.30, 0.00],
                [0.69, 0.31, 0.00],
                [0.00, 0.00, 1.00],
            ],
            dtype=torch.float32,
        )
        dissimilar = torch.eye(3, dtype=torch.float32)

        similar_loss = dictionary_diversity_loss(
            similar,
            mode="hinge_cosine",
            threshold=0.30,
        )
        dissimilar_loss = dictionary_diversity_loss(
            dissimilar,
            mode="hinge_cosine",
            threshold=0.30,
        )
        similar_mse = dictionary_diversity_loss(similar, mode="mse_offdiag")
        dissimilar_mse = dictionary_diversity_loss(dissimilar, mode="mse_offdiag")

        self.assertGreater(float(similar_loss), float(dissimilar_loss))
        self.assertAlmostEqual(float(dissimilar_loss), 0.0, places=6)
        self.assertGreater(float(similar_mse), float(dissimilar_mse))
        self.assertAlmostEqual(float(dissimilar_mse), 0.0, places=6)

    def test_dictionary_diversity_hinge_ignores_cosine_below_threshold(self):
        import torch

        dictionaries = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.2, 0.0, 0.98],
            ],
            dtype=torch.float32,
        )

        loss = dictionary_diversity_loss(
            dictionaries,
            mode="hinge_cosine",
            threshold=0.30,
        )

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_usage_balance_loss_penalizes_collapsed_usage_more_than_uniform_usage(self):
        import torch

        collapsed = torch.tensor(
            [
                [0.95, 0.05, 0.00, 0.00],
                [0.90, 0.10, 0.00, 0.00],
                [0.92, 0.08, 0.00, 0.00],
            ],
            dtype=torch.float32,
        )
        balanced = torch.full((3, 4), 0.25, dtype=torch.float32)

        collapsed_loss = usage_balance_loss(collapsed, mode="kl_uniform_to_usage")
        balanced_loss = usage_balance_loss(balanced, mode="kl_uniform_to_usage")
        collapsed_mse = usage_balance_loss(collapsed, mode="mse")
        balanced_mse = usage_balance_loss(balanced, mode="mse")

        self.assertGreater(float(collapsed_loss), float(balanced_loss) + 1.0)
        self.assertAlmostEqual(float(balanced_loss), 0.0, places=6)
        self.assertGreater(float(collapsed_mse), float(balanced_mse))
        self.assertAlmostEqual(float(balanced_mse), 0.0, places=6)

    def test_sample_entropy_target_loss_prefers_target_effective_program_count(self):
        import torch

        one_hot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        near_two_programs = torch.tensor([[0.5, 0.5, 0.0, 0.0]], dtype=torch.float32)
        fully_uniform = torch.full((1, 4), 0.25, dtype=torch.float32)

        one_hot_loss = sample_entropy_target_loss(one_hot, target_effective_programs=2.0)
        target_loss = sample_entropy_target_loss(near_two_programs, target_effective_programs=2.0)
        uniform_loss = sample_entropy_target_loss(fully_uniform, target_effective_programs=2.0)

        self.assertAlmostEqual(float(target_loss), 0.0, places=6)
        self.assertGreater(float(one_hot_loss), float(target_loss))
        self.assertGreater(float(uniform_loss), float(target_loss))

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

    def test_train_bottleneck_model_reports_regularization_metrics_when_enabled(self):
        rng = np.random.default_rng(1)
        embeddings = rng.normal(size=(16, 5)).astype(np.float32)
        taxa_targets = rng.random(size=(16, 4)).astype(np.float32)
        taxa_targets = taxa_targets / taxa_targets.sum(axis=1, keepdims=True)

        result = train_bottleneck_model(
            embeddings=embeddings,
            taxa_targets=taxa_targets,
            num_programs=4,
            epochs=4,
            batch_size=8,
            learning_rate=0.03,
            seed=0,
            device_name="cpu",
            usage_balance_weight=0.05,
            usage_balance_mode="kl_uniform_to_usage",
            sample_entropy_weight=0.10,
            sample_entropy_target_effective=2.0,
            dictionary_diversity_weight=0.05,
            dictionary_diversity_mode="hinge_cosine",
            dictionary_diversity_threshold=0.30,
        )

        final_metrics = result["metrics"][-1]
        for key in [
            "reconstruction_loss",
            "regularized_loss",
            "usage_balance_loss",
            "sample_entropy_mean",
            "sample_entropy_target_loss",
            "dictionary_diversity_loss",
        ]:
            self.assertIn(key, final_metrics)
            self.assertTrue(np.isfinite(final_metrics[key]))
        self.assertGreaterEqual(final_metrics["regularized_loss"], final_metrics["reconstruction_loss"])

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
                usage_balance_weight=0.05,
                usage_balance_mode="kl_uniform_to_usage",
                sample_entropy_weight=0.10,
                sample_entropy_target_effective=2.0,
                dictionary_diversity_weight=0.02,
                dictionary_diversity_mode="hinge_cosine",
                dictionary_diversity_threshold=0.25,
            )

            metrics = pd.read_csv(root / "bottleneck" / "reconstruction_metrics.csv")
            self.assertEqual(set(metrics["model"]), {"real", "shuffle", "mean_baseline"})
            self.assertEqual(set(metrics["split"]), {"train", "valid", "test"})
            self.assertTrue((root / "bottleneck" / "program_diagnostics.csv").exists())
            self.assertTrue((root / "bottleneck" / "top_taxa_per_program.csv").exists())
            self.assertEqual(manifest["controls"], "all")
            self.assertEqual(manifest["usage_balance_weight"], 0.05)
            self.assertEqual(manifest["usage_balance_loss"], "kl_uniform_to_usage")
            self.assertEqual(manifest["sample_entropy_weight"], 0.10)
            self.assertEqual(manifest["sample_entropy_target_effective"], 2.0)
            self.assertEqual(manifest["dictionary_diversity_weight"], 0.02)
            self.assertEqual(manifest["dictionary_diversity_loss_type"], "hinge_cosine")
            self.assertEqual(manifest["dictionary_diversity_threshold"], 0.25)
            training_metrics = pd.read_csv(root / "bottleneck" / "training_metrics.csv")
            self.assertIn("regularized_loss", training_metrics.columns)
            self.assertIn("usage_balance_loss", training_metrics.columns)
            self.assertIn("dictionary_diversity_loss", training_metrics.columns)


class CompassBiomeNmfWarmStartTests(unittest.TestCase):
    def _write_small_dataset(self, root: Path) -> tuple[Path, Path]:
        sample_ids = [f"S{i}" for i in range(8)]
        x = np.array(
            [
                [0.80, 0.15, 0.03, 0.02],
                [0.75, 0.20, 0.03, 0.02],
                [0.05, 0.85, 0.08, 0.02],
                [0.04, 0.80, 0.12, 0.04],
                [0.02, 0.06, 0.84, 0.08],
                [0.03, 0.05, 0.80, 0.12],
                [0.10, 0.10, 0.10, 0.70],
                [0.12, 0.08, 0.10, 0.70],
            ],
            dtype=np.float32,
        )
        dataset_dir = root / "dataset"
        dataset_dir.mkdir()
        pd.DataFrame({"sample_id": sample_ids}).to_csv(dataset_dir / "sample_ids.csv", index=False)
        pd.DataFrame({"taxon": ["g__A", "g__B", "g__C", "g__D"]}).to_csv(dataset_dir / "taxa_names.csv", index=False)
        pd.DataFrame({"sample_id": sample_ids, "biome_1": ["A", "A", "B", "B", "C", "C", "D", "D"]}).to_csv(
            dataset_dir / "metadata.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "sample_id": sample_ids,
                "biome_1": ["A", "A", "B", "B", "C", "C", "D", "D"],
                "split": ["train", "train", "train", "train", "valid", "valid", "test", "test"],
            }
        ).to_csv(dataset_dir / "sample_splits.csv", index=False)
        np.savez_compressed(
            dataset_dir / "X_taxa.npz",
            sample_ids=np.array(sample_ids),
            taxa_names=np.array(["g__A", "g__B", "g__C", "g__D"]),
            X=x,
            retained_mass=np.ones(len(sample_ids), dtype=np.float32),
        )
        embeddings = np.eye(8, 4, dtype=np.float32)
        embeddings_path = root / "embeddings.npz"
        np.savez_compressed(embeddings_path, sample_ids=np.array(sample_ids), embeddings=embeddings)
        return dataset_dir, embeddings_path

    def test_fit_nmf_dictionary_uses_train_split_and_writes_normalized_dictionary(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir, _ = self._write_small_dataset(Path(tmp))

            manifest = fit_nmf_dictionary(
                dataset_dir=dataset_dir,
                output_dir=Path(tmp) / "nmf",
                num_programs=2,
                max_iter=30,
                seed=0,
            )

            self.assertEqual(manifest["num_train_samples"], 4)
            self.assertEqual(manifest["num_samples"], 8)
            self.assertEqual(manifest["num_taxa"], 4)
            self.assertEqual(manifest["num_programs"], 2)
            with np.load(Path(tmp) / "nmf" / "nmf_dictionary.npz", allow_pickle=False) as payload:
                dictionary = payload["taxa_programs"]
                self.assertEqual(dictionary.shape, (2, 4))
                self.assertTrue(np.all(np.isfinite(dictionary)))
                self.assertTrue(np.all(dictionary >= 0.0))
                self.assertTrue(np.allclose(dictionary.sum(axis=1), 1.0, atol=1e-5))
            metrics = pd.read_csv(Path(tmp) / "nmf" / "nmf_oracle_metrics.csv")
            self.assertEqual(set(metrics["model"]), {"nmf_oracle"})
            self.assertTrue({"train", "valid", "test"}.issubset(set(metrics["split"])))
            self.assertTrue((Path(tmp) / "nmf" / "top_taxa_per_program.csv").exists())

    def test_train_bottleneck_model_fixed_taxa_dictionary_preserves_dictionary(self):
        rng = np.random.default_rng(2)
        embeddings = rng.normal(size=(12, 5)).astype(np.float32)
        taxa_targets = rng.random(size=(12, 4)).astype(np.float32)
        taxa_targets = taxa_targets / taxa_targets.sum(axis=1, keepdims=True)
        dictionary_init = np.array(
            [
                [0.70, 0.20, 0.05, 0.05],
                [0.10, 0.70, 0.10, 0.10],
                [0.05, 0.05, 0.80, 0.10],
            ],
            dtype=np.float32,
        )

        result = train_bottleneck_model(
            embeddings=embeddings,
            taxa_targets=taxa_targets,
            num_programs=3,
            epochs=4,
            batch_size=6,
            learning_rate=0.05,
            seed=0,
            device_name="cpu",
            taxa_dictionary_init=dictionary_init,
            freeze_taxa_dictionary=True,
        )

        expected = dictionary_init / dictionary_init.sum(axis=1, keepdims=True)
        self.assertTrue(np.allclose(result["taxa_programs"], expected, atol=1e-6))

    def test_train_bottleneck_model_anchor_reports_anchor_loss(self):
        rng = np.random.default_rng(3)
        embeddings = rng.normal(size=(12, 5)).astype(np.float32)
        taxa_targets = rng.random(size=(12, 4)).astype(np.float32)
        taxa_targets = taxa_targets / taxa_targets.sum(axis=1, keepdims=True)
        dictionary_init = np.full((3, 4), 0.25, dtype=np.float32)

        result = train_bottleneck_model(
            embeddings=embeddings,
            taxa_targets=taxa_targets,
            num_programs=3,
            epochs=4,
            batch_size=6,
            learning_rate=0.05,
            seed=0,
            device_name="cpu",
            taxa_dictionary_init=dictionary_init,
            dictionary_anchor_weight=0.05,
            dictionary_anchor_mode="kl_anchor_to_current",
        )

        final_metrics = result["metrics"][-1]
        self.assertIn("dictionary_anchor_loss", final_metrics)
        self.assertIn("batch_dictionary_anchor_loss_mean", final_metrics)
        self.assertTrue(np.isfinite(final_metrics["dictionary_anchor_loss"]))

    def test_run_pilot_records_taxa_dictionary_init_manifest_and_training_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir, embeddings_path = self._write_small_dataset(root)
            dictionary_init = np.array(
                [
                    [0.70, 0.20, 0.05, 0.05],
                    [0.10, 0.70, 0.10, 0.10],
                ],
                dtype=np.float32,
            )
            dictionary_path = root / "dictionary.npz"
            np.savez_compressed(dictionary_path, taxa_programs=dictionary_init)

            manifest = run_pilot(
                dataset_dir=dataset_dir,
                embeddings_path=embeddings_path,
                output_dir=root / "bottleneck",
                num_programs=2,
                epochs=4,
                batch_size=4,
                learning_rate=0.05,
                seed=0,
                device_name="cpu",
                top_k=2,
                label_column="biome_1",
                controls="none",
                patience=None,
                make_plots=False,
                taxa_dictionary_init_path=dictionary_path,
                freeze_taxa_dictionary=True,
                dictionary_anchor_weight=0.05,
                dictionary_anchor_mode="kl_anchor_to_current",
            )

            self.assertEqual(manifest["taxa_dictionary_init"], str(dictionary_path))
            self.assertTrue(manifest["freeze_taxa_dictionary"])
            self.assertEqual(manifest["dictionary_anchor_weight"], 0.05)
            self.assertEqual(manifest["dictionary_anchor_loss"], "kl_anchor_to_current")
            training_metrics = pd.read_csv(root / "bottleneck" / "training_metrics.csv")
            self.assertIn("dictionary_anchor_loss", training_metrics.columns)
            with np.load(root / "bottleneck" / "bottleneck_outputs.npz", allow_pickle=False) as payload:
                expected = dictionary_init / dictionary_init.sum(axis=1, keepdims=True)
                self.assertTrue(np.allclose(payload["taxa_programs"], expected, atol=1e-6))


class CompassBiomeNmfWarmStartSummaryTests(unittest.TestCase):
    def _write_metrics(self, path: Path, real_top20: float, real_bray: float) -> None:
        pd.DataFrame(
            [
                {
                    "model": "real",
                    "split": "test",
                    "num_samples": 4,
                    "cross_entropy": 3.5,
                    "top_20_recall": real_top20,
                    "bray_curtis_similarity": real_bray,
                },
                {
                    "model": "mean_baseline",
                    "split": "test",
                    "num_samples": 4,
                    "cross_entropy": 5.0,
                    "top_20_recall": 0.20,
                    "bray_curtis_similarity": 0.12,
                },
                {
                    "model": "shuffle",
                    "split": "test",
                    "num_samples": 4,
                    "cross_entropy": 5.1,
                    "top_20_recall": 0.19,
                    "bray_curtis_similarity": 0.13,
                },
            ]
        ).to_csv(path, index=False)

    def _write_bottleneck_run(
        self,
        output_dir: Path,
        label: str,
        taxa_programs: np.ndarray,
        real_top20: float,
        real_bray: float,
        freeze: bool = False,
        anchor_weight: float = 0.0,
    ) -> None:
        output_dir.mkdir(parents=True)
        activations = np.full((4, taxa_programs.shape[0]), 1.0 / taxa_programs.shape[0], dtype=np.float32)
        reconstructions = np.full((4, taxa_programs.shape[1]), 1.0 / taxa_programs.shape[1], dtype=np.float32)
        np.savez_compressed(
            output_dir / "bottleneck_outputs.npz",
            sample_ids=np.array(["S1", "S2", "S3", "S4"]),
            activations=activations,
            taxa_programs=taxa_programs.astype(np.float32),
            reconstructions=reconstructions,
            splits=np.array(["train", "train", "valid", "test"]),
        )
        self._write_metrics(output_dir / "reconstruction_metrics.csv", real_top20=real_top20, real_bray=real_bray)
        pd.DataFrame(program_diagnostics_rows(activations, taxa_programs)).to_csv(
            output_dir / "program_diagnostics.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "epoch": 1,
                    "regularized_loss": 3.5,
                    "reconstruction_loss": 3.4,
                    "dictionary_diversity_loss": 0.0,
                    "dictionary_anchor_loss": 0.01,
                    "sample_entropy_mean": 1.0,
                    "usage_balance_loss": 0.0,
                }
            ]
        ).to_csv(output_dir / "training_metrics.csv", index=False)
        manifest = {
            "num_samples": 4,
            "num_taxa": int(taxa_programs.shape[1]),
            "embedding_width": 3,
            "num_programs": int(taxa_programs.shape[0]),
            "epochs_ran": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "usage_balance_weight": 0.05,
            "sample_entropy_weight": 0.10,
            "dictionary_diversity_weight": 0.02,
            "dictionary_anchor_weight": anchor_weight,
            "freeze_taxa_dictionary": freeze,
            "taxa_dictionary_init": "dictionary.npz" if label != "parent" else None,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")

    def test_summarize_nmf_warm_start_reports_tradeoff_and_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            init_dictionary = np.array(
                [
                    [0.80, 0.10, 0.05, 0.05],
                    [0.05, 0.80, 0.10, 0.05],
                ],
                dtype=np.float32,
            )
            nmf_dir = root / "nmf"
            nmf_dir.mkdir()
            np.savez_compressed(nmf_dir / "nmf_dictionary.npz", taxa_programs=init_dictionary)
            (nmf_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "num_samples": 4,
                        "num_taxa": 4,
                        "num_programs": 2,
                        "n_iter": 12,
                        "reconstruction_err": 1.2,
                    }
                )
                + "\n"
            )
            pd.DataFrame(
                [
                    {
                        "model": "nmf_oracle",
                        "split": "test",
                        "num_samples": 4,
                        "cross_entropy": 3.0,
                        "top_20_recall": 0.60,
                        "bray_curtis_similarity": 0.55,
                    }
                ]
            ).to_csv(nmf_dir / "nmf_oracle_metrics.csv", index=False)

            parent_dir = root / "parent"
            anchor_dir = root / "anchor"
            self._write_bottleneck_run(
                parent_dir,
                label="parent",
                taxa_programs=init_dictionary,
                real_top20=0.54,
                real_bray=0.44,
            )
            anchor_dictionary = np.array(
                [
                    [0.72, 0.18, 0.05, 0.05],
                    [0.08, 0.72, 0.15, 0.05],
                ],
                dtype=np.float32,
            )
            self._write_bottleneck_run(
                anchor_dir,
                label="anchor",
                taxa_programs=anchor_dictionary,
                real_top20=0.53,
                real_bray=0.46,
                anchor_weight=0.05,
            )

            summary, recommendation = summarize_nmf_warm_start_runs(
                parent_run=("K32_parent", parent_dir),
                nmf_dir=nmf_dir,
                runs=[("K32_nmf_anchor005", anchor_dir)],
                output_summary=root / "summary.csv",
                output_recommendation=root / "recommendation.json",
            )

            self.assertEqual(set(summary["run"]), {"K32_parent", "NMF_oracle", "K32_nmf_anchor005"})
            anchor_row = summary.loc[summary["run"].eq("K32_nmf_anchor005")].iloc[0]
            self.assertGreater(anchor_row["nmf_init_to_final_kl_mean"], 0.0)
            self.assertEqual(recommendation["verdict"], "tradeoff")
            self.assertEqual(recommendation["recommended_run"], "K32_nmf_anchor005")
            self.assertTrue((root / "summary.csv").exists())
            self.assertTrue((root / "recommendation.json").exists())


class CompassBiomeKResolutionSummaryTests(unittest.TestCase):
    def _write_synthetic_bottleneck_run(
        self,
        output_dir: Path,
        label: str,
        num_programs: int,
        real_top20: float,
        real_bray: float,
        shuffle_top20: float = 0.20,
        shuffle_bray: float = 0.12,
        dead_fraction_hint: float = 0.0,
    ) -> None:
        output_dir.mkdir(parents=True)
        sample_ids = np.array(["S1", "S2", "S3", "S4"])
        splits = np.array(["train", "train", "valid", "test"])
        activations = np.full((4, num_programs), 1.0 / num_programs, dtype=np.float32)
        if dead_fraction_hint > 0:
            dead_count = max(1, int(round(num_programs * dead_fraction_hint)))
            activations[:, -dead_count:] = 0.0
            activations = activations / activations.sum(axis=1, keepdims=True)
        taxa_programs = np.eye(num_programs, 6, dtype=np.float32)
        taxa_programs = taxa_programs / taxa_programs.sum(axis=1, keepdims=True)
        reconstructions = np.full((4, 6), 1.0 / 6.0, dtype=np.float32)
        np.savez_compressed(
            output_dir / "bottleneck_outputs.npz",
            sample_ids=sample_ids,
            activations=activations,
            taxa_programs=taxa_programs,
            reconstructions=reconstructions,
            splits=splits,
        )
        manifest = {
            "num_samples": 4,
            "num_taxa": 6,
            "embedding_width": 256,
            "num_programs": num_programs,
            "epochs_ran": 7,
            "batch_size": 2048,
            "learning_rate": 0.03,
            "usage_balance_weight": 0.05,
            "sample_entropy_weight": 0.10,
            "sample_entropy_target_effective": 2.0,
            "dictionary_diversity_weight": 0.02,
            "dictionary_diversity_loss_type": "hinge_cosine",
            "dictionary_diversity_threshold": 0.25,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")
        pd.DataFrame(
            [
                {"model": "real", "split": "test", "num_samples": 1, "cross_entropy": 3.0, "top_20_recall": real_top20, "bray_curtis_similarity": real_bray},
                {"model": "mean_baseline", "split": "test", "num_samples": 1, "cross_entropy": 5.0, "top_20_recall": 0.18, "bray_curtis_similarity": 0.11},
                {"model": "shuffle", "split": "test", "num_samples": 1, "cross_entropy": 5.1, "top_20_recall": shuffle_top20, "bray_curtis_similarity": shuffle_bray},
            ]
        ).to_csv(output_dir / "reconstruction_metrics.csv", index=False)
        pd.DataFrame(
            {
                "epoch": [1.0, 7.0],
                "regularized_loss": [4.0, 3.0],
                "reconstruction_loss": [3.9, 2.9],
                "dictionary_diversity_loss": [0.05, 0.01],
            }
        ).to_csv(output_dir / "training_metrics.csv", index=False)
        pd.DataFrame(
            [
                {"program": program, "rank": 1, "taxon": f"g__T{program}", "weight": 0.9}
                for program in range(num_programs)
            ]
        ).to_csv(output_dir / "top_taxa_per_program.csv", index=False)
        pd.DataFrame(
            [
                {"program": program, "label": f"Biome{program % 2}", "mean_activation": 0.4 + program * 0.01, "num_samples": 2}
                for program in range(num_programs)
            ]
        ).to_csv(output_dir / "biome_enrichment.csv", index=False)

    def test_summarize_k_resolution_runs_recomputes_diagnostics_and_program_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "K4" / "bottleneck"
            self._write_synthetic_bottleneck_run(
                run_dir,
                label="K4",
                num_programs=4,
                real_top20=0.52,
                real_bray=0.42,
            )

            summary, program_report, recommendation = summarize_k_resolution_runs(
                {"K4": run_dir},
                output_summary=Path(tmp) / "summary.csv",
                output_program_report=Path(tmp) / "program_report.csv",
                output_recommendation=Path(tmp) / "recommendation.json",
            )

            self.assertEqual(summary.loc[0, "run"], "K4")
            self.assertEqual(summary.loc[0, "num_programs"], 4)
            self.assertAlmostEqual(summary.loc[0, "test_real_top20_recall"], 0.52)
            self.assertIn("dictionary_cosine_max_offdiag", summary.columns)
            self.assertEqual(program_report.loc[0, "run"], "K4")
            self.assertIn("top_biome_label", program_report.columns)
            self.assertEqual(recommendation["primary_k"], 4)
            self.assertTrue((Path(tmp) / "summary.csv").exists())
            self.assertTrue((Path(tmp) / "program_report.csv").exists())
            self.assertTrue((Path(tmp) / "recommendation.json").exists())

    def test_recommend_k_resolution_prefers_smallest_high_fidelity_k_and_gates_k64(self):
        summary = pd.DataFrame(
            [
                {
                    "run": "K16",
                    "num_programs": 16,
                    "test_real_top20_recall": 0.50,
                    "test_real_bray_curtis": 0.40,
                    "test_shuffle_top20_recall": 0.20,
                    "test_shuffle_bray_curtis": 0.12,
                    "dead_program_fraction": 0.10,
                    "dictionary_cosine_p95_offdiag": 0.15,
                },
                {
                    "run": "K32",
                    "num_programs": 32,
                    "test_real_top20_recall": 0.54,
                    "test_real_bray_curtis": 0.44,
                    "test_shuffle_top20_recall": 0.21,
                    "test_shuffle_bray_curtis": 0.13,
                    "dead_program_fraction": 0.12,
                    "dictionary_cosine_p95_offdiag": 0.16,
                },
                {
                    "run": "K64",
                    "num_programs": 64,
                    "test_real_top20_recall": 0.548,
                    "test_real_bray_curtis": 0.443,
                    "test_shuffle_top20_recall": 0.21,
                    "test_shuffle_bray_curtis": 0.13,
                    "dead_program_fraction": 0.20,
                    "dictionary_cosine_p95_offdiag": 0.18,
                },
            ]
        )

        recommendation = recommend_k_resolution(summary)

        self.assertEqual(recommendation["primary_k"], 16)
        self.assertEqual(recommendation["compact_k"], 16)
        self.assertFalse(recommendation["prefer_k64_over_k32"])

    def test_recommend_k_resolution_rejects_weak_control_gap(self):
        summary = pd.DataFrame(
            [
                {
                    "run": "K8",
                    "num_programs": 8,
                    "test_real_top20_recall": 0.25,
                    "test_real_bray_curtis": 0.18,
                    "test_shuffle_top20_recall": 0.20,
                    "test_shuffle_bray_curtis": 0.12,
                    "dead_program_fraction": 0.00,
                    "dictionary_cosine_p95_offdiag": 0.10,
                },
                {
                    "run": "K32",
                    "num_programs": 32,
                    "test_real_top20_recall": 0.54,
                    "test_real_bray_curtis": 0.44,
                    "test_shuffle_top20_recall": 0.21,
                    "test_shuffle_bray_curtis": 0.13,
                    "dead_program_fraction": 0.12,
                    "dictionary_cosine_p95_offdiag": 0.16,
                },
            ]
        )

        recommendation = recommend_k_resolution(summary)

        self.assertEqual(recommendation["primary_k"], 32)
        self.assertEqual(recommendation["compact_k"], 32)
        self.assertIn("K8", recommendation["rejected_runs"])


if __name__ == "__main__":
    unittest.main()
