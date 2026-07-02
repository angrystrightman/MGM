import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.check_mgm_resources import verify_required_files
from scripts.extract_mgm_representations import topk_attention_rows
from scripts.prepare_infant_smoke_data import main as prepare_main
from scripts.prepare_infant_smoke_data import parse_delivery_label


class PrepareInfantSmokeDataTests(unittest.TestCase):
    def test_parse_delivery_label_reads_cv_suffix(self):
        self.assertEqual(parse_delivery_label("12M(C)"), "C")
        self.assertEqual(parse_delivery_label("B(V)"), "V")
        with self.assertRaises(ValueError):
            parse_delivery_label("unknown")

    def test_prepare_writes_subset_abundance_labels_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            abundance = root / "abundance.csv"
            metadata = root / "meta.csv"
            output = root / "out"

            with abundance.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Samples", "S1", "S2", "S3", "S4"])
                writer.writerow(["k__Bacteria;p__P;g__A", "1", "2", "3", "4"])
                writer.writerow(["k__Bacteria;p__P;g__B", "0", "1", "0", "1"])

            with metadata.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["SampleID", "Env"])
                writer.writerow(["S1", "B(C)"])
                writer.writerow(["S2", "B(V)"])
                writer.writerow(["S3", "12M(C)"])
                writer.writerow(["S4", "12M(V)"])

            exit_code = prepare_main(
                [
                    "--abundance",
                    str(abundance),
                    "--metadata",
                    str(metadata),
                    "--output-dir",
                    str(output),
                    "--max-samples",
                    "4",
                    "--seed",
                    "7",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "abundance.csv").exists())
            self.assertTrue((output / "labels_env.csv").exists())
            self.assertTrue((output / "labels_delivery_cv.csv").exists())

            with (output / "labels_delivery_cv.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["delivery_cv"] for row in rows}, {"C", "V"})
            self.assertEqual(len(rows), 4)

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["num_selected_samples"], 4)
            self.assertEqual(manifest["delivery_counts"], {"C": 2, "V": 2})


class CheckMgmResourcesTests(unittest.TestCase):
    def test_verify_required_files_reports_expected_resource_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "mgm" / "resources" / "general_model"
            model_dir.mkdir(parents=True)
            (root / "mgm" / "resources" / "MicroTokenizer.pkl").write_bytes(b"tok")
            (model_dir / "config.json").write_text("{}")
            (model_dir / "generation_config.json").write_text("{}")

            report = verify_required_files(root)

            self.assertTrue(report["MicroTokenizer.pkl"]["exists"])
            self.assertFalse(report["general_model/pytorch_model.bin"]["exists"])
            self.assertEqual(report["general_model/config.json"]["size_bytes"], 2)


class ExtractRepresentationsTests(unittest.TestCase):
    def test_topk_attention_rows_sorts_and_filters_special_tokens(self):
        rows = topk_attention_rows(
            sample_id="S1",
            token_names=["<pad>", "g__A", "g__B", "<eos>", "g__C"],
            scores=[0.99, 0.2, 0.5, 0.8, 0.3],
            k=2,
        )

        self.assertEqual(
            rows,
            [
                {"sample_id": "S1", "rank": 1, "token": "g__B", "attention": 0.5},
                {"sample_id": "S1", "rank": 2, "token": "g__C", "attention": 0.3},
            ],
        )


class SmokeRunnerTests(unittest.TestCase):
    def test_smoke_runner_defaults_to_one_visible_gpu(self):
        script_path = Path("scripts/run_infant_smoke.sh")
        script = script_path.read_text()

        self.assertIn('SMOKE_CUDA_VISIBLE_DEVICES="${SMOKE_CUDA_VISIBLE_DEVICES:-0}"', script)
        self.assertIn('export CUDA_VISIBLE_DEVICES="${SMOKE_CUDA_VISIBLE_DEVICES}"', script)

    def test_smoke_runner_has_valid_bash_syntax(self):
        subprocess.run(["bash", "-n", "scripts/run_infant_smoke.sh"], check=True)


if __name__ == "__main__":
    unittest.main()
