import unittest
from pathlib import Path


class ScaffoldTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_expected_structure_exists(self):
        expected = [
            "src/cn_spoil/datasets.py",
            "src/cn_spoil/train.py",
            "src/cn_spoil/evaluate.py",
            "src/cn_spoil/inference.py",
            "scripts/train.py",
            "scripts/evaluate.py",
            "scripts/infer.py",
            "configs/train.yaml",
            "configs/eval.yaml",
            "configs/infer.yaml",
            "data/CN-Spoil/README.md",
            "checkpoints/pretrained/README.md",
        ]
        for rel in expected:
            self.assertTrue((self.root / rel).exists(), f"Missing scaffold file: {rel}")


if __name__ == "__main__":
    unittest.main()
