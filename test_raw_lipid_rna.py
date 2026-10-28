import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from raw_lipid_rna import (
    as_notebook_frames, detect_export_format, file_hash,
    interpolate_without_extrapolation, load_raw_lipid_rna, read_raw_export,
)


def write_long(path, offset=0):
    lines = []
    for s in range(5):
        for w in (453., 452., 451.):
            lines.append(f"{s}\t0\t{w}\t{offset + s * 10 + w}\n")
    path.write_text("".join(lines))


def write_wide(path, offset=0):
    lines = ["\t\t451\t452\t453\n"]
    for s in range(5):
        lines.append(f"{s}\t1\t{offset+s*10+451}\t{offset+s*10+452}\t{offset+s*10+453}\n")
    path.write_text("".join(lines))


class RawLipidRnaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.entries = []
        for i, instrument in enumerate(("Renishaw", "Horiba")):
            for j, molecule in enumerate(("RNA", "Lipid")):
                p = self.root / f"{instrument}_{molecule}.txt"
                (write_long if i == 0 else write_wide)(p, offset=100 * (1 + 2*i + j))
                self.entries.append({"relative_path": p.name, "instrument": instrument,
                                     "molecule": molecule, "sha256": file_hash(p)})
        self.manifest = self.root / "manifest.json"
        self.write_manifest(self.entries)

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self, entries):
        self.manifest.write_text(json.dumps({"schema_version": 1, "files": entries}))

    def load(self):
        return load_raw_lipid_rna(self.manifest, np.arange(451, 454), self.root, max_per_class=3, seed=42)

    def test_both_layouts_preserve_intensity_and_positions(self):
        for instrument in ("Renishaw", "Horiba"):
            p = self.root / f"{instrument}_RNA.txt"
            parsed = read_raw_export(p, instrument)
            np.testing.assert_array_equal(parsed["wavenumbers"], [451, 452, 453])
            offset = 100 if instrument == "Renishaw" else 300
            np.testing.assert_array_equal(parsed["spectra"][0], offset + np.array([451, 452, 453]))
            np.testing.assert_array_equal(parsed["xy"][:, 0], np.arange(5))

    def test_format_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "manifest assigns"):
            read_raw_export(self.root / "Renishaw_RNA.txt", "Horiba")

    def test_incomplete_long_spectrum_rejected(self):
        p = self.root / "Renishaw_RNA.txt"
        p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            read_raw_export(p)

    def test_no_endpoint_extrapolation(self):
        with self.assertRaisesRegex(ValueError, "refusing extrapolation"):
            interpolate_without_extrapolation(np.array([450.855, 452, 453]), np.ones((2,3)), np.arange(450,454), "test")

    def test_labels_provenance_and_sampling_independent_of_manifest_order(self):
        first = self.load()
        np.random.seed(889)
        np.random.normal(size=300)
        self.write_manifest(list(reversed(self.entries)))
        second = self.load()
        for key in first["data"]:
            np.testing.assert_array_equal(first["data"][key], second["data"][key])
        self.assertEqual(first["rows"].to_dict("records"), second["rows"].to_dict("records"))
        self.assertEqual(len(first["rows"]["source_id"].unique()),12)
        for frame in as_notebook_frames(first).values():
            self.assertEqual(frame["Virus"].value_counts().to_dict(), {"RNA":3,"Lipid":3})
            self.assertEqual(frame.columns[-3:].tolist(), ["Virus","Subtype","Strain"])

    def test_changed_file_and_missing_file_do_not_fallback(self):
        p = self.root / self.entries[0]["relative_path"]
        p.write_text(p.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "content changed"):
            self.load()
        p.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "no legacy CSV fallback"):
            self.load()

    def test_duplicate_contents_rejected(self):
        source = self.root / self.entries[0]["relative_path"]
        duplicate = self.root / "copy.txt"
        duplicate.write_bytes(source.read_bytes())
        entry = {**self.entries[0], "relative_path": duplicate.name}
        self.write_manifest(self.entries + [entry])
        with self.assertRaisesRegex(ValueError, "Duplicate file content"):
            self.load()


if __name__ == "__main__":
    unittest.main()
