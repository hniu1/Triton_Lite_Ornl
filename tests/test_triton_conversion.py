import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from types import SimpleNamespace
import io
import tempfile
import json
import hashlib
import zipfile
import netCDF4

import numpy as np

from data_preprocessing.m0_generate_netcdf_from_zip import bin2array, get_output_files, output_step
from stage2.target_integrity import check_target_metadata, CORRECTED_CONVERSION
from stage2.audit_raw_targets import raw_patch, archive_members
from data_preprocessing import m0_generate_netcdf_from_zip as converter
from data_preprocessing.corrected_event_pilot import read_member, validate


class ConversionTest(unittest.TestCase):
    def test_sampling_guard_rejects_legacy_and_wrong_manifest(self):
        from stage2.target_integrity import require_corrected_sampling
        with tempfile.TemporaryDirectory() as root:
            manifest, sampling = Path(root) / "manifest", Path(root) / "sampling"
            manifest.mkdir()
            sampling.mkdir()
            (manifest / "dynamic_manifest.parquet").write_bytes(b"corrected manifest")
            metadata_path = sampling / "sampling_metadata.json"
            metadata_path.write_text('{}')
            with self.assertRaisesRegex(ValueError, "legacy"):
                require_corrected_sampling(sampling, manifest)
            metadata = dict(conversion_version=CORRECTED_CONVERSION, source_manifest_sha256="wrong")
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "this corrected manifest"):
                require_corrected_sampling(sampling, manifest)
            metadata["source_manifest_sha256"] = hashlib.sha256(b"corrected manifest").hexdigest()
            metadata_path.write_text(json.dumps(metadata))
            require_corrected_sampling(sampling, manifest)

    def test_streaming_conversion_end_to_end(self):
        with tempfile.TemporaryDirectory() as root:
            source, output = Path(root)/"D029.zip", Path(root)/"D029.nc"
            with zipfile.ZipFile(source, "w") as archive:
                for prefix in "HUV":
                    for step in range(1, 121):
                        values = np.zeros((4, 5), dtype="<f4")
                        values[1:3, 1:4] = np.array([[step, 0, -step], [2, -9999, 3]])
                        archive.writestr(f"D029/{prefix}_{step:02d}_00.dat", values.tobytes())
            with zipfile.ZipFile(source) as archive:
                members = {p: [Path(n) for n in archive_members(archive, p)[1]] for p in "HUV"}
                converter.create_netcdf(output, members, 3, 2, converter.CONASAUGA_DEFAULT_GT,
                                        converter.CONASAUGA_DEFAULT_REF_TIME_UNITS,
                                        converter.parse_time_origin(converter.CONASAUGA_DEFAULT_TIME_ORIGIN),
                                        0., 1, 20, 10800, 1800, converter.CONASAUGA_DEFAULT_CRS_WKT, 1, 0,
                                        array_reader=lambda member, **kw: read_member(archive, member, **kw))
                window = SimpleNamespace(row_start=0, col_start=0, height=2, width=3)
                record, checks = validate(output, archive, members, [(2264, window)], 2, 3,
                                          (8, 9, 29, 98, 99, 119))
                self.assertEqual(record["n_times"], 120)
                self.assertEqual(sum(c["negative_cells"] for c in checks), 12)
            with netCDF4.Dataset(output, "a") as nc:
                nc.triton_conversion_complete = "false"
            from data_preprocessing.m3_build_dynamic_manifest import inspect_file
            with self.assertRaisesRegex(ValueError, "incomplete"):
                inspect_file(output, "D029", "conasauga",
                             dict(depth="output_depth", component_x="output_velocity_x",
                                  component_y="output_velocity_y", time="out_time"), (2, 3))

    def test_independent_raw_patch_reader_preserves_signed_values_and_border(self):
        values = np.arange(20, dtype=np.float32).reshape(4, 5) - 10
        archive = Mock()
        archive.getinfo.return_value.file_size = values.nbytes
        archive.open.return_value = io.BytesIO(values.tobytes())
        window = SimpleNamespace(row_start=0, col_start=1, height=2, width=2)
        actual = raw_patch(archive, "U_01_00.dat", window, rows=2, cols=3)
        np.testing.assert_array_equal(actual, values[1:3, 2:4])

    def test_audit_reports_lexical_numeric_order_difference(self):
        archive = Mock()
        archive.namelist.return_value = [f"D/H_{step:02d}_00.dat" for step in range(1, 121)]
        legacy, numeric = archive_members(archive, "H")
        self.assertNotEqual(legacy, numeric)
        self.assertEqual([output_step(Path(name)) for name in numeric], list(range(1, 121)))
    def test_legacy_and_wrong_semantics_cannot_pass_training_guard(self):
        with self.assertRaisesRegex(ValueError, "contaminated"):
            check_target_metadata({"component_semantics": "unit_discharge"})
        with self.assertRaisesRegex(ValueError, "semantics"):
            check_target_metadata({"conversion_version": CORRECTED_CONVERSION, "component_semantics": "velocity"})
        check_target_metadata({"conversion_version": CORRECTED_CONVERSION, "component_semantics": "unit_discharge"})
    def test_numeric_steps_not_lexicographic_filenames(self):
        names = [f"/test/D/output/flood2d/bin/H_{step:02d}_00.dat" for step in range(1, 121)]
        with patch("data_preprocessing.m0_generate_netcdf_from_zip.glob.glob", return_value=sorted(names)):
            files = get_output_files(Path("/test"), "H")
        self.assertEqual([output_step(path) for path in files], list(range(1, 121)))
        self.assertLess(files.index(Path(names[29])), files.index(Path(names[99])))

    def test_missing_and_duplicate_output_steps_fail(self):
        for names in (["/a/H_01_00.dat", "/a/H_03_00.dat"],
                      ["/a/H_01_00.dat", "/b/H_01_00.dat"]):
            with patch("data_preprocessing.m0_generate_netcdf_from_zip.glob.glob", return_value=names):
                with self.assertRaisesRegex(ValueError, "nonsequential"):
                    get_output_files(Path("/test"), "H")

    def test_signed_discharge_is_preserved_depth_threshold_is_not_changed(self):
        path = Mock()
        path.stat.return_value.st_size = 4 * 4
        array = np.array([-2, 0, 3, -9999], dtype=np.float32)
        with patch("data_preprocessing.m0_generate_netcdf_from_zip.np.fromfile", return_value=array):
            signed = bin2array(path, rows=2, cols=2, threshold=0, pad=0, preserve_signed=True)
            depth = bin2array(path, rows=2, cols=2, threshold=0, pad=0)
        self.assertEqual(float(signed[0, 0]), -2)
        self.assertEqual(float(signed[0, 1]), 0)
        self.assertTrue(np.isnan(signed[1, 1]))
        self.assertTrue(np.isnan(depth[0, 0]))
        self.assertTrue(np.isnan(depth[0, 1]))
        self.assertEqual(float(depth[1, 0]), 3)


if __name__ == "__main__":
    unittest.main()
