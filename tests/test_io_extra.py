"""Extra unit tests for IO module."""

from __future__ import annotations

import os
import tempfile
from isodde.io import write_coords_to_sdf, parse_sdf_file


def test_sdf_write_and_parse():
    coords = [
        [1.234, 5.678, -9.012],
        [-2.345, 0.000, 4.567]
    ]
    elements = ["C", "O"]

    with tempfile.TemporaryDirectory() as tmpdir:
        sdf_path = os.path.join(tmpdir, "test.sdf")
        
        # Write
        write_coords_to_sdf(coords, elements, sdf_path, molecule_name="TestMolecule")
        
        assert os.path.exists(sdf_path)

        # Parse
        parsed_coords, parsed_elements = parse_sdf_file(sdf_path)

        assert len(parsed_coords) == 2
        assert len(parsed_elements) == 2
        assert parsed_elements == ["C", "O"]
        
        # Check tolerance (due to float formatting in file output)
        assert abs(parsed_coords[0][0] - 1.234) < 1e-3
        assert abs(parsed_coords[0][1] - 5.678) < 1e-3
        assert abs(parsed_coords[0][2] - (-9.012)) < 1e-3
        assert abs(parsed_coords[1][0] - (-2.345)) < 1e-3
