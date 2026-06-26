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


def test_cli_sdf_input_and_output():
    from isodde.cli import main
    coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    elements = ["C", "O"]
    with tempfile.TemporaryDirectory() as tmpdir:
        sdf_in = os.path.join(tmpdir, "in.sdf")
        write_coords_to_sdf(coords, elements, sdf_in)
        
        output_dir = os.path.join(tmpdir, "out")
        
        # Run CLI with --sdf-ligand
        ret = main(["--sequence", "MTEYKL", "--sdf-ligand", sdf_in, "--output-dir", output_dir, "--small", "--num-seeds", "1"])
        assert ret == 0
        
        # Verify that output predicted_ligand.sdf exists
        assert os.path.exists(os.path.join(output_dir, "predicted_ligand.sdf"))
        assert os.path.exists(os.path.join(output_dir, "predicted_complex.pdb"))


def test_web_sdf_response():
    from fastapi.testclient import TestClient
    from isodde.web import app
    client = TestClient(app)
    response = client.post(
        "/api/predict",
        json={
            "protein_sequence": "MTEYKL",
            "ligand": "C,O,N",
            "num_seeds": 1
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "sdf_content" in data
    assert data["sdf_content"] is not None
    assert "V2000" in data["sdf_content"]
