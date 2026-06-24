"""Unit and integration tests for the IsoDDE web server API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from isodde.web import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Test that the health check endpoint returns 200 and correct status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "model" in response.json()


def test_predict_endpoint_invalid_sequence() -> None:
    """Test prediction endpoint with empty and invalid protein sequences."""
    # Empty sequence
    response = client.post("/api/predict", json={"protein_sequence": ""})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

    # Invalid sequence containing non-amino-acid characters
    response = client.post("/api/predict", json={"protein_sequence": "MTEYKL123"})
    assert response.status_code == 400
    assert "amino acid codes" in response.json()["detail"].lower()


def test_predict_endpoint_valid_run() -> None:
    """Test running a prediction end-to-end using a short sequence and custom seeds."""
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
    
    # Check fields
    assert "pLDDT" in data
    assert "ptm" in data
    assert "binding_affinity_pkd" in data
    assert "pockets" in data
    assert "interface_contact_probs" in data
    assert "protein_ligand_contact_probs" in data
    assert "pdb_content" in data
    
    # Check types and properties
    assert isinstance(data["pLDDT"], float)
    assert isinstance(data["ptm"], float)
    assert data["binding_affinity_pkd"] is None or isinstance(data["binding_affinity_pkd"], float)
    assert isinstance(data["pockets"], list)
    assert data["protein_length"] == 6
    assert data["ligand_length"] == 3
    assert len(data["elements"]) == 9
    assert "ATOM  " in data["pdb_content"]
