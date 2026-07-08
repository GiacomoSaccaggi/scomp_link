# -*- coding: utf-8 -*-
"""
Smoke tests for scomp-link serve REST API.
Tests the Flask app using Flask's test client (no server needed).
"""
import json
import tempfile
import pytest
import numpy as np
import pandas as pd

try:
    from flask import Flask
    flask_available = True
except ImportError:
    flask_available = False

pytestmark = pytest.mark.skipif(not flask_available, reason="flask not installed")


@pytest.fixture
def trained_artifact(tmp_path):
    """Create a trained .scomp artifact for testing."""
    import scomp_link
    scomp_link.set_verbosity("silent")

    np.random.seed(42)
    df = pd.DataFrame({"x1": np.random.randn(100), "x2": np.random.randn(100),
                       "y": np.random.randn(100)})
    pipe = scomp_link.ScompLinkPipeline("test_serve")
    pipe.import_and_clean_data(df)
    pipe.select_variables(target_col="y")
    pipe.choose_model("numerical_prediction")
    pipe.run_pipeline(task_type="regression")

    artifact_path = str(tmp_path / "model.scomp")
    artifact = scomp_link.ScompArtifact()
    artifact.set_model(pipe.model)
    artifact.set_config(task_type="regression", target_col="y")
    artifact.set_metrics({"r2": 0.5})
    artifact.set_feature_schema(df[["x1", "x2"]])
    artifact.save(artifact_path)
    return artifact_path


@pytest.fixture
def serve_app(trained_artifact):
    """Create the Flask app from a trained artifact (mirrors cmd_serve logic)."""
    import scomp_link
    from flask import Flask, jsonify, request

    scomp_link.set_verbosity("silent")
    artifact = scomp_link.ScompArtifact.load(trained_artifact)

    app = Flask("scomp-link-serve-test")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": type(artifact.model).__name__})

    @app.route("/info", methods=["GET"])
    def info():
        return jsonify(artifact.info())

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True)
        if "instances" in data:
            df = pd.DataFrame(data["instances"])
        elif "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame([data])
        predictions = artifact.predict(df)
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()
        return jsonify({"predictions": predictions})

    @app.route("/schema", methods=["GET"])
    def schema():
        return jsonify({"features": artifact.feature_schema, "config": artifact.config})

    return app


class TestServeHealth:
    def test_health_returns_ok(self, serve_app):
        client = serve_app.test_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "model" in data

    def test_info_returns_metadata(self, serve_app):
        client = serve_app.test_client()
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "config" in data or "model_type" in data


class TestServePredict:
    def test_predict_instances(self, serve_app):
        client = serve_app.test_client()
        payload = {"instances": [{"x1": 0.5, "x2": -0.3}, {"x1": 1.0, "x2": 0.2}]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2

    def test_predict_data_key(self, serve_app):
        client = serve_app.test_client()
        payload = {"data": [{"x1": 0.1, "x2": 0.2}]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["predictions"]) == 1

    def test_predict_single_instance(self, serve_app):
        client = serve_app.test_client()
        payload = {"x1": 0.0, "x2": 0.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["predictions"]) == 1


class TestServeSchema:
    def test_schema_returns_features(self, serve_app):
        client = serve_app.test_client()
        resp = client.get("/schema")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "config" in data
