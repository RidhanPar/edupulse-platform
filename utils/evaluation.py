"""Evaluation outputs of a trained model, read from its ModelArtifact record."""
from __future__ import annotations


def load_metrics(artifact) -> dict | None:
    if artifact is None:
        return None
    return {"best_model": artifact.algorithm_name, **artifact.metrics}


def load_importances(artifact) -> dict | None:
    return artifact.feature_importances if artifact is not None else None


def load_model_comparison(artifact) -> list[dict] | None:
    return artifact.model_comparison if artifact is not None else None
