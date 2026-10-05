"""Thin wrapper over MLflow tracking + model registry.

MLflow 3 deprecates registry *stages* in favour of *aliases*, so a "stage"
here is an alias: ``promote(name, "7", "production")`` points
``models:/<name>@production`` at version 7. Rolling back is promoting an
older version — nothing is deleted.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

EXPERIMENT = "property-valuation"
PRODUCTION = "production"


def _resolve_uri(mlflow_uri: str) -> tuple[str, str | None]:
    """A bare path becomes a SQLite store inside that directory (the file
    store is deprecated in MLflow 3); explicit URIs are passed through."""
    if "://" in mlflow_uri or mlflow_uri.startswith("databricks"):
        return mlflow_uri, None
    root = Path(mlflow_uri).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{root / 'mlflow.db'}", (root / "artifacts").as_uri()


class Registry:
    def __init__(self, mlflow_uri: str) -> None:
        self.tracking_uri, artifact_root = _resolve_uri(mlflow_uri)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.tracking_uri)
        self.client = MlflowClient(self.tracking_uri, self.tracking_uri)
        exp = self.client.get_experiment_by_name(EXPERIMENT)
        self.experiment_id = (exp.experiment_id if exp else
                              self.client.create_experiment(EXPERIMENT, artifact_location=artifact_root))

    def log_run(self, model: Any, params: dict, metrics: dict, artifacts: dict[str, str],
                model_name: str, tags: dict[str, str] | None = None) -> str:
        """Log params, metrics, artifact files and the pickled model. Returns the run id."""
        with mlflow.start_run(experiment_id=self.experiment_id) as run:
            mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if isinstance(v, (int, float)) and v == v})
            for name, path in artifacts.items():
                mlflow.log_artifact(path, artifact_path=name)
            info = mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
            mlflow.set_tags({"model_name": model_name, "model_uri": info.model_uri, **(tags or {})})
            return run.info.run_id

    def register_best(self, model_name: str, metric: str, higher_is_better: bool = True,
                      training_id: str | None = None) -> str:
        """Register the best run (by ``metric``) for ``model_name``. Returns the new version.

        ``training_id`` restricts the search to one training invocation, so a
        run scored on a different dataset can never be picked.
        """
        order = "DESC" if higher_is_better else "ASC"
        flt = f"tags.model_name = '{model_name}'"
        if training_id:
            flt += f" and tags.training_id = '{training_id}'"
        runs = self.client.search_runs(
            [self.experiment_id], filter_string=flt,
            order_by=[f"metrics.{metric} {order}"], max_results=1,
        )
        if not runs:
            raise LookupError(f"no runs logged for {model_name!r}")
        best = runs[0]
        mv = mlflow.register_model(best.data.tags["model_uri"], model_name,
                                   tags={"run_id": best.info.run_id})
        return str(mv.version)

    def promote(self, model_name: str, version: str, stage: str) -> None:
        """Point alias ``stage`` (e.g. "production", "staging") at ``version``."""
        self.client.set_registered_model_alias(model_name, stage.lower(), str(version))

    def production_version(self, model_name: str) -> Any:
        return self.client.get_model_version_by_alias(model_name, PRODUCTION)

    def load_production(self, model_name: str) -> Any:
        """Load the model currently aliased ``production``."""
        return mlflow.sklearn.load_model(f"models:/{model_name}@{PRODUCTION}")

    def download_production_artifact(self, model_name: str, artifact_path: str) -> str:
        """Local path to an artifact logged with the production version's run."""
        mv = self.production_version(model_name)
        run_id = mv.tags.get("run_id") or mv.run_id
        dst = tempfile.mkdtemp(prefix="pvs-")
        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path,
                                                   dst_path=dst, tracking_uri=self.tracking_uri)

    def load_production_json(self, model_name: str, artifact_path: str) -> dict:
        path = self.download_production_artifact(model_name, artifact_path)
        if os.path.isdir(path):
            path = os.path.join(path, os.listdir(path)[0])
        with open(path) as f:
            return json.load(f)

    def list_versions(self, model_name: str) -> list[dict[str, Any]]:
        versions = self.client.search_model_versions(f"name = '{model_name}'")
        return [{"version": v.version, "run_id": v.run_id, "aliases": list(v.aliases),
                 "created": v.creation_timestamp} for v in versions]
