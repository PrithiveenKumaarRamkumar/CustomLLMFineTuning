"""
Airflow DAG to orchestrate the user-uploaded dataset pipeline with DVC stages.

Stages:
  1) organize     -> python scripts/dataset_filter.py (dvc stage: organize)
  2) preprocessing-> python scripts/preprocessing.py (dvc stage: preprocessing)
  3) validation   -> python scripts/schema_validation.py (dvc stage: validation)
  4) statistics   -> python scripts/statistics_generation.py (dvc stage: statistics)

This DAG assumes the repository is available on the worker and the working
Directory for commands is the repo root. If needed, set the environment
variable AIRFLOW_REPO_DIR to an absolute path, and the tasks will `cd` into it
before running commands.
"""

from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from typing import Any, Dict
import sys


# Ensure we can import the monitoring exporter from the repo
_REPO_DIR = os.environ.get("AIRFLOW_REPO_DIR", ".")
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)
try:
    from monitoring.scripts.metrics_exporter import MetricsExporter  # type: ignore
except Exception:
    MetricsExporter = None  # Fallback if not available


def _with_cd(cmd: str) -> str:
    repo_dir = os.environ.get("AIRFLOW_REPO_DIR", ".")
    # Use POSIX-style chaining; Airflow runs on Linux typically
    return f"cd {repo_dir} && {cmd}"


def _dvc_repro(stage: str) -> str:
    return _with_cd(f"dvc repro {stage}")


def _dvc_repro_or_all(stage: str | None = None) -> str:
    return _with_cd("dvc repro" if stage is None else f"dvc repro {stage}")


default_args = {
    "owner": "data-platform",
    "depends_on_past": False,
    "retries": 1,
}

with DAG(
    dag_id="data_pipeline_dag",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # trigger manually; can be set to '@daily'
    catchup=False,
    tags=["dataset", "code", "dvc"],
) as dag:

    def _on_task_success(context: Dict[str, Any]):
        """Push basic task metrics to Prometheus Pushgateway if available."""
        if MetricsExporter is None:
            return
        ti = context["ti"]
        start = ti.start_date
        end = ti.end_date
        duration = (end - start).total_seconds() if start and end else 0.0
        stage_name = ti.task_id
        # Use default gateway url localhost:9091 (within docker-compose it's pushgateway:9091)
        # If AIRFLOW is in a container alongside pushgateway, set PUSHGATEWAY_URL accordingly in env.
        gateway_url = os.environ.get("PUSHGATEWAY_URL", "pushgateway:9091")
        try:
            exporter = MetricsExporter(pushgateway_url=gateway_url, job_name="airflow_pipeline")
            exporter.export_batch_metrics(stage_name=stage_name, records_processed=0, duration=duration)
        except Exception:
            pass

    def _on_task_failure(context: Dict[str, Any]):
        # Also push a zero-duration metric to indicate failure (or we could use a dedicated status gauge)
        if MetricsExporter is None:
            return
        ti = context["ti"]
        stage_name = ti.task_id
        gateway_url = os.environ.get("PUSHGATEWAY_URL", "pushgateway:9091")
        try:
            exporter = MetricsExporter(pushgateway_url=gateway_url, job_name="airflow_pipeline")
            exporter.export_batch_metrics(stage_name=stage_name, records_processed=0, duration=0.0)
        except Exception:
            pass

    organize = BashOperator(
        task_id="organize",
        bash_command=_dvc_repro("organize"),
        on_success_callback=_on_task_success,
        on_failure_callback=_on_task_failure,
    )

    preprocessing = BashOperator(
        task_id="preprocessing",
        bash_command=_dvc_repro("preprocessing"),
        on_success_callback=_on_task_success,
        on_failure_callback=_on_task_failure,
    )

    validation = BashOperator(
        task_id="validation",
        bash_command=_dvc_repro("validation"),
        on_success_callback=_on_task_success,
        on_failure_callback=_on_task_failure,
    )

    statistics = BashOperator(
        task_id="statistics",
        bash_command=_dvc_repro("statistics"),
        on_success_callback=_on_task_success,
        on_failure_callback=_on_task_failure,
    )

    # Dependencies
    organize >> preprocessing >> validation >> statistics
