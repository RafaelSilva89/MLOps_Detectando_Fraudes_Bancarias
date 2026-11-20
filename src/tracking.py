"""Rastreio de experimentos com MLflow: onde fica o servidor e como abrir/reabrir a run do pipeline.

Etapa do ciclo MLOps: **Experiment Tracking**.

Conceitos:
- *Experimento* (`deteccao_fraudes`): a "pasta" que agrupa todas as execuções deste projeto.
- *Run*: UMA execução do pipeline. Guarda parâmetros (o que foi usado), métricas (o que saiu),
  artefatos (arquivos) e o modelo. Comparar runs no `mlflow ui` é como comparar versões do notebook.

Como as etapas compartilham a mesma run:
1. `train_model` cria a run e grava o `run_id` em `artifacts/mlflow_run.json`;
2. `optimize_threshold` e `evaluate_model` leem esse arquivo e REABREM a mesma run para
   acrescentar o limiar, as métricas de teste e o model card.
Assim, uma run = um modelo completo (treino + decisão + avaliação).

Experimentos do DVC (`dvc exp run`): o DVC define a variável DVC_EXP_NAME. Nesse caso, cada
experimento vira uma run FILHA de uma run "mãe" marcada com a tag `dvc_exp`, como no material
do curso. O `register_model` usa isso para escolher o melhor experimento.

Onde os dados ficam: MLFLOW_TRACKING_URI (no .env). Padrão: `sqlite:///mlflow.db` na raiz.
O backend em banco (SQLite) é necessário para o Model Registry.
"""
import logging
import os
from contextlib import contextmanager

import mlflow

from src.config import PATHS
from src.utils import load_json, save_json

logger = logging.getLogger("src.tracking")

EXPERIMENTO = "deteccao_fraudes"


def setup_mlflow() -> str:
    """Aponta o MLflow para o servidor/arquivo configurado e seleciona o experimento.

    Saída: o id do experimento.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{(PATHS.root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENTO) is None:
        # Num servidor remoto (DagsHub), ele decide onde guardar os arquivos. Localmente, os
        # artefatos (modelos, JSONs) ficam em <raiz>/mlruns, e não na pasta de onde o comando rodou.
        local = uri.startswith("sqlite") or uri.startswith("file")
        mlflow.create_experiment(EXPERIMENTO, artifact_location=(PATHS.root / "mlruns").as_uri() if local else None)
    experimento = mlflow.set_experiment(EXPERIMENTO)
    return experimento.experiment_id


def _parent_run_id_dvc(experiment_id: str) -> str:
    """Run "mãe" dos experimentos do DVC (cria na primeira vez)."""
    runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string="tags.dvc_exp = 'True'",
                              order_by=["start_time DESC"])
    if not runs.empty:
        return runs.iloc[0].run_id
    with mlflow.start_run(run_name="experimentos_dvc") as parent:
        mlflow.set_tag("dvc_exp", "True")
        return parent.info.run_id


@contextmanager
def start_training_run():
    """Abre uma run NOVA (usada só pelo estágio de treino) e grava o run_id para os próximos estágios."""
    experiment_id = setup_mlflow()
    extra = {}
    nome_exp = os.getenv("DVC_EXP_NAME")
    if nome_exp:
        extra = {"parent_run_id": _parent_run_id_dvc(experiment_id), "run_name": nome_exp, "nested": True}
    with mlflow.start_run(**extra) as run:
        save_json({"run_id": run.info.run_id, "experiment_id": experiment_id}, PATHS.mlflow_run)
        logger.info(f"MLflow: run {run.info.run_id} aberta no experimento '{EXPERIMENTO}'")
        yield run


@contextmanager
def resume_pipeline_run():
    """Reabre a run criada pelo treino (para os estágios seguintes acrescentarem informações)."""
    setup_mlflow()
    run_id = load_json(PATHS.mlflow_run)["run_id"]
    with mlflow.start_run(run_id=run_id) as run:
        yield run
