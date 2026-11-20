"""Registro do modelo no MLflow Model Registry (fora do `dvc repro`, como no curso).

Etapa do ciclo MLOps: **Model Registry** (controle de versões dos modelos).

- Entrada: a run do último pipeline (`artifacts/mlflow_run.json`) e as runs no MLflow.
- Saída: uma nova VERSÃO do modelo registrado `fraude_xgboost`, com o alias `champion`
  apontando para ela, e as tags da versão (PR-AUC no teste, limiar, run de origem).

Por que um registro, se o modelo já está na run? A run guarda TODOS os modelos que já foram
treinados (inclusive os ruins). O registry guarda só os que foram PROMOVIDOS, numerados
(v1, v2, v3...). O app e outros sistemas pedem "o champion" e não precisam saber o run_id:
  - promover um modelo novo = mover o alias para a nova versão;
  - voltar atrás (rollback) = mover o alias de volta para a versão anterior.

Qual run é registrada?
- Pipeline normal (`dvc repro`): a run gravada em artifacts/mlflow_run.json.
- Experimentos do DVC (`dvc exp run`): as runs são filhas de uma run "mãe"; registra a filha de
  MAIOR PR-AUC no teste (o mesmo raciocínio do `register_artifacts.py` do curso, que usava a acurácia).

Por que não é um estágio do dvc.yaml? Registrar cria uma versão NOVA a cada execução (efeito
colateral fora do projeto) e não produz arquivo de saída para o DVC controlar. No Airflow, ele é a
tarefa que vem depois de todos os estágios do DVC.

Executar: python -m src.model_registry.register_model
"""
import logging

import mlflow
from mlflow import MlflowClient

from src.config import PATHS, load_params
from src.tracking import setup_mlflow
from src.utils import load_json

logger = logging.getLogger("src.model_registry.register_model")


def choose_run(client: MlflowClient, experiment_id: str, run_id: str) -> str:
    """Devolve a run a registrar: a do pipeline ou, se ela for experimento do DVC, a melhor irmã."""
    run = client.get_run(run_id)
    parent_id = run.data.tags.get("mlflow.parentRunId")
    if not parent_id:
        return run_id
    irmas = client.search_runs(experiment_ids=[experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
                               order_by=["metrics.test_pr_auc DESC"], max_results=1)
    melhor = irmas[0]
    logger.info(f"Experimentos do DVC: melhor run {melhor.info.run_id} ({melhor.data.tags.get('mlflow.runName')}) "
                f"com test_pr_auc = {melhor.data.metrics.get('test_pr_auc'):.4f}")
    return melhor.info.run_id


def register_model(run_id: str, nome: str, alias: str) -> int:
    """Cria uma nova versão do modelo registrado a partir da run e aponta o alias para ela.

    Saída: o número da versão criada.
    """
    client = MlflowClient()
    versao = mlflow.register_model(f"runs:/{run_id}/model", nome)
    run = client.get_run(run_id)
    for chave in ("test_pr_auc", "test_recall", "test_economia_liquida"):
        if chave in run.data.metrics:
            client.set_model_version_tag(nome, versao.version, chave, f"{run.data.metrics[chave]:.4f}")
    client.set_model_version_tag(nome, versao.version, "limiar", run.data.params.get("limiar", ""))
    client.set_registered_model_alias(nome, alias, versao.version)
    logger.info(f"Modelo '{nome}' versão {versao.version} registrado a partir da run {run_id}; alias '{alias}' -> v{versao.version}")
    return int(versao.version)


def main() -> None:
    """Orquestra o registro: escolhe a run e cria a versão com o alias configurado no params.yaml."""
    cfg = load_params()["inference"]
    experiment_id = setup_mlflow()
    run_id = choose_run(MlflowClient(), experiment_id, load_json(PATHS.mlflow_run)["run_id"])
    register_model(run_id, cfg["registered_model_name"], cfg["alias"])
    logger.info("Registro do modelo concluído")


if __name__ == "__main__":
    main()
