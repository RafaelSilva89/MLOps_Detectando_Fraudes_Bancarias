"""DAG do Apache Airflow: orquestra o pipeline de ponta a ponta, de forma agendada.

Etapa do ciclo MLOps: **Pipeline Orchestration** (automação e agendamento).

DVC x Airflow: quem faz o quê?
- O DVC sabe COMO reproduzir o pipeline (dependências, cache, versões dos dados).
- O Airflow sabe QUANDO e EM QUE ORDEM rodar, com novas tentativas em caso de falha, histórico de
  execuções e uma interface web. Ele chama o DVC: cada estágio do dvc.yaml vira uma tarefa.

Tarefas (como no material do curso):
  dvc_load_data >> dvc_validate_data >> ... >> dvc_monitor >> register_model >> build_app_image

Detalhes de implementação:
- Os estágios são lidos do próprio dvc.yaml: adicionar um estágio lá cria a tarefa aqui.
- O projeto roda num ambiente Python SEPARADO do Airflow (PROJECT_PYTHON), para que as versões
  das bibliotecas do projeto (pandas, MLflow...) não briguem com as do Airflow.
- `build_app_image` monta a imagem do app (Dockerfile) usando o Docker-in-Docker do compose.

Como rodar: veja docker-compose.airflow.yaml e a seção "Airflow" do README.
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = os.getenv("PROJECT_PYTHON", "python")
# Os estágios do dvc.yaml chamam "python -m ...". Para esse "python" ser o do projeto (com scikit-learn,
# XGBoost...) e não o do Airflow, a pasta do ambiente do projeto vai na frente do PATH em cada tarefa.
PROJECT_BIN = str(Path(PROJECT_PYTHON).parent) if os.sep in PROJECT_PYTHON else ""
PREFIXO = f'export PATH="{PROJECT_BIN}:$PATH" && ' if PROJECT_BIN else ""


def get_dvc_stages() -> list[str]:
    """Nomes dos estágios, na ordem em que aparecem no dvc.yaml."""
    with open(PROJECT_ROOT / "dvc.yaml", encoding="utf-8") as f:
        return list(yaml.safe_load(f)["stages"])


default_args = {"owner": "mlops", "retries": 1, "retry_delay": timedelta(minutes=2)}

with DAG(
    dag_id="fraud_detection_pipeline",
    description="Detecção de fraudes: estágios do DVC -> registro no MLflow -> imagem do app",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@monthly",       # retreino mensal (plano de retreino do notebook); dá para disparar à mão
    catchup=False,
    tags=["mlops", "fraude"],
) as dag:
    dvc_tasks = [
        BashOperator(task_id=f"dvc_{stage}", cwd=str(PROJECT_ROOT), bash_command=f"{PREFIXO}{PROJECT_PYTHON} -m dvc repro {stage}")
        for stage in get_dvc_stages()
    ]

    register_model = BashOperator(
        task_id="register_model", cwd=str(PROJECT_ROOT),
        bash_command=f"{PREFIXO}{PROJECT_PYTHON} -m src.model_registry.register_model",
    )

    build_app_image = BashOperator(
        task_id="build_app_image", cwd=str(PROJECT_ROOT),
        bash_command="docker build -t fraude-api:latest .",
    )

    for anterior, seguinte in zip(dvc_tasks, dvc_tasks[1:]):
        anterior >> seguinte
    dvc_tasks[-1] >> register_model >> build_app_image
