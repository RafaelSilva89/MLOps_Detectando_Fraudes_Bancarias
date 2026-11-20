"""Pacote `src`: o código do pipeline de ML, um subpacote por etapa do ciclo MLOps.

Etapa do ciclo MLOps: **Logging** (transversal a todas as etapas).

Ao importar qualquer módulo de `src`, este arquivo roda primeiro e:
1. carrega as variáveis do arquivo `.env` (ex.: MLFLOW_TRACKING_URI), sem escrevê-las no código;
2. configura o logging UMA vez para o projeto todo: cada mensagem sai no terminal e é
   gravada em `logs/pipeline.log`, com data, módulo, linha e nível.

Por que logging e não print? O log tem nível (INFO, WARNING, ERROR), horário e origem, e
fica salvo: quando o pipeline roda sozinho (DVC, Airflow), é o log que conta o que aconteceu.
"""
import logging
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

_LOGS = _ROOT / "logs"
_LOGS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(_LOGS / "pipeline.log", encoding="utf-8")],
)
# Bibliotecas muito "falantes" ficam só com avisos
for _ruidoso in ("mlflow", "alembic", "urllib3", "git", "matplotlib"):
    logging.getLogger(_ruidoso).setLevel(logging.WARNING)
