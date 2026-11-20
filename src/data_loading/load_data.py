"""Estágio 1 do pipeline: carregar a base original e gravá-la como dado bruto.

Etapa do ciclo MLOps: **Data Ingestion / Organização dos dados**.

- Entrada: `data/external/transferencias.zip` (a base original, versionada com `dvc add`).
- Saída:   `data/raw/raw.csv` (284.807 transações x 31 colunas, sem nenhuma alteração).

Por que separar "carregar" de "tratar"? O dado bruto é a "verdade" de onde tudo parte.
Guardá-lo intacto permite refazer qualquer etapa seguinte e auditar o que mudou.
No projeto-exemplo, este estágio baixava a base do scikit-learn; aqui lê o zip local.

Executar: python -m src.data_loading.load_data
"""
import logging
from pathlib import Path

import pandas as pd

from src.config import PATHS, load_params

logger = logging.getLogger("src.data_loading.load_data")


def fetch_data(source: Path) -> pd.DataFrame:
    """Lê a base original. O pandas abre o CSV direto de dentro do zip.

    Entrada: caminho do zip (ou CSV).
    Saída: DataFrame com as 31 colunas originais.
    """
    logger.info(f"Lendo a base original de {source}")
    data = pd.read_csv(source)
    logger.info(f"Base lida: {data.shape[0]:,} linhas x {data.shape[1]} colunas")
    return data


def save_data(data: pd.DataFrame, output_path: Path) -> None:
    """Grava o dado bruto em CSV, sem índice."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Gravando os dados brutos em {output_path}")
    data.to_csv(output_path, index=False)


def main() -> None:
    """Orquestra o estágio: lê o params.yaml, carrega a base e grava em data/raw/."""
    params = load_params()
    raw = fetch_data(PATHS.root / params["load_data"]["source"])
    save_data(raw, PATHS.raw)
    logger.info("Carregamento dos dados concluído")


if __name__ == "__main__":
    main()
