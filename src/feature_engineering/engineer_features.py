"""Estágio 4 do pipeline: criar e transformar os atributos (features).

Etapa do ciclo MLOps: **Feature Engineering**.

- Entrada: `data/preprocessed/{train,val,test}.csv`.
- Saída:   `data/processed/{train,val,test}.csv` (30 variáveis do modelo + Target).

A transformação (notebook v2, Fase 02 item 4):
- cria `hora_do_dia` = (segundos desde a 1ª transação módulo 86.400) / 3.600, de 0 a 24 h.
  Captura o padrão da madrugada (fraude 3,5x mais comum entre 0 h e 6 h);
- remove o `Timestamp` cru: ele é a posição na janela de coleta de 48 h, um valor que nunca se
  repetiria no futuro (vazamento temporal);
- mantém as 28 componentes e o valor como estão (usar só as 10 melhores piorou a PR-AUC).

Por que uma função "sem estado" importa? `build_features` não aprende nada com os dados (não há
`fit`), então pode ser aplicada a treino, teste e transações novas sem vazamento. E é a MESMA
função usada na inferência (`src/inference/predict.py`): treino e produção transformam os dados
do mesmo jeito, evitando o erro clássico de *training-serving skew*.

Executar: python -m src.feature_engineering.engineer_features
"""
import logging

import pandas as pd

from src.config import ALVO, COLUNA_TEMPO, COLUNAS_MODELO, ID, PATHS
from src.utils import load_splits, save_splits

logger = logging.getLogger("src.feature_engineering.engineer_features")

SEGUNDOS_POR_DIA = 24 * 60 * 60


def hora_do_dia(timestamp: pd.Series) -> pd.Series:
    """Converte os segundos desde a 1ª transação na hora do dia (0 a 24).

    Supõe que a contagem começa perto da meia-noite: o volume de transações despenca entre
    1 h e 7 h dessa escala, como numa madrugada (conferido na EDA do notebook, H2).
    """
    return (timestamp % SEGUNDOS_POR_DIA) / 3600


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma uma base no formato bruto nas 30 variáveis do modelo (na ordem de COLUNAS_MODELO).

    Entrada: DataFrame com Timestamp, as 28 componentes e hist_saldo (id_transacao e Target são opcionais).
    Saída: DataFrame com COLUNAS_MODELO, mesmo índice; id_transacao vem antes e Target depois, se existirem.
    """
    features = df.assign(hora_do_dia=hora_do_dia(df[COLUNA_TEMPO]))[COLUNAS_MODELO]
    if ID in df.columns:
        features.insert(0, ID, df[ID])
    if ALVO in df.columns:
        features = features.assign(**{ALVO: df[ALVO]})
    return features


def main() -> None:
    """Aplica `build_features` às três partes e grava em data/processed/."""
    splits = load_splits(PATHS.preprocessed_dir)
    processados = {nome: build_features(df) for nome, df in splits.items()}
    logger.info(f"Features criadas: {len(COLUNAS_MODELO)} variáveis ({', '.join(COLUNAS_MODELO[-2:])} no fim)")
    save_splits(processados, PATHS.processed_dir)
    logger.info("Engenharia de atributos concluída")


if __name__ == "__main__":
    main()
