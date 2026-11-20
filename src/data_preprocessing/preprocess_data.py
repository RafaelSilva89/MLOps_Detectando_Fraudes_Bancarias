"""Estágio 3 do pipeline: limpar os dados e separar treino, validação e teste.

Etapa do ciclo MLOps: **Data Preprocessing**.

- Entrada: `data/raw/raw.csv`.
- Saídas:
  - `data/preprocessed/{train,val,test}.csv`: as três partes, com a coluna `id_transacao`;
  - `data/preprocessed/train_val_ids.csv`: a ordem das linhas de treino + validação (ver abaixo);
  - `metrics/preprocessing.json`: quantas duplicatas saíram e quantas fraudes há em cada parte.

O que faz (decisões do notebook v2, Fase 02 item 2.3 e Fase 03 item 4.1):
1. Remove as linhas duplicadas ANTES de separar. Com duplicatas, uma cópia pode cair no treino
   e a outra no teste: o modelo seria avaliado em algo que já viu (vazamento).
2. Cria `id_transacao` (a posição da linha na base sem duplicatas) para rastrear cada transação
   pelas etapas seguintes.
3. Separa 60% treino / 20% validação / 20% teste, ESTRATIFICADO pelo alvo: com 0,17% de fraudes,
   uma divisão comum poderia deixar a validação com bem menos fraudes que o treino.

Por que guardar a ordem de treino + validação? A validação cruzada que escolhe o algoritmo e o
modelo final usam treino + validação NA ORDEM em que a 1ª divisão os deixou (é assim no notebook).
A ordem muda a composição das dobras; guardando-a, o pipeline reproduz o notebook exatamente.

Por que não há imputação ou escala aqui (o projeto-exemplo tinha)? A base não tem nulos (o
estágio de validação garante) e os modelos de árvore não precisam de escala. A Regressão Logística
recebe o StandardScaler DENTRO do seu Pipeline, ajustado só no treino (ver model_training/models.py).

Executar: python -m src.data_preprocessing.preprocess_data
"""
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ALVO, ID, PATHS, load_params
from src.utils import read_csv, save_json, save_splits

logger = logging.getLogger("src.data_preprocessing.preprocess_data")


def remover_duplicatas(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove linhas idênticas em todas as colunas.

    Entrada: base bruta. Saída: (base sem duplicatas com índice 0..n-1, quantidade removida).
    """
    sem_duplicatas = df.drop_duplicates().reset_index(drop=True)
    return sem_duplicatas, len(df) - len(sem_duplicatas)


def split_data(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> dict[str, pd.DataFrame]:
    """Separa a base em treino, validação e teste, estratificando pelo alvo.

    Entradas: base sem duplicatas; `test_size` (fração da base); `val_size` (fração do que sobra); semente.
    Saída: {'train': ..., 'val': ..., 'test': ..., 'train_val': ...}. 'train_val' é a parte que
    sobrou da 1ª divisão (treino + validação), na ordem usada pela validação cruzada.

    É a mesma sequência de `train_test_split` do notebook (mesmas sementes e proporções), por isso
    as partes têm exatamente as mesmas transações que lá.
    """
    temp, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[ALVO])
    train, val = train_test_split(temp, test_size=val_size, random_state=seed, stratify=temp[ALVO])
    return {"train": train, "val": val, "test": test, "train_val": temp}


def resumo_split(splits: dict[str, pd.DataFrame]) -> dict:
    """Quantidade de transações e de fraudes em cada parte (para conferir a estratificação)."""
    return {nome: {"transacoes": len(df), "fraudes": int(df[ALVO].sum()), "taxa_fraude": round(float(df[ALVO].mean()), 6)}
            for nome, df in splits.items()}


def main() -> None:
    """Orquestra o estágio: lê o bruto, remove duplicatas, separa e grava."""
    params = load_params()
    raw = read_csv(PATHS.raw)
    limpo, removidas = remover_duplicatas(raw)
    limpo.insert(0, ID, limpo.index)
    logger.info(f"Duplicatas removidas: {removidas:,} | restam {len(limpo):,} transações ({int(limpo[ALVO].sum())} fraudes)")

    cfg = params["preprocess_data"]
    splits = split_data(limpo, cfg["test_size"], cfg["val_size"], params["seed"])
    train_val = splits.pop("train_val")
    resumo = resumo_split(splits)
    for nome, r in resumo.items():
        logger.info(f"{nome:>5}: {r['transacoes']:,} transações, {r['fraudes']} fraudes ({r['taxa_fraude']:.3%})")

    save_splits(splits, PATHS.preprocessed_dir)
    train_val[[ID]].to_csv(PATHS.train_val_order, index=False)
    save_json({"duplicatas_removidas": removidas, "linhas_sem_duplicatas": len(limpo), "splits": resumo},
              PATHS.metrics_dir / "preprocessing.json")
    logger.info("Pré-processamento concluído")


if __name__ == "__main__":
    main()
