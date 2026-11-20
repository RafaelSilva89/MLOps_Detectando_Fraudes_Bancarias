"""Funções utilitárias pequenas usadas por vários estágios (ler e gravar JSON e CSV).

Por que existe? Sem isto, cada estágio repetiria o mesmo `open(...)`/`json.dump(...)`.
Centralizar garante o mesmo formato em todos os arquivos (UTF-8, indentação, acentos legíveis).
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _converter(obj: Any) -> Any:
    """Converte tipos do numpy (int64, float32, ndarray) para tipos que o JSON aceita."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Tipo não serializável em JSON: {type(obj)}")


def save_json(data: dict, path: Path) -> None:
    """Grava `data` em `path` (cria a pasta se preciso)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_converter)


def load_json(path: Path) -> dict:
    """Lê um JSON gravado por `save_json`."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    """Lê um CSV gravado pelo pipeline com `float_precision="round_trip"`.

    Por que? O pandas grava floats com todas as casas necessárias; lendo em modo "round_trip",
    cada número volta EXATAMENTE igual ao gravado. Sem isso, o último dígito de alguns valores
    pode mudar e o modelo treinado deixa de ser idêntico entre execuções (reprodutibilidade).
    """
    return pd.read_csv(path, float_precision="round_trip")


def save_splits(splits: dict[str, pd.DataFrame], folder: Path) -> None:
    """Grava {nome: DataFrame} como `<folder>/<nome>.csv` (ex.: train.csv, val.csv, test.csv)."""
    folder.mkdir(parents=True, exist_ok=True)
    for nome, df in splits.items():
        df.to_csv(folder / f"{nome}.csv", index=False)


def load_splits(folder: Path, nomes: tuple[str, ...] = ("train", "val", "test")) -> dict[str, pd.DataFrame]:
    """Lê `<folder>/<nome>.csv` para cada nome pedido."""
    return {nome: read_csv(folder / f"{nome}.csv") for nome in nomes}
