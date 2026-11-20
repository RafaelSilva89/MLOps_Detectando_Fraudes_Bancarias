"""Fixtures compartilhadas pelos testes.

Por que uma base SINTÉTICA? Os testes precisam ser rápidos (segundos) e rodar em qualquer máquina,
inclusive no CI, sem baixar 150 MB nem treinar por 40 minutos. A base sintética tem EXATAMENTE o
mesmo esquema da real (31 colunas, mesmos nomes), poucas linhas e fraudes fáceis de separar.
O que se testa é o ENCANAMENTO (os estágios conversam entre si?), não a qualidade do modelo.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import ALVO, COLUNA_TEMPO, COLUNA_VALOR, COMPONENTES_PCA


def make_synthetic_base(n: int = 4000, taxa_fraude: float = 0.04, seed: int = 0) -> pd.DataFrame:
    """Base com o esquema da real: Timestamp (48 h), 28 componentes, hist_saldo e Target.

    As fraudes têm `crt` mais baixo e `cep` mais alto (como na base real), para o modelo ter o que aprender.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < taxa_fraude).astype(int)
    df = pd.DataFrame({COLUNA_TEMPO: np.sort(rng.uniform(0, 48 * 3600, n)).round()})
    for coluna in COMPONENTES_PCA:
        df[coluna] = rng.normal(0, 1, n)
    df["crt"] -= 4 * y
    df["cep"] += 3 * y
    df[COLUNA_VALOR] = rng.exponential(80, n).round(2)
    df[ALVO] = y
    # algumas duplicatas, para o pré-processamento ter o que remover
    return pd.concat([df, df.sample(10, random_state=seed)], ignore_index=True)


@pytest.fixture
def base_sintetica() -> pd.DataFrame:
    return make_synthetic_base()
