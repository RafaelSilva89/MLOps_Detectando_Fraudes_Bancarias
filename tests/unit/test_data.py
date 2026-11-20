"""Testes unitários das etapas de dados: validação, pré-processamento e feature engineering.

Um teste unitário verifica UMA função isolada, com uma entrada pequena e conhecida, e confere
a saída. Se uma mudança futura quebrar a regra, o teste falha e aponta exatamente onde.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import ALVO, COLUNAS_BRUTAS, COLUNAS_MODELO, COLUNAS_ORIGINAIS, COMPONENTES_PCA, ID
from src.data_preprocessing.preprocess_data import remover_duplicatas, split_data
from src.data_validation.validate_data import validate_dataframe
from src.feature_engineering.engineer_features import build_features, hora_do_dia


# ---------- esquema ----------
def test_dicionario_cobre_as_31_colunas_e_28_componentes():
    assert len(COLUNAS_ORIGINAIS) == 31 and len(COMPONENTES_PCA) == 28
    assert COLUNAS_ORIGINAIS["hist_saldo"] == "Amount" and COLUNAS_ORIGINAIS["valor"] == "V10"
    assert len(COLUNAS_MODELO) == 30 and "Timestamp" not in COLUNAS_MODELO


# ---------- validação ----------
def _checagem(relatorio, nome):
    return next(c for c in relatorio["checagens"] if c["checagem"] == nome)


def test_validacao_aprova_base_correta(base_sintetica):
    relatorio = validate_dataframe(base_sintetica)
    assert relatorio["valido"]
    assert relatorio["resumo"]["duplicatas"] == 10


def test_validacao_reprova_coluna_faltando(base_sintetica):
    relatorio = validate_dataframe(base_sintetica.drop(columns=["crt"]))
    assert not relatorio["valido"] and "crt" in relatorio["checagens"][0]["detalhe"]


def test_validacao_reprova_nulos_e_valores_negativos(base_sintetica):
    df = base_sintetica.copy()
    df.loc[0, "cep"] = np.nan
    df.loc[1, "hist_saldo"] = -5
    relatorio = validate_dataframe(df)
    assert not relatorio["valido"]
    assert not _checagem(relatorio, "sem valores nulos")["ok"]
    assert not _checagem(relatorio, "valor da transação >= 0")["ok"]


def test_validacao_reprova_alvo_fora_de_0_1(base_sintetica):
    df = base_sintetica.copy()
    df.loc[0, ALVO] = 2
    assert not validate_dataframe(df)["valido"]


def test_validacao_na_inferencia_nao_exige_o_alvo(base_sintetica):
    assert validate_dataframe(base_sintetica.drop(columns=[ALVO]), require_target=False)["valido"]


def test_validacao_reprova_coluna_de_texto(base_sintetica):
    df = base_sintetica.astype({"ip": str})
    assert not validate_dataframe(df)["valido"]


# ---------- pré-processamento ----------
def test_remover_duplicatas_conta_as_linhas_removidas(base_sintetica):
    sem, removidas = remover_duplicatas(base_sintetica)
    assert removidas == 10 and len(sem) == len(base_sintetica) - 10
    assert list(sem.index) == list(range(len(sem)))


def test_split_60_20_20_estratificado_e_sem_sobreposicao(base_sintetica):
    df, _ = remover_duplicatas(base_sintetica)
    df.insert(0, ID, df.index)
    partes = split_data(df, test_size=0.2, val_size=0.25, seed=42)
    tamanhos = {nome: len(p) / len(df) for nome, p in partes.items() if nome != "train_val"}
    assert tamanhos == pytest.approx({"train": 0.6, "val": 0.2, "test": 0.2}, abs=0.01)
    taxa = df[ALVO].mean()
    for nome in ("train", "val", "test"):
        assert partes[nome][ALVO].mean() == pytest.approx(taxa, abs=0.01)   # estratificação
    ids = [set(partes[n][ID]) for n in ("train", "val", "test")]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set(partes["train_val"][ID]) == ids[0] | ids[1]


def test_split_e_reprodutivel_com_a_mesma_semente(base_sintetica):
    a = split_data(base_sintetica, 0.2, 0.25, seed=7)["test"].index.tolist()
    b = split_data(base_sintetica, 0.2, 0.25, seed=7)["test"].index.tolist()
    assert a == b


# ---------- feature engineering ----------
def test_hora_do_dia_volta_a_zero_no_segundo_dia():
    assert hora_do_dia(pd.Series([0, 3600, 86400, 90000])).tolist() == pytest.approx([0, 1, 0, 1])


def test_build_features_troca_timestamp_pela_hora_e_mantem_a_ordem(base_sintetica):
    X = build_features(base_sintetica)
    assert list(X.columns) == COLUNAS_MODELO + [ALVO]
    assert X["hora_do_dia"].between(0, 24).all()
    sem_alvo = build_features(base_sintetica[[c for c in COLUNAS_BRUTAS if c != ALVO]])
    assert list(sem_alvo.columns) == COLUNAS_MODELO   # na inferência, sem Target
