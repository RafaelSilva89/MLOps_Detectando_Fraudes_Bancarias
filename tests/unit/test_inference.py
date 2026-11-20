"""Testes unitários da inferência (FraudPredictor) e do app Flask, com um modelo pequeno treinado na hora.

Não dependem do pipeline ter rodado: o modelo é treinado aqui, em segundos, na base sintética.
"""
import pytest
import xgboost as xgb

from app.main import create_app
from src.config import ALVO
from src.data_validation.validate_data import DataValidationError
from src.feature_engineering.engineer_features import build_features
from src.inference.predict import FraudPredictor

POLITICA = {"limiar": 0.5, "regra_valor": {"corte_valor": 200.0, "limiar_valor": 0.05}, "custo_revisao": 5.0}


@pytest.fixture
def predictor(base_sintetica):
    dados = build_features(base_sintetica)
    modelo = xgb.XGBClassifier(n_estimators=30, max_depth=3, random_state=0).fit(dados.drop(columns=[ALVO]), dados[ALVO])
    return FraudPredictor(modelo, POLITICA, origem="teste")


def test_predict_devolve_probabilidade_e_decisao(predictor, base_sintetica):
    novas = base_sintetica.drop(columns=[ALVO]).head(100)
    resultado = predictor.predict(novas)
    assert list(resultado.columns) == ["probabilidade_fraude", "decisao"]
    assert resultado.index.equals(novas.index)
    assert resultado["probabilidade_fraude"].between(0, 1).all()
    assert set(resultado["decisao"]) <= {"revisar", "liberar"}


def test_predict_aplica_a_regra_de_valor(predictor, base_sintetica):
    novas = base_sintetica.head(200)
    resultado = predictor.predict(novas)
    pela_regra = (novas["hist_saldo"] >= 200) & (resultado["probabilidade_fraude"] >= 0.05)
    assert (resultado.loc[pela_regra, "decisao"] == "revisar").all()


def test_predict_rejeita_entrada_invalida(predictor, base_sintetica):
    with pytest.raises(DataValidationError, match="crt"):
        predictor.predict(base_sintetica.drop(columns=["crt"]))


def test_app_health_e_erros_de_entrada(predictor):
    cliente = create_app(predictor).test_client()
    assert cliente.get("/health").json["limiar"] == 0.5
    assert cliente.post("/predict", json={}).status_code == 400
    assert cliente.post("/upload", data={}, content_type="multipart/form-data").status_code == 400
    assert "Regra de decisão" in cliente.get("/").get_data(as_text=True)
