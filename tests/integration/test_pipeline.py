"""Teste de INTEGRAÇÃO: executa o pipeline inteiro, de ponta a ponta, numa pasta temporária.

O que garante: os 8 estágios conversam entre si (cada um acha o que o anterior gravou), o MLflow
recebe a run, o registro cria a versão com o alias, a inferência funciona tanto a partir dos
arquivos locais quanto do Model Registry, e o app Flask responde.

Como funciona: FRAUD_PROJECT_ROOT aponta todos os caminhos para `tmp_path`; MLFLOW_TRACKING_URI
aponta para um SQLite temporário. Os dados e o mlflow.db reais não são tocados. O params.yaml é o
do projeto, com orçamentos menores (3 combinações na busca, 50 amostras de bootstrap) para levar segundos.
"""
import json
import shutil

import pytest
import yaml

from src.config import ROOT
from tests.conftest import make_synthetic_base

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def projeto(tmp_path_factory):
    """Cria o projeto temporário, roda todos os estágios e o registro, e devolve a pasta."""
    raiz = tmp_path_factory.mktemp("projeto")
    mp = pytest.MonkeyPatch()
    mp.setenv("FRAUD_PROJECT_ROOT", str(raiz))
    mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(raiz / 'mlflow.db').as_posix()}")
    mp.delenv("DVC_EXP_NAME", raising=False)

    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    params["load_data"]["source"] = "data/external/base.csv"
    params["train"]["cv_folds_comparacao"] = 3
    params["tune"].update({"n_iter": 3, "cv_folds": 3})
    params["evaluate"]["bootstrap_n"] = 50
    (raiz / "params.yaml").write_text(yaml.safe_dump(params, allow_unicode=True), encoding="utf-8")
    (raiz / "data" / "external").mkdir(parents=True)
    make_synthetic_base().to_csv(raiz / "data" / "external" / "base.csv", index=False)

    from src.data_loading import load_data
    from src.data_preprocessing import preprocess_data
    from src.data_validation import validate_data
    from src.decision_policy import optimize_threshold
    from src.feature_engineering import engineer_features
    from src.model_evaluation import evaluate_model
    from src.model_registry import register_model
    from src.model_training import train_model
    from src.monitoring import monitor

    for estagio in (load_data, validate_data, preprocess_data, engineer_features, train_model,
                    optimize_threshold, evaluate_model, monitor, register_model):
        estagio.main()
    yield raiz
    mp.undo()
    shutil.rmtree(raiz, ignore_errors=True)


def test_todos_os_estagios_gravam_suas_saidas(projeto):
    esperados = ["data/raw/raw.csv", "data/preprocessed/train.csv", "data/preprocessed/train_val_ids.csv",
                 "data/processed/test.csv", "models/model.joblib", "models/model_validation.joblib",
                 "models/model_card.json", "artifacts/decision_policy.json", "artifacts/mlflow_run.json",
                 "metrics/data_validation.json", "metrics/train.json", "metrics/decision_policy.json",
                 "metrics/evaluation.json", "metrics/monitoring.json"]
    faltando = [arquivo for arquivo in esperados if not (projeto / arquivo).exists()]
    assert not faltando


def test_split_sem_duplicatas_e_sem_sobreposicao(projeto):
    import pandas as pd

    partes = {nome: pd.read_csv(projeto / "data" / "preprocessed" / f"{nome}.csv") for nome in ("train", "val", "test")}
    ids = [set(df["id_transacao"]) for df in partes.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])   # nenhuma transação em duas partes
    preproc = json.loads((projeto / "metrics" / "preprocessing.json").read_text(encoding="utf-8"))
    assert preproc["duplicatas_removidas"] == 10


def test_avaliacao_e_politica_coerentes(projeto):
    avaliacao = json.loads((projeto / "metrics" / "evaluation.json").read_text(encoding="utf-8"))
    politica = json.loads((projeto / "artifacts" / "decision_policy.json").read_text(encoding="utf-8"))
    assert 0.01 <= politica["limiar"] <= 0.99
    assert avaliacao["metricas_regra_final"]["PR-AUC"] > 0.5      # fraudes sintéticas são fáceis de separar
    assert set(avaliacao["sla"]) == {"pr_auc_minima", "recall_minimo"}


def test_mlflow_registra_run_e_versao_com_alias(projeto):
    from mlflow import MlflowClient

    from src.tracking import setup_mlflow

    setup_mlflow()
    run_id = json.loads((projeto / "artifacts" / "mlflow_run.json").read_text(encoding="utf-8"))["run_id"]
    run = MlflowClient().get_run(run_id)
    assert "test_pr_auc" in run.data.metrics and "limiar" in run.data.params
    versao = MlflowClient().get_model_version_by_alias("fraude_xgboost", "champion")
    assert versao.run_id == run_id


def test_inferencia_local_e_registry_dao_o_mesmo_resultado(projeto):
    import pandas as pd

    from src.inference.predict import FraudPredictor

    novas = pd.read_csv(projeto / "data" / "preprocessed" / "test.csv").head(50)
    local = FraudPredictor.from_local().predict(novas)
    registry = FraudPredictor.from_registry("fraude_xgboost", "champion").predict(novas)
    assert set(local["decisao"]) <= {"revisar", "liberar"}
    pd.testing.assert_frame_equal(local, registry)


def test_app_flask_health_predict_e_upload(projeto):
    import io

    import pandas as pd

    from app.main import create_app
    from src.inference.predict import FraudPredictor

    cliente = create_app(FraudPredictor.from_local()).test_client()
    assert cliente.get("/health").json["status"] == "ok"

    novas = pd.read_csv(projeto / "data" / "preprocessed" / "test.csv").head(5)
    resposta = cliente.post("/predict", json={"transacoes": novas.to_dict(orient="records")})
    assert resposta.status_code == 200 and len(resposta.json["previsoes"]) == 5

    sem_coluna = cliente.post("/predict", json={"transacoes": novas.drop(columns=["crt"]).to_dict(orient="records")})
    assert sem_coluna.status_code == 400 and "crt" in sem_coluna.json["erro"]

    csv = io.BytesIO(novas.to_csv(index=False).encode())
    pagina = cliente.post("/upload", data={"file": (csv, "lote.csv")}, content_type="multipart/form-data")
    assert pagina.status_code == 200 and "transações avaliadas" in pagina.get_data(as_text=True)
