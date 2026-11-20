"""Teste da serialização no MLflow: TODO modelo que o pipeline pode escolher precisa ser gravável e
recarregável pelo MLflow (formato skops, com a lista SKOPS_TRUSTED_TYPES).

Este teste nasceu de uma falha real: na base sintética o vencedor era a Regressão Logística (tipos só
do scikit-learn), mas na base real o vencedor é o XGBoost, cujos tipos o skops recusa sem a lista.
Por isso ele testa os 4 algoritmos e as duas estratégias de balanceamento, sem depender de qual vence.
"""
import mlflow
import numpy as np
import pytest

from src.model_training.models import SKOPS_TRUSTED_TYPES, build_models


@pytest.mark.parametrize("balanceamento", ["class_weight", "smote"])
def test_todos_os_modelos_sao_gravados_e_recarregados_pelo_mlflow(tmp_path, monkeypatch, balanceamento):
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("teste_serializacao")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] > 1).astype(int)
    for nome, modelo in build_models(proporcao=5, balanceamento=balanceamento, semente=0, n_jobs=1).items():
        if hasattr(modelo, "set_params") and "n_estimators" in modelo.get_params():
            modelo.set_params(n_estimators=5)
        modelo.fit(X, y)
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(modelo, name="model", skops_trusted_types=SKOPS_TRUSTED_TYPES)
        recarregado = mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
        np.testing.assert_allclose(recarregado.predict_proba(X), modelo.predict_proba(X), err_msg=nome)
