"""Inferência: prever se transações novas devem ser revisadas, de forma independente do treino.

Etapa do ciclo MLOps: **Inference / Predict** (servir o modelo).

- Entradas: um CSV (ou DataFrame) no formato BRUTO: Timestamp, as 28 componentes e hist_saldo.
  O modelo treinado e a política de decisão (de `models/` + `artifacts/` ou do Model Registry).
- Saída: para cada transação, a probabilidade de fraude e a decisão ("revisar" ou "liberar").

A classe `FraudPredictor` junta TUDO o que é preciso para prever, na mesma ordem do treino:
1. valida a entrada com a mesma função do estágio de validação (colunas, tipos, nulos...);
2. cria as features com a MESMA função do estágio de feature engineering (`build_features`);
3. calcula a probabilidade com o modelo final;
4. aplica a regra de decisão escolhida na validação (limiar + regra de valor).
Reaproveitar as funções do pipeline evita o *training-serving skew* (o modelo receber em
produção dados transformados de um jeito diferente do treino).

Quem usa: a CLI abaixo, o app Flask (`app/main.py`) e os testes.

Executar (CLI):
    python -m src.inference.predict --input transacoes.csv --output previsoes.csv
    python -m src.inference.predict --input transacoes.csv --source registry   # modelo "champion" do MLflow
"""
import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.config import COLUNA_VALOR, COLUNAS_MODELO, PATHS, load_params
from src.data_validation.validate_data import DataValidationError, validate_dataframe
from src.feature_engineering.engineer_features import build_features
from src.model_evaluation.metrics import aplicar_regra
from src.utils import load_json

logger = logging.getLogger("src.inference.predict")


class FraudPredictor:
    """Modelo + regra de decisão, prontos para prever transações novas."""

    def __init__(self, model, policy: dict, origem: str = "") -> None:
        self.model = model
        self.policy = policy
        self.origem = origem

    @classmethod
    def from_local(cls, model_path: Path | None = None, policy_path: Path | None = None) -> "FraudPredictor":
        """Carrega os arquivos gerados pelo pipeline (models/model.joblib e artifacts/decision_policy.json)."""
        model_path, policy_path = model_path or PATHS.model_final, policy_path or PATHS.decision_policy
        logger.info(f"Carregando o modelo local {model_path}")
        return cls(joblib.load(model_path), load_json(policy_path), origem=f"local: {model_path.name}")

    @classmethod
    def from_registry(cls, name: str, alias: str) -> "FraudPredictor":
        """Carrega a versão apontada pelo alias no MLflow Model Registry e a política da mesma run."""
        import mlflow
        from mlflow import MlflowClient

        from src.tracking import setup_mlflow

        setup_mlflow()
        versao = MlflowClient().get_model_version_by_alias(name, alias)
        logger.info(f"Carregando '{name}' v{versao.version} (alias '{alias}') do Model Registry")
        model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
        caminho = mlflow.artifacts.download_artifacts(run_id=versao.run_id, artifact_path="decision_policy.json")
        return cls(model, load_json(Path(caminho)), origem=f"registry: {name} v{versao.version}")

    @classmethod
    def from_config(cls) -> "FraudPredictor":
        """Escolhe a origem pelo params.yaml (inference.model_source: local | registry)."""
        cfg = load_params()["inference"]
        if cfg["model_source"] == "registry":
            return cls.from_registry(cfg["registered_model_name"], cfg["alias"])
        return cls.from_local()

    def predict(self, transacoes: pd.DataFrame) -> pd.DataFrame:
        """Prevê a probabilidade de fraude e a decisão para cada transação.

        Entrada: DataFrame no formato bruto (colunas extras, como Target, são ignoradas).
        Saída: DataFrame com 'probabilidade_fraude' e 'decisao', no mesmo índice da entrada.
        Levanta DataValidationError se a entrada não passar na validação.
        """
        relatorio = validate_dataframe(transacoes, require_target=False)
        if not relatorio["valido"]:
            erros = [f"{c['checagem']} ({c['detalhe']})" for c in relatorio["checagens"] if c["bloqueante"] and not c["ok"]]
            raise DataValidationError("Entrada inválida: " + "; ".join(erros))
        X = build_features(transacoes)[COLUNAS_MODELO]
        proba = self.model.predict_proba(X)[:, 1]
        revisar = aplicar_regra(proba, X[COLUNA_VALOR], self.policy["limiar"], self.policy["regra_valor"])
        return pd.DataFrame({"probabilidade_fraude": proba.round(6),
                             "decisao": pd.Series(revisar, index=transacoes.index).map({True: "revisar", False: "liberar"})},
                            index=transacoes.index)


def main() -> None:
    """CLI: lê um CSV, prevê e grava (ou mostra) o resultado."""
    parser = argparse.ArgumentParser(description="Prevê fraudes em transações (formato bruto da base).")
    parser.add_argument("--input", required=True, type=Path, help="CSV com Timestamp, as 28 componentes e hist_saldo")
    parser.add_argument("--output", type=Path, help="CSV de saída (padrão: mostra no terminal)")
    parser.add_argument("--source", choices=["local", "registry"], help="origem do modelo (padrão: params.yaml)")
    args = parser.parse_args()

    if args.source == "local":
        predictor = FraudPredictor.from_local()
    elif args.source == "registry":
        cfg = load_params()["inference"]
        predictor = FraudPredictor.from_registry(cfg["registered_model_name"], cfg["alias"])
    else:
        predictor = FraudPredictor.from_config()

    transacoes = pd.read_csv(args.input)
    resultado = transacoes.join(predictor.predict(transacoes))
    revisar = int((resultado["decisao"] == "revisar").sum())
    logger.info(f"{len(resultado)} transações avaliadas ({predictor.origem}); {revisar} enviadas para revisão")
    if args.output:
        resultado.to_csv(args.output, index=False)
        logger.info(f"Previsões gravadas em {args.output}")
    else:
        print(resultado[["hist_saldo", "probabilidade_fraude", "decisao"]].to_string())


if __name__ == "__main__":
    main()
