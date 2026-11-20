"""Estágio 6 do pipeline: escolher a REGRA DE DECISÃO (limiar e regra de valor) pelo custo em reais.

Etapa do ciclo MLOps: **Model Evaluation / decisão de negócio** (pós-treino, antes do teste).

- Entradas: `models/model_validation.joblib` (treinado só no treino) e `data/processed/val.csv`.
- Saídas:
  - `artifacts/decision_policy.json`: a regra que a inferência aplica (limiar + regra de valor);
  - `metrics/decision_policy.json`: a comparação dos limiares e da grade da regra de valor.

Por que um estágio só para isso? O modelo devolve uma PROBABILIDADE; quem decide "revisar ou
liberar" é a regra. No notebook (Fase 03, item 8), ela é escolhida pelo menor custo em reais:
- fraude não detectada custa o valor da transação; cada revisão custa R$ 5 (params.yaml);
- limiar de custo mínimo entre 0,01 e 0,99 (resultado do notebook: 0,11);
- regra complementar: revisar também transações acima de um valor com probabilidade a partir de
  um limiar menor (grade 4 x 7; resultado do notebook: R$ 200 e 0,5%). Só é adotada se aumentar a
  economia na validação.

Por que na VALIDAÇÃO e com o modelo treinado só no treino? Escolher o limiar no teste, ou com um
modelo que já viu a validação, seria vazamento: o resultado no teste deixaria de ser honesto.
A regra é um ARTEFATO versionado junto com o modelo: trocar o limiar é uma nova versão.

Executar: python -m src.decision_policy.optimize_threshold
"""
import logging

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.config import COLUNA_VALOR, PATHS, load_params
from src.model_evaluation.metrics import aplicar_regra, custo_financeiro, resumo_financeiro
from src.model_training.train_model import split_xy
from src.tracking import resume_pipeline_run
from src.utils import load_splits, save_json

logger = logging.getLogger("src.decision_policy.optimize_threshold")


def grade_limiares(cfg: dict) -> np.ndarray:
    """Limiares candidatos: de `inicio` a `fim`, de `passo` em `passo` (0,01 a 0,99)."""
    return np.round(np.arange(cfg["inicio"], cfg["fim"] + cfg["passo"] / 2, cfg["passo"]), 2)


def escolher_limiar(y, proba, valores, custo_revisao: float, limiares) -> tuple[float, pd.DataFrame]:
    """Limiar de menor custo total em reais. Saída: (limiar, tabela de custo de todos os limiares)."""
    custos = custo_financeiro(y, proba, valores, custo_revisao, limiares)
    return float(custos.loc[custos["Custo total"].idxmin(), "Limiar"]), custos


def escolher_regra_valor(y, proba, valores, limiar: float, custo_revisao: float, cfg: dict) -> tuple[dict | None, pd.DataFrame, float]:
    """Testa a grade (corte de valor x limiar de valor) mantendo o limiar principal.

    Saída: (regra adotada ou None, grade ordenada pela economia, economia só com o limiar).
    """
    linhas = []
    for corte in cfg["cortes_valor"]:
        for limiar_valor in cfg["limiares_valor"]:
            regra = {"corte_valor": corte, "limiar_valor": limiar_valor}
            r = resumo_financeiro(y, aplicar_regra(proba, valores, limiar, regra), valores, custo_revisao)
            linhas.append({**regra, "Revisadas": r["Transações revisadas"], "% do valor barrado": r["% do valor barrado"],
                           "Economia líquida": r["Economia líquida"]})
    grade = pd.DataFrame(linhas).sort_values("Economia líquida", ascending=False)
    economia_simples = resumo_financeiro(y, aplicar_regra(proba, valores, limiar), valores, custo_revisao)["Economia líquida"]
    melhor = grade.iloc[0]
    regra = ({"corte_valor": float(melhor["corte_valor"]), "limiar_valor": float(melhor["limiar_valor"])}
             if melhor["Economia líquida"] > economia_simples else None)
    return regra, grade, economia_simples


def optimize_policy(modelo, val: pd.DataFrame, params: dict) -> dict:
    """Calcula a política de decisão completa na validação.

    Saída: {'politica': {...} (vai para artifacts/), 'relatorio': {...} (vai para metrics/)}.
    """
    cfg = params["decision_policy"]
    X_val, y_val = split_xy(val)
    proba = modelo.predict_proba(X_val)[:, 1]
    valores = X_val[COLUNA_VALOR].to_numpy()
    custo = cfg["custo_revisao"]

    limiar, custos = escolher_limiar(y_val, proba, valores, custo, grade_limiares(cfg["limiares"]))
    regra, grade, economia_simples = escolher_regra_valor(y_val, proba, valores, limiar, custo, cfg["regra_valor"])
    comparacao = {nome: resumo_financeiro(y_val, aplicar_regra(proba, valores, corte, r), valores, custo)
                  for nome, corte, r in [("limiar 0,50", 0.5, None), (f"limiar {limiar:.2f}", limiar, None),
                                         ("regra final", limiar, regra)]}
    politica = {"limiar": limiar, "regra_valor": regra, "custo_revisao": custo,
                "criterio": f"custo mínimo na validação: fraude perdida = valor da transação, revisão = R$ {custo:.0f}"}
    relatorio = {"politica": politica, "economia_so_limiar": economia_simples, "comparacao_validacao": comparacao,
                 "top5_regra_valor": grade.head(5).to_dict(orient="records"),
                 "custo_por_limiar": custos[["Limiar", "Fraudes detectadas", "Alarmes falsos", "Custo total"]]
                 .iloc[::10].to_dict(orient="records")}
    return {"politica": politica, "relatorio": relatorio}


def main() -> None:
    """Orquestra o estágio: carrega o modelo de validação, escolhe a regra e grava/registra."""
    params = load_params()
    modelo = joblib.load(PATHS.model_validation)
    val = load_splits(PATHS.processed_dir, ("val",))["val"]
    resultado = optimize_policy(modelo, val, params)
    politica = resultado["politica"]
    logger.info(f"Limiar de custo mínimo: {politica['limiar']:.2f} | regra de valor: {politica['regra_valor']}")
    for nome, r in resultado["relatorio"]["comparacao_validacao"].items():
        logger.info(f"Validação, {nome:<12}: {r['Transações revisadas']} revisões | economia R$ {r['Economia líquida']:,.2f}")

    save_json(politica, PATHS.decision_policy)
    save_json(resultado["relatorio"], PATHS.metrics_dir / "decision_policy.json")
    with resume_pipeline_run():
        mlflow.log_params({"limiar": politica["limiar"], "custo_revisao": politica["custo_revisao"],
                           "regra_valor_corte": (politica["regra_valor"] or {}).get("corte_valor"),
                           "regra_valor_limiar": (politica["regra_valor"] or {}).get("limiar_valor")})
        mlflow.log_metric("val_economia_regra_final", resultado["relatorio"]["comparacao_validacao"]["regra final"]["Economia líquida"])
        mlflow.log_artifact(str(PATHS.decision_policy))
    logger.info("Política de decisão concluída")


if __name__ == "__main__":
    main()
