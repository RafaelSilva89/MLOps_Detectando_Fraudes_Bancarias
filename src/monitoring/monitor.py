"""Estágio 8 do pipeline: simular o monitoramento em produção com os dois dias da base.

Etapa do ciclo MLOps: **Monitoring** (data drift, concept drift e SLA).

- Entradas: `data/preprocessed/{train,val,test}.csv` (precisa do Timestamp), os dois modelos,
  `artifacts/decision_policy.json` e `metrics/evaluation.json` (o SLA).
- Saída: `metrics/monitoring.json` (+ registro na run do MLflow).

O que faz (notebook v2, Fase 04 item 1.2 e Fase 05 item 3):
1. **Data drift, dia 2 x dia 1:** PSI e K-S do score do modelo e das variáveis monitoradas.
   (No notebook: `lim_crt` muda muito entre os dias, PSI ≈ 1,05, e `cep` pede atenção.)
2. **Validação temporal:** o mesmo modelo (mesmos hiperparâmetros) treinado nas primeiras 36 h e
   aplicado às últimas 12 h, simulando "prever o futuro". Nada é escolhido aqui: é conferência.
3. **Lotes de 4 h contra o SLA:** as últimas 12 h divididas em lotes, como se o gabarito de cada
   lote chegasse depois. Cada lote é comparado ao SLA definido no evaluate (concept drift).

Em produção real, este estágio rodaria todo dia sobre as transações novas (por exemplo, agendado
no Airflow). Aqui, com só 2 dias de dados, ele simula esse acompanhamento.

Executar: python -m src.monitoring.monitor
"""
import logging

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.config import ALVO, COLUNA_TEMPO, COLUNA_VALOR, COLUNAS_MODELO, ID, PATHS, load_params
from src.feature_engineering.engineer_features import build_features
from src.model_evaluation.metrics import aplicar_regra, resumo_financeiro
from src.monitoring.drift import painel_lotes, psi, relatorio_drift
from src.tracking import resume_pipeline_run
from src.utils import load_json, load_splits, save_json

logger = logging.getLogger("src.monitoring.monitor")


def full_base() -> pd.DataFrame:
    """Base completa sem duplicatas, na ordem original (junta as três partes pelo id_transacao)."""
    partes = load_splits(PATHS.preprocessed_dir)
    return pd.concat(partes.values()).sort_values(ID).reset_index(drop=True)


def drift_between_days(base: pd.DataFrame, modelo_final, variaveis: list[str]) -> pd.DataFrame:
    """Passo 1: PSI e K-S do dia 2 em relação ao dia 1 (variáveis + score do modelo)."""
    X = build_features(base)
    dia_2 = (base[COLUNA_TEMPO] >= 24 * 3600).to_numpy()
    score = modelo_final.predict_proba(X[COLUNAS_MODELO])[:, 1]
    dia_1_valores = {c: X.loc[~dia_2, c].to_numpy() for c in variaveis} | {"score do modelo": score[~dia_2]}
    dia_2_valores = {c: X.loc[dia_2, c].to_numpy() for c in variaveis} | {"score do modelo": score[dia_2]}
    return relatorio_drift(dia_1_valores, dia_2_valores, list(dia_1_valores))


def temporal_validation(base: pd.DataFrame, modelo_escolhido, politica: dict, sla: dict, cfg: dict) -> dict:
    """Passos 2 e 3: treina com o passado (primeiras N horas), avalia o futuro e o divide em lotes."""
    X, y = build_features(base)[COLUNAS_MODELO], base[ALVO]
    horas = base[COLUNA_TEMPO] / 3600
    passado, futuro = (horas < cfg["horas_treino_temporal"]).to_numpy(), (horas >= cfg["horas_treino_temporal"]).to_numpy()

    modelo_temporal = clone(modelo_escolhido).fit(X[passado], y[passado])
    prob = modelo_temporal.predict_proba(X[futuro])[:, 1]
    valores = X.loc[futuro, COLUNA_VALOR].to_numpy()
    decisao = aplicar_regra(prob, valores, politica["limiar"], politica["regra_valor"])
    y_fut = y[futuro].to_numpy()
    resumo = {"transacoes": int(futuro.sum()), "fraudes": int(y_fut.sum()),
              "PR-AUC": average_precision_score(y_fut, prob), "Recall": recall_score(y_fut, decisao),
              "Precisão": precision_score(y_fut, decisao, zero_division=0),
              "% do valor barrado": resumo_financeiro(y_fut, decisao, valores, politica["custo_revisao"])["% do valor barrado"]}

    # Lotes de `horas_lote` horas nas horas "futuras"
    inicio_lote = cfg["horas_treino_temporal"] + ((horas[futuro] - cfg["horas_treino_temporal"]) // cfg["horas_lote"]) * cfg["horas_lote"]
    rotulos = inicio_lote.astype(int).map(lambda h: f"{h} a {h + cfg['horas_lote']} h").to_numpy()
    painel = painel_lotes(y_fut, prob, decisao, rotulos, sla["pr_auc_minima"], sla["recall_minimo"])
    # PSI de cada variável monitorada em cada lote, contra o passado (a hora do dia fica de fora:
    # um lote de poucas horas cobre só parte do dia e sempre "mudaria")
    for coluna in [c for c in cfg["variaveis"] if c != "hora_do_dia"]:
        painel[f"PSI {coluna}"] = [psi(X.loc[passado, coluna], X.loc[futuro, coluna][rotulos == lote]) for lote in painel.index]
    return {"validacao_temporal": resumo, "lotes": painel}


def main() -> None:
    """Orquestra o estágio: drift entre os dias, validação temporal e lotes contra o SLA."""
    cfg = load_params()["monitor"]
    base = full_base()
    politica = load_json(PATHS.decision_policy)
    sla = load_json(PATHS.metrics_dir / "evaluation.json")["sla"]

    drift = drift_between_days(base, joblib.load(PATHS.model_final), cfg["variaveis"])
    for var, linha in drift.iterrows():
        logger.info(f"Drift dia 2 x dia 1 | {var:<16} PSI {linha['PSI']:.4f} | K-S {linha['K-S (distância)']:.3f} | {linha['Leitura']}")

    temporal = temporal_validation(base, joblib.load(PATHS.model_validation), politica, sla, cfg)
    t = temporal["validacao_temporal"]
    logger.info(f"Validação temporal (a partir de {cfg['horas_treino_temporal']} h): PR-AUC {t['PR-AUC']:.3f} | "
                f"recall {t['Recall']:.3f} | precisão {t['Precisão']:.3f}")
    for lote, linha in temporal["lotes"].iterrows():
        logger.info(f"Lote {lote}: PR-AUC {linha['PR-AUC']:.3f} | recall {linha['Recall']:.3f} | {linha['Situação']}")

    alertas_psi = [v for v, linha in drift.iterrows() if linha["PSI"] > cfg["psi_alerta"]]
    lotes_abaixo = int((temporal["lotes"]["Situação"] == "abaixo do SLA").sum())
    relatorio = {"drift_dia2_x_dia1": drift.to_dict(orient="index"), "validacao_temporal": t,
                 "lotes": temporal["lotes"].replace({np.nan: None}).to_dict(orient="index"),
                 "alertas": {"variaveis_com_psi_acima_do_alerta": alertas_psi, "lotes_abaixo_do_sla": lotes_abaixo,
                             "recomendacao": ("investigar a origem do drift antes de retreinar" if alertas_psi else "estável")
                             + ("; retreinar se 2 lotes seguidos ficarem abaixo do SLA" if lotes_abaixo else "")}}
    save_json(relatorio, PATHS.metrics_dir / "monitoring.json")
    with resume_pipeline_run():
        mlflow.log_metrics({"temporal_pr_auc": t["PR-AUC"], "temporal_recall": t["Recall"], "temporal_precisao": t["Precisão"],
                            "lotes_abaixo_sla": lotes_abaixo, "psi_score_dia2": float(drift.loc["score do modelo", "PSI"])})
        mlflow.log_artifact(str(PATHS.metrics_dir / "monitoring.json"))
    logger.info("Monitoramento concluído")


if __name__ == "__main__":
    main()
