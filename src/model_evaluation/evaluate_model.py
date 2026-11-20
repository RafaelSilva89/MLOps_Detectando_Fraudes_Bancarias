"""Estágio 7 do pipeline: avaliar o modelo final no TESTE, uma única vez.

Etapas do ciclo MLOps: **Model Evaluation** e **documentação do modelo (model card)**.

- Entradas: `models/model.joblib`, `artifacts/decision_policy.json`, `data/processed/test.csv`,
  `metrics/train.json`, params.yaml.
- Saídas:
  - `metrics/evaluation.json`: métricas no teste, resultado em reais, intervalos do bootstrap, SLA;
  - `models/model_card.json`: a "ficha" do modelo (o que é, como usar, limites), como no notebook;
  - as mesmas métricas e a ficha na run do MLflow.

Avaliação (notebook v2, Fase 04, itens 1 e 1.1):
- aplica a regra final (limiar + regra de valor) ao teste e compara com o limiar 0,5;
- métricas para classes desbalanceadas: PR-AUC (principal), recall, precisão, F1, AUC-ROC;
- resultado em reais: valor barrado, custo das revisões, economia líquida;
- bootstrap estratificado (1.000 amostras): com só ~95 fraudes no teste, os números variam muito.
  Os limites inferiores de 2,5% viram o SLA de produção (PR-AUC e recall mínimos), usado pelo
  monitoramento para dizer se o modelo continua funcionando.

Por que o teste fica trancado até aqui? Se qualquer escolha (algoritmo, hiperparâmetros, limiar)
olhasse o teste, ele deixaria de medir o desempenho em dados novos.

Executar: python -m src.model_evaluation.evaluate_model
"""
import logging
from datetime import datetime

import joblib
import mlflow
import numpy as np

from src.config import ALVO, COLUNA_VALOR, COLUNAS_MODELO, PATHS, load_params
from src.model_evaluation.metrics import (aplicar_regra, bootstrap_decisoes, bootstrap_pr_auc,
                                          calcular_metricas_completas, resumo_financeiro)
from src.model_training.train_model import split_xy
from src.tracking import resume_pipeline_run
from src.utils import load_json, load_splits, save_json

logger = logging.getLogger("src.model_evaluation.evaluate_model")


def evaluate(modelo, test, politica: dict, n_bootstrap: int, seed: int) -> dict:
    """Avalia o modelo final no teste com a regra de decisão escolhida na validação.

    Entradas: modelo final, teste processado, política (limiar/regra), nº de amostras do bootstrap, semente.
    Saída: dicionário com métricas, resultado em reais, intervalos de confiança e SLA.
    """
    X_test, y_test = split_xy(test)
    prob = modelo.predict_proba(X_test)[:, 1]
    valores = X_test[COLUNA_VALOR].to_numpy()
    custo = politica["custo_revisao"]
    decisoes = {"Regra final": aplicar_regra(prob, valores, politica["limiar"], politica["regra_valor"]),
                "Só o limiar": aplicar_regra(prob, valores, politica["limiar"]),
                "Limiar 0,50": prob >= 0.5}

    metricas = calcular_metricas_completas(y_test, decisoes["Regra final"].astype(int), prob)
    reais = {nome: resumo_financeiro(y_test, d, valores, custo) for nome, d in decisoes.items()}

    # Bootstrap: mesmas amostras para as duas regras (diferença medida amostra a amostra)
    amostras = bootstrap_decisoes(y_test, valores, {k: decisoes[k] for k in ("Regra final", "Limiar 0,50")},
                                  custo, n=n_bootstrap, semente=seed)
    intervalos = (amostras.groupby("regra")[["Recall", "Precisão", "% do valor barrado", "Economia líquida"]]
                  .quantile([0.025, 0.5, 0.975]))
    final = intervalos.loc["Regra final"]
    diferenca = (amostras.pivot(index="amostra", columns="regra", values="Economia líquida")
                 .eval("`Regra final` - `Limiar 0,50`"))
    pr_auc_boot = bootstrap_pr_auc(y_test, prob, n=n_bootstrap, semente=seed)

    ic = {col: [float(final.loc[0.025, col]), float(final.loc[0.975, col])] for col in final.columns}
    return {
        "transacoes": len(y_test), "fraudes": int(y_test.sum()),
        "metricas_regra_final": metricas,
        "resultado_em_reais": reais,
        "intervalo_95_regra_final": ic,
        "intervalo_95_pr_auc": [float(np.quantile(pr_auc_boot, 0.025)), float(np.quantile(pr_auc_boot, 0.975))],
        "regra_final_melhor_que_0_5_em": float((diferenca > 0).mean()),
        "sla": {"pr_auc_minima": round(float(np.quantile(pr_auc_boot, 0.025)), 3),
                "recall_minimo": round(float(final.loc[0.025, "Recall"]), 3)},
    }


def build_model_card(avaliacao: dict, treino: dict, politica: dict, params: dict) -> dict:
    """Monta a ficha do modelo: tudo o que é preciso saber para usá-lo corretamente."""
    m = avaliacao["metricas_regra_final"]
    ajuste = treino["ajuste"]
    return {
        "data_treino": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "modelo": f"{treino['modelo_escolhido']}, balanceamento = {treino['balanceamento']}",
        "arquivo": "models/model.joblib",
        "treinado_com": f"{treino['treino_final']['transacoes']:,} transações (treino + validação), "
                        f"{treino['treino_final']['fraudes']} fraudes".replace(",", "."),
        "entrada": "CSV no formato bruto: Timestamp, as 28 componentes e hist_saldo (a hora do dia é calculada)",
        "colunas_do_modelo": COLUNAS_MODELO,
        "alvo": f"{ALVO} (1 = fraude, 0 = legítima)",
        "regra_de_decisao": politica,
        "hiperparametros": ajuste.get("melhores_hiperparametros") if ajuste["usar_ajustado"] else "padrão do algoritmo",
        "metodo_de_busca": ajuste["metodo"] if ajuste["usar_ajustado"] else "nenhum (hiperparâmetros padrão)",
        "metricas_teste": {k: round(float(m[k]), 4) for k in ["Precisão", "Recall", "F1", "PR-AUC", "AUC-ROC"]},
        "resultado_teste_reais": {k: round(float(v), 2) for k, v in avaliacao["resultado_em_reais"]["Regra final"].items()},
        "intervalo_95_economia_teste": [round(v, 2) for v in avaliacao["intervalo_95_regra_final"]["Economia líquida"]],
        "monitoramento": {"sla_pr_auc_minima": avaliacao["sla"]["pr_auc_minima"],
                          "sla_recall_minimo": avaliacao["sla"]["recall_minimo"],
                          "psi_alerta": params["monitor"]["psi_alerta"],
                          "variaveis": ["score do modelo"] + params["monitor"]["variaveis"]},
        "avisos": ["28 das 30 variáveis são componentes PCA anônimas: só servem para bases geradas com o mesmo PCA.",
                   "hora_do_dia supõe que a contagem do tempo começa à meia-noite.",
                   "Base de apenas 2 dias: acompanhar o desempenho em produção e retreinar.",
                   f"O limiar depende da premissa de custo de revisão (R$ {politica['custo_revisao']:.0f})."],
    }


def main() -> None:
    """Orquestra o estágio: avalia no teste, grava métricas e ficha e registra no MLflow."""
    params = load_params()
    modelo = joblib.load(PATHS.model_final)
    politica = load_json(PATHS.decision_policy)
    test = load_splits(PATHS.processed_dir, ("test",))["test"]

    avaliacao = evaluate(modelo, test, politica, params["evaluate"]["bootstrap_n"], params["seed"])
    m, r = avaliacao["metricas_regra_final"], avaliacao["resultado_em_reais"]["Regra final"]
    logger.info(f"TESTE (regra final): PR-AUC {m['PR-AUC']:.4f} | recall {m['Recall']:.3f} | precisão {m['Precisão']:.3f} | "
                f"{m['Verdadeiros Positivos']} de {avaliacao['fraudes']} fraudes")
    logger.info(f"TESTE em reais: barrado R$ {r['Valor das fraudes barradas']:,.2f} ({r['% do valor barrado']:.1%}) | "
                f"{r['Transações revisadas']} revisões | economia líquida R$ {r['Economia líquida']:,.2f}")
    logger.info(f"SLA para produção: PR-AUC >= {avaliacao['sla']['pr_auc_minima']} e recall >= {avaliacao['sla']['recall_minimo']}")

    ficha = build_model_card(avaliacao, load_json(PATHS.metrics_dir / "train.json"), politica, params)
    save_json(avaliacao, PATHS.metrics_dir / "evaluation.json")
    save_json(ficha, PATHS.model_card)

    with resume_pipeline_run():
        mlflow.log_metrics({"test_pr_auc": m["PR-AUC"], "test_auc_roc": m["AUC-ROC"], "test_recall": m["Recall"],
                            "test_precision": m["Precisão"], "test_f1": m["F1"],
                            "test_economia_liquida": r["Economia líquida"], "test_pct_valor_barrado": r["% do valor barrado"],
                            "sla_pr_auc_minima": avaliacao["sla"]["pr_auc_minima"],
                            "sla_recall_minimo": avaliacao["sla"]["recall_minimo"]})
        mlflow.log_artifact(str(PATHS.metrics_dir / "evaluation.json"))
        mlflow.log_artifact(str(PATHS.model_card))
    logger.info("Avaliação do modelo concluída")


if __name__ == "__main__":
    main()
