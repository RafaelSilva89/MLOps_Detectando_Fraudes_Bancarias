"""Estágio 5 do pipeline: comparar algoritmos, ajustar hiperparâmetros e treinar o modelo.

Etapas do ciclo MLOps: **Model Training**, **Experiment Tracking** e **Model Serialization**.

- Entradas: `data/processed/{train,val}.csv`, `data/preprocessed/train_val_ids.csv`, params.yaml.
- Saídas:
  - `models/model_validation.joblib`: modelo escolhido, treinado SÓ no treino. Usado pelo próximo
    estágio para escolher o limiar na validação (dados que ele não viu);
  - `models/model.joblib`: o mesmo modelo retreinado em treino + validação (80% da base). É o modelo
    de produção, avaliado uma única vez no teste;
  - `metrics/train.json`: comparação dos algoritmos, validação cruzada e resultado da busca;
  - run no MLflow com parâmetros, métricas e o modelo (+ `artifacts/mlflow_run.json` com o run_id).

Fluxo (notebook v2, Fase 03, itens 5 a 7.6):
1. Treina os 4 algoritmos (com peso da classe) no treino e mede na validação (limiar 0,5).
2. Validação cruzada estratificada de 5 dobras em treino + validação: escolhe o algoritmo pela
   maior PR-AUC média (a validação tem só ~94 fraudes; as dobras dão um resultado mais estável).
3. Random Search (20 combinações, 3 dobras do treino) no algoritmo vencedor, otimizando a PR-AUC.
4. Usa o modelo ajustado só se a PR-AUC da busca superar a do modelo padrão nas MESMAS 3 dobras.
5. Retreina o escolhido em treino + validação: o modelo final.

O TESTE não é lido aqui (só no evaluate_model): nenhuma escolha pode ver o teste.

Executar: python -m src.model_training.train_model   (≈ 40 min com os parâmetros do notebook)
"""
import logging
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

from src.config import ALVO, COLUNAS_MODELO, ID, PATHS, load_params
from src.model_evaluation.metrics import calcular_metricas_completas
from src.model_training.models import SKOPS_TRUSTED_TYPES, build_models, prefix_for_pipeline, search_space
from src.tracking import start_training_run
from src.utils import load_splits, read_csv, save_json

logger = logging.getLogger("src.model_training.train_model")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as 30 variáveis do modelo (X) e o alvo (y)."""
    return df[COLUNAS_MODELO], df[ALVO]


def train_val_in_order(train: pd.DataFrame, val: pd.DataFrame, ordem: pd.Series) -> pd.DataFrame:
    """Junta treino + validação na ordem da 1ª divisão (a mesma do notebook)."""
    return pd.concat([train, val]).set_index(ID).loc[ordem].reset_index()


def compare_models(modelos: dict[str, BaseEstimator], X_train, y_train, X_val, y_val) -> dict:
    """Passo 1: treina cada algoritmo no treino e mede na validação (limiar 0,5).

    Saída: {nome: métricas}, com a PR-AUC no treino para medir o sobreajuste.
    Os modelos de `modelos` ficam treinados (o vencedor é reaproveitado como "modelo padrão").
    """
    resultados = {}
    for nome, modelo in modelos.items():
        t0 = time.time()
        modelo.fit(X_train, y_train)
        proba = modelo.predict_proba(X_val)[:, 1]
        metricas = calcular_metricas_completas(y_val, (proba >= 0.5).astype(int), proba)
        metricas["PR-AUC treino"] = average_precision_score(y_train, modelo.predict_proba(X_train)[:, 1])
        metricas["Tempo de treino (s)"] = round(time.time() - t0, 1)
        resultados[nome] = metricas
        logger.info(f"{nome:<20} validação: PR-AUC {metricas['PR-AUC']:.4f} | recall {metricas['Recall']:.3f} | "
                    f"precisão {metricas['Precisão']:.3f} | treino PR-AUC {metricas['PR-AUC treino']:.4f}")
    return resultados


def cross_validate_models(modelos: dict[str, BaseEstimator], X, y, folds: int, seed: int) -> dict:
    """Passo 2: validação cruzada estratificada; decide o algoritmo pela PR-AUC média.

    Saída: {nome: {PR-AUC média, desvio, recall médio, precisão média}}.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    resultados = {}
    for nome, modelo in modelos.items():
        t0 = time.time()
        r = cross_validate(clone(modelo), X, y, cv=cv, scoring=["average_precision", "recall", "precision"])
        resultados[nome] = {"PR-AUC (média)": float(r["test_average_precision"].mean()),
                            "PR-AUC (desvio)": float(r["test_average_precision"].std()),
                            "Recall (média)": float(r["test_recall"].mean()),
                            "Precisão (média)": float(r["test_precision"].mean()),
                            "Tempo (s)": round(time.time() - t0, 1)}
        logger.info(f"CV {nome:<20} PR-AUC {resultados[nome]['PR-AUC (média)']:.4f} ± {resultados[nome]['PR-AUC (desvio)']:.4f}")
    return resultados


def tune_model(nome: str, modelo_padrao: BaseEstimator, X_train, y_train, proporcao: float, params: dict) -> dict:
    """Passos 3 e 4: Random Search e a regra "ajustado só se a validação cruzada melhorar".

    Entradas: nome e modelo vencedor (já treinado no treino, com hiperparâmetros padrão), treino,
    proporção de legítimas por fraude e o params.yaml completo.
    Saída: {'modelo': estimador escolhido treinado no treino, 'usar_ajustado': bool, ...métricas da busca}.
    """
    seed, tune = params["seed"], params["tune"]
    cv_busca = StratifiedKFold(n_splits=tune["cv_folds"], shuffle=True, random_state=seed)

    # O modelo padrão nas MESMAS dobras da busca: é a referência para saber se o ajuste valeu a pena
    cv_padrao = cross_validate(clone(modelo_padrao), X_train, y_train, cv=cv_busca, scoring="average_precision",
                               return_train_score=True)
    resultado = {"pr_auc_cv_padrao": float(cv_padrao["test_score"].mean()),
                 "pr_auc_cv_padrao_desvio": float(cv_padrao["test_score"].std())}
    if not tune["enabled"]:
        logger.info("Ajuste de hiperparâmetros desligado (tune.enabled = false): usando o modelo padrão")
        return {**resultado, "modelo": modelo_padrao, "usar_ajustado": False, "metodo": "nenhum"}
    if tune["method"] != "random_search":
        raise ValueError("Só o random_search está implementado no pipeline (Grid e Optuna empataram no notebook)")

    espaco = prefix_for_pipeline(modelo_padrao, search_space(nome, proporcao, params["train"]["balanceamento"]))
    busca = RandomizedSearchCV(clone(modelo_padrao), espaco, n_iter=tune["n_iter"], scoring="average_precision",
                               cv=cv_busca, random_state=seed, return_train_score=True)
    t0 = time.time()
    busca.fit(X_train, y_train)
    i = busca.best_index_
    resultado.update({
        "metodo": "Random Search", "pr_auc_cv_busca": float(busca.best_score_),
        "pr_auc_cv_busca_desvio": float(busca.cv_results_["std_test_score"][i]),
        "pr_auc_treino_busca": float(busca.cv_results_["mean_train_score"][i]),
        "combinacoes": len(busca.cv_results_["params"]), "tempo_busca_min": round((time.time() - t0) / 60, 1),
        "melhores_hiperparametros": {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()},
    })
    usar_ajustado = bool(busca.best_score_ > resultado["pr_auc_cv_padrao"])
    logger.info(f"Random Search: PR-AUC CV {busca.best_score_:.4f} x padrão {resultado['pr_auc_cv_padrao']:.4f} "
                f"({resultado['tempo_busca_min']} min) -> {'ajustado' if usar_ajustado else 'padrão'}")
    return {**resultado, "modelo": busca.best_estimator_ if usar_ajustado else modelo_padrao, "usar_ajustado": usar_ajustado}


def train_pipeline(train: pd.DataFrame, val: pd.DataFrame, ordem: pd.Series, params: dict) -> dict:
    """Executa os 5 passos e devolve os dois modelos e o relatório (sem gravar nada em disco).

    Saída: {'modelo_validacao', 'modelo_final', 'relatorio'}.
    """
    seed, cfg = params["seed"], params["train"]
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_temp, y_temp = split_xy(train_val_in_order(train, val, ordem))

    # Peso da classe calculado SÓ com o treino (usar a base toda seria vazamento)
    proporcao = float((y_train == 0).sum() / y_train.sum())
    logger.info(f"Treino: {len(X_train):,} transações, {int(y_train.sum())} fraudes | proporção {proporcao:.1f} legítimas/fraude")

    modelos = build_models(proporcao, cfg["balanceamento"], seed, cfg["n_jobs"])
    modelos = {nome: m for nome, m in modelos.items() if nome in cfg["modelos"]}
    validacao = compare_models(modelos, X_train, y_train, X_val, y_val)
    cv = cross_validate_models(modelos, X_temp, y_temp, cfg["cv_folds_comparacao"], seed)
    melhor = max(cv, key=lambda nome: cv[nome]["PR-AUC (média)"])
    logger.info(f"Melhor algoritmo na validação cruzada: {melhor}")

    ajuste = tune_model(melhor, modelos[melhor], X_train, y_train, proporcao, params)
    modelo_validacao = ajuste.pop("modelo")
    modelo_final = clone(modelo_validacao).fit(X_temp, y_temp)
    nome_escolhido = f"{melhor} ajustado" if ajuste["usar_ajustado"] else f"{melhor} (hiperparâmetros padrão)"
    logger.info(f"Modelo escolhido: {nome_escolhido}; final treinado em {len(X_temp):,} transações")

    relatorio = {"algoritmo": melhor, "modelo_escolhido": nome_escolhido, "proporcao_legitimas_por_fraude": proporcao,
                 "balanceamento": cfg["balanceamento"], "validacao": validacao, "validacao_cruzada": cv, "ajuste": ajuste,
                 "treino_final": {"transacoes": len(X_temp), "fraudes": int(y_temp.sum())}}
    return {"modelo_validacao": modelo_validacao, "modelo_final": modelo_final, "relatorio": relatorio}


def log_to_mlflow(resultado: dict, params: dict, X_exemplo: pd.DataFrame) -> None:
    """Registra no MLflow o que foi usado (parâmetros) e o que saiu (métricas e modelo)."""
    rel = resultado["relatorio"]
    mlflow.log_params({"seed": params["seed"], "balanceamento": rel["balanceamento"], "algoritmo": rel["algoritmo"],
                       "cv_folds_comparacao": params["train"]["cv_folds_comparacao"], "tune_enabled": params["tune"]["enabled"],
                       "tune_n_iter": params["tune"]["n_iter"], "tune_cv_folds": params["tune"]["cv_folds"],
                       "usar_ajustado": rel["ajuste"]["usar_ajustado"]})
    if rel["ajuste"].get("melhores_hiperparametros"):
        mlflow.log_params({f"hp_{k}": v for k, v in rel["ajuste"]["melhores_hiperparametros"].items()})
    for nome, m in rel["validacao_cruzada"].items():
        chave = nome.lower().replace(" ", "_").replace("ã", "a").replace("í", "i")
        mlflow.log_metric(f"cv_pr_auc_{chave}", m["PR-AUC (média)"])
        mlflow.log_metric(f"val_pr_auc_{chave}", rel["validacao"][nome]["PR-AUC"])
    for chave in ("pr_auc_cv_padrao", "pr_auc_cv_busca"):
        if chave in rel["ajuste"]:
            mlflow.log_metric(chave, rel["ajuste"][chave])
    mlflow.set_tag("modelo_escolhido", rel["modelo_escolhido"])
    mlflow.log_artifact(str(PATHS.metrics_dir / "train.json"))
    # O modelo vai para o MLflow com a "assinatura" (nomes e tipos das 30 colunas de entrada)
    modelo = resultado["modelo_final"]
    assinatura = infer_signature(X_exemplo, modelo.predict_proba(X_exemplo)[:, 1])
    mlflow.sklearn.log_model(modelo, name="model", signature=assinatura, input_example=X_exemplo.head(3),
                             skops_trusted_types=SKOPS_TRUSTED_TYPES)


def main() -> None:
    """Orquestra o estágio: lê os dados, treina, grava os modelos e registra no MLflow."""
    params = load_params()
    dados = load_splits(PATHS.processed_dir, ("train", "val"))
    ordem = read_csv(PATHS.train_val_order)[ID]

    with start_training_run():
        resultado = train_pipeline(dados["train"], dados["val"], ordem, params)
        PATHS.model_final.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(resultado["modelo_validacao"], PATHS.model_validation)
        joblib.dump(resultado["modelo_final"], PATHS.model_final)
        save_json(resultado["relatorio"], PATHS.metrics_dir / "train.json")
        log_to_mlflow(resultado, params, dados["val"][COLUNAS_MODELO].head(100))
    logger.info(f"Modelos gravados em {PATHS.model_validation.name} e {PATHS.model_final.name}. Treino concluído")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
