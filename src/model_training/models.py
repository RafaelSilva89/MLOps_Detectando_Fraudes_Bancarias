"""Fábrica dos modelos candidatos e dos espaços de busca de hiperparâmetros.

Etapa do ciclo MLOps: **Model Training** (definição dos modelos).

Por que separar a *definição* dos modelos do *treino*? Assim o treino (train_model.py) não
precisa saber os detalhes de cada algoritmo, e um experimento que muda o balanceamento
(`dvc exp run -S train.balanceamento=smote`) altera só a construção, não o fluxo.

Tudo aqui reproduz o notebook v2 (Fase 03, itens 4.2, 5 e 7.2):
- 4 algoritmos: XGBoost, Random Forest, LightGBM e Regressão Logística;
- balanceamento pelo PESO DA CLASSE (`scale_pos_weight` no XGBoost, `class_weight='balanced'`
  nos demais). No notebook, o SMOTE empatou e foi descartado; ele continua disponível aqui
  só para experimentos, SEMPRE dentro de um Pipeline do imblearn (aplicado só ao treino de
  cada dobra, sem vazamento para a validação);
- a Regressão Logística recebe um StandardScaler dentro do Pipeline (ajustado só no treino);
- espaços de busca do Random Search idênticos aos do notebook, na mesma ordem (a ordem das
  chaves muda o sorteio das combinações).
"""
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as PipelineImb
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BALANCEAMENTOS = ("class_weight", "smote", "none")

# O MLflow 3 grava modelos scikit-learn no formato `skops`, que (ao contrário do pickle) não executa
# código arbitrário ao carregar e só aceita os tipos declarados como confiáveis. Tipos do próprio
# scikit-learn já são confiáveis; abaixo, os das outras bibliotecas que os nossos modelos podem conter.
SKOPS_TRUSTED_TYPES = [
    "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier",
    "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier", "collections.OrderedDict",
    "imblearn.pipeline.Pipeline", "imblearn.over_sampling._smote.base.SMOTE",
    "sklearn.metrics._dist_metrics.EuclideanDistance64", "sklearn.neighbors._kd_tree.KDTree",
    # Árvores do Random Forest: o skops pede confirmação porque um arquivo ADULTERADO poderia ter índices
    # inválidos. Aqui os arquivos são gerados pelo próprio pipeline (origem confiável).
    "sklearn.tree._tree.Tree",
]


def _montar(modelo: BaseEstimator, balanceamento: str, semente: int, escala: bool = False) -> BaseEstimator:
    """Envolve o modelo nos passos necessários: escala (se pedida) e SMOTE (se pedido)."""
    passos = [("escala", StandardScaler())] if escala else []
    if balanceamento == "smote":
        passos.append(("smote", SMOTE(random_state=semente)))
        return PipelineImb(passos + [("modelo", modelo)])
    return Pipeline(passos + [("modelo", modelo)]) if passos else modelo


def build_models(proporcao: float, balanceamento: str, semente: int, n_jobs: int) -> dict[str, BaseEstimator]:
    """Cria os 4 modelos candidatos (ainda não treinados).

    Entradas:
        proporcao: legítimas por fraude NO TREINO (vira o `scale_pos_weight` do XGBoost).
        balanceamento: 'class_weight' (padrão do projeto), 'smote' ou 'none'.
        semente, n_jobs: reprodutibilidade e paralelismo.
    Saída: {nome: estimador}.
    """
    if balanceamento not in BALANCEAMENTOS:
        raise ValueError(f"balanceamento deve ser um de {BALANCEAMENTOS}, recebido: {balanceamento}")
    peso = balanceamento == "class_weight"
    xgb_kw = {"scale_pos_weight": proporcao} if peso else {}
    cw_kw = {"class_weight": "balanced"} if peso else {}
    return {
        "XGBoost": _montar(xgb.XGBClassifier(random_state=semente, n_jobs=n_jobs, **xgb_kw), balanceamento, semente),
        "Random Forest": _montar(RandomForestClassifier(random_state=semente, n_jobs=n_jobs, **cw_kw),
                                 balanceamento, semente),
        "LightGBM": _montar(lgb.LGBMClassifier(random_state=semente, verbose=-1, **cw_kw), balanceamento, semente),
        "Regressão Logística": _montar(LogisticRegression(max_iter=3000, **cw_kw), balanceamento, semente, escala=True),
    }


def search_space(nome: str, proporcao: float, balanceamento: str) -> dict:
    """Espaço de busca do Random Search para o algoritmo `nome` (notebook, item 7.2).

    Distribuições contínuas (`loguniform` para a taxa de aprendizado, `uniform` para frações)
    exploram mais valores que uma grade fixa com o mesmo orçamento.
    """
    pesos = [1, np.sqrt(proporcao), proporcao] if balanceamento == "class_weight" else [1]
    espacos = {
        "XGBoost": {"n_estimators": randint(200, 700), "max_depth": randint(3, 9), "learning_rate": loguniform(0.02, 0.3),
                    "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                    "min_child_weight": randint(1, 10), "scale_pos_weight": pesos},
        "LightGBM": {"n_estimators": randint(200, 700), "num_leaves": randint(15, 127), "learning_rate": loguniform(0.02, 0.3),
                     "subsample": uniform(0.6, 0.4), "subsample_freq": [1], "colsample_bytree": uniform(0.5, 0.5)},
        "Random Forest": {"n_estimators": randint(100, 400), "max_depth": [None, 10, 20, 30],
                          "min_samples_leaf": randint(1, 10), "max_features": ["sqrt", "log2", 0.5]},
        "Regressão Logística": {"C": loguniform(0.001, 100)},
    }
    return espacos[nome]


def prefix_for_pipeline(estimador: BaseEstimator, espaco: dict) -> dict:
    """Se o modelo está dentro de um Pipeline, os hiperparâmetros viram 'modelo__<nome>'."""
    if hasattr(estimador, "steps"):
        return {f"modelo__{k}": v for k, v in espaco.items()}
    return espaco


def final_estimator(modelo: BaseEstimator) -> BaseEstimator:
    """O estimador "de verdade" (último passo do Pipeline, ou o próprio modelo)."""
    return modelo.steps[-1][1] if hasattr(modelo, "steps") else modelo
