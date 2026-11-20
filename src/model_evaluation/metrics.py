"""Métricas de classificação, custo em reais e a regra de decisão do projeto.

Etapa do ciclo MLOps: **Model Evaluation** (biblioteca de funções, sem estado).

Funções portadas de `src/avaliacao.py` do projeto original (notebook v2), sem os gráficos e
`display()` de notebook. Os nomes foram mantidos para você achar cada uma no notebook.

Quem usa: train_model (métricas da comparação), decision_policy (limiar e regra de valor),
evaluate_model (teste e bootstrap), monitoring (lotes) e inference (aplicar a regra).

Por que a PR-AUC e o custo em reais, e não a acurácia? Com 0,17% de fraudes, um "modelo" que
diz sempre "legítima" acerta 99,83%. A PR-AUC mede se as fraudes ficam no topo da lista (sorteio
= 0,0017) e o custo em reais é o que o negócio decide.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def calcular_metricas_completas(y_true, y_pred, y_proba) -> dict:
    """Todas as métricas de um classificador binário (classe de interesse = 1, fraude).

    Entradas: gabarito, previsão binária (depois do limiar) e probabilidade.
    Saída: dicionário com acurácia, precisão, recall, F1, AUC-ROC, PR-AUC, especificidade e a matriz de confusão.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precisão": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
        "Especificidade": tn / (tn + fp) if (tn + fp) else 0.0,
        "Verdadeiros Positivos": int(tp), "Verdadeiros Negativos": int(tn),
        "Falsos Positivos": int(fp), "Falsos Negativos": int(fn),
    }


def custo_financeiro(y, proba, valores, custo_revisao: float, limiares) -> pd.DataFrame:
    """Custo em reais de cada limiar.

    Premissa de negócio: uma fraude não detectada custa o valor da transação; cada transação
    marcada (fraude ou não) custa uma revisão manual de `custo_revisao` reais.
    Saída: uma linha por limiar, com fraudes detectadas/perdidas, alarmes falsos e 'Custo total'.
    """
    y, proba, valores = np.asarray(y), np.asarray(proba), np.asarray(valores)
    linhas = []
    for limiar in limiares:
        marcado = proba >= limiar
        perdidas = (~marcado) & (y == 1)
        linhas.append({"Limiar": float(limiar), "Fraudes detectadas": int((marcado & (y == 1)).sum()),
                       "Fraudes perdidas": int(perdidas.sum()), "Alarmes falsos": int((marcado & (y == 0)).sum()),
                       "Valor das fraudes perdidas": float(valores[perdidas].sum()),
                       "Custo das revisões": float(marcado.sum() * custo_revisao)})
    tabela = pd.DataFrame(linhas)
    tabela["Custo total"] = tabela["Valor das fraudes perdidas"] + tabela["Custo das revisões"]
    return tabela


def resumo_financeiro(y, previsto, valores, custo_revisao: float) -> dict:
    """Resultado em reais de uma regra de decisão, comparado a não ter modelo (nenhuma fraude barrada).

    Saída: valor total das fraudes, valor barrado, % barrado, custo das revisões, economia líquida
    (barrado - revisões) e quantidade de transações revisadas.
    """
    y, previsto, valores = np.asarray(y), np.asarray(previsto).astype(bool), np.asarray(valores)
    valor_fraudes = float(valores[y == 1].sum())
    evitado = float(valores[previsto & (y == 1)].sum())
    revisoes = float(previsto.sum() * custo_revisao)
    return {"Valor total das fraudes": valor_fraudes, "Valor das fraudes barradas": evitado,
            "% do valor barrado": evitado / valor_fraudes if valor_fraudes else 0.0,
            "Custo das revisões": revisoes, "Economia líquida": evitado - revisoes,
            "Transações revisadas": int(previsto.sum())}


def aplicar_regra(proba, valores, limiar: float, regra_valor: dict | None = None) -> np.ndarray:
    """A REGRA DE DECISÃO do projeto: quais transações vão para revisão manual.

    Revisar se: probabilidade >= limiar
           OU (se houver regra complementar) valor >= corte_valor E probabilidade >= limiar_valor.
    Saída: vetor booleano (True = revisar).
    """
    proba = np.asarray(proba, dtype=float)
    marcado = proba >= limiar
    if regra_valor:
        marcado |= (np.asarray(valores, dtype=float) >= regra_valor["corte_valor"]) & (proba >= regra_valor["limiar_valor"])
    return marcado


def bootstrap_decisoes(y, valores, decisoes: dict, custo_revisao: float, n: int = 1000, semente: int = 42) -> pd.DataFrame:
    """Bootstrap estratificado do resultado de cada regra (intervalos de confiança).

    Sorteia, com reposição, fraudes e legítimas separadamente (mesmas quantidades do original).
    `decisoes` = {nome: vetor booleano de transações revisadas}. Todas as regras usam as MESMAS
    amostras, então a diferença entre elas pode ser medida amostra a amostra.
    Saída: DataFrame longo com Recall, Precisão, % do valor barrado e Economia líquida por amostra e regra.
    """
    y, valores = np.asarray(y), np.asarray(valores)
    rng = np.random.default_rng(semente)
    fraudes, legitimas = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    linhas = []
    for amostra in range(n):
        idx = np.concatenate([rng.choice(fraudes, len(fraudes)), rng.choice(legitimas, len(legitimas))])
        for nome, marcado in decisoes.items():
            marcado_idx = np.asarray(marcado)[idx]
            r = resumo_financeiro(y[idx], marcado_idx, valores[idx], custo_revisao)
            achadas = (marcado_idx & (y[idx] == 1)).sum()
            linhas.append({"amostra": amostra, "regra": nome, "Recall": achadas / len(fraudes),
                           "Precisão": achadas / max(marcado_idx.sum(), 1),
                           "% do valor barrado": r["% do valor barrado"], "Economia líquida": r["Economia líquida"]})
    return pd.DataFrame(linhas)


def bootstrap_pr_auc(y, proba, n: int = 1000, semente: int = 42) -> np.ndarray:
    """PR-AUC em `n` amostras bootstrap estratificadas (o limite inferior de 2,5% vira o SLA)."""
    y, proba = np.asarray(y), np.asarray(proba)
    rng = np.random.default_rng(semente)
    fraudes, legitimas = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    valores = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(fraudes, len(fraudes)), rng.choice(legitimas, len(legitimas))])
        valores.append(average_precision_score(y[idx], proba[idx]))
    return np.array(valores)
