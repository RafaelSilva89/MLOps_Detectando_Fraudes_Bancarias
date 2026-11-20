"""Funções de monitoramento: data drift (PSI e teste K-S) e desempenho por lote contra o SLA.

Etapa do ciclo MLOps: **Monitoring** (biblioteca de funções, sem estado).

Portado de `src/monitoramento.py` e `psi` de `src/avaliacao.py` do projeto original.

Dois tipos de mudança precisam ser acompanhados em produção:
- *Data drift*: a distribuição das ENTRADAS muda. Detectado pelo PSI e pelo K-S, sem precisar do gabarito.
- *Concept drift*: a RELAÇÃO entre entradas e fraude muda. Só aparece nas métricas do modelo
  (PR-AUC, recall) quando o gabarito chega (contestações dos clientes) -> comparação com o SLA.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score

PSI_ATENCAO = 0.10
PSI_MUDANCA = 0.25


def psi(esperado, atual, faixas: int = 10) -> float:
    """Population Stability Index: o quanto a distribuição `atual` se afastou da `esperado`.

    Faixas pelos quantis da distribuição esperada. Leitura: < 0,10 estável; 0,10 a 0,25 atenção;
    > 0,25 mudança relevante (investigar ou retreinar).
    """
    esperado, atual = np.asarray(esperado), np.asarray(atual)
    cortes = np.unique(np.quantile(esperado, np.linspace(0, 1, faixas + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p_esp = np.histogram(esperado, cortes)[0] / len(esperado)
    p_atu = np.histogram(atual, cortes)[0] / len(atual)
    p_esp, p_atu = np.clip(p_esp, 1e-6, None), np.clip(p_atu, 1e-6, None)
    return float(np.sum((p_atu - p_esp) * np.log(p_atu / p_esp)))


def leitura_psi(valor: float) -> str:
    """Estável (até 0,10), Atenção (até 0,25) ou Mudança (acima)."""
    if valor <= PSI_ATENCAO:
        return "Estável"
    return "Atenção" if valor <= PSI_MUDANCA else "Mudança"


def relatorio_drift(referencia, atual, colunas) -> pd.DataFrame:
    """PSI e teste K-S de cada coluna, comparando `atual` com `referencia` (DataFrames ou dicionários).

    Com lotes grandes, o p-valor do K-S fica perto de 0 até para diferenças sem importância
    prática: a leitura usa o PSI e a distância do K-S (de 0 a 1).
    """
    linhas = {}
    for coluna in colunas:
        ks = ks_2samp(referencia[coluna], atual[coluna])
        valor_psi = psi(referencia[coluna], atual[coluna])
        linhas[coluna] = {"PSI": valor_psi, "K-S (distância)": float(ks.statistic), "K-S (p-valor)": float(ks.pvalue),
                          "Leitura": leitura_psi(valor_psi)}
    return pd.DataFrame.from_dict(linhas, orient="index")


def desempenho_lote(real, probabilidade, revisar) -> dict:
    """Métricas de um lote depois que o gabarito chega. Sem fraudes no lote, PR-AUC e recall são NaN."""
    real, revisar = np.asarray(real), np.asarray(revisar, dtype=bool)
    fraudes, revisoes = int(real.sum()), int(revisar.sum())
    achadas = int((revisar & (real == 1)).sum())
    return {"Transações": len(real), "Fraudes": fraudes,
            "PR-AUC": average_precision_score(real, probabilidade) if fraudes else np.nan,
            "Recall": achadas / fraudes if fraudes else np.nan,
            "Precisão da fila": achadas / revisoes if revisoes else 0.0, "Revisões": revisoes}


def situacao_sla(pr_auc: float, recall: float, sla_pr_auc: float, sla_recall: float) -> str:
    """'dentro do SLA', 'abaixo do SLA' ou 'sem fraudes no lote'."""
    if np.isnan(pr_auc) or np.isnan(recall):
        return "sem fraudes no lote"
    return "abaixo do SLA" if pr_auc < sla_pr_auc or recall < sla_recall else "dentro do SLA"


def painel_lotes(real, probabilidade, revisar, lotes, sla_pr_auc: float, sla_recall: float) -> pd.DataFrame:
    """Desempenho de cada lote (ex.: janelas de 4 horas) e a situação em relação ao SLA."""
    real, probabilidade, revisar = np.asarray(real), np.asarray(probabilidade), np.asarray(revisar, dtype=bool)
    lotes = pd.Categorical(lotes)
    linhas = {}
    for lote in lotes.categories:
        no_lote = np.asarray(lotes == lote)
        if no_lote.any():
            linhas[lote] = desempenho_lote(real[no_lote], probabilidade[no_lote], revisar[no_lote])
    painel = pd.DataFrame.from_dict(linhas, orient="index")
    painel["Situação"] = [situacao_sla(p, r, sla_pr_auc, sla_recall) for p, r in zip(painel["PR-AUC"], painel["Recall"])]
    return painel
