"""Testes unitários das métricas, do custo em reais, da regra de decisão e do monitoramento.

Vários vêm do projeto original (tests/test_funcoes.py), adaptados aos novos módulos.
"""
import numpy as np
import pandas as pd
import pytest

from src.decision_policy.optimize_threshold import escolher_limiar, escolher_regra_valor, grade_limiares
from src.model_evaluation.metrics import (aplicar_regra, bootstrap_decisoes, bootstrap_pr_auc,
                                          calcular_metricas_completas, custo_financeiro, resumo_financeiro)
from src.model_training.models import build_models, prefix_for_pipeline, search_space
from src.monitoring.drift import desempenho_lote, painel_lotes, psi, relatorio_drift


# ---------- métricas ----------
def test_metricas_da_classe_fraude():
    m = calcular_metricas_completas([0, 0, 1, 1], [0, 1, 1, 0], [0.1, 0.6, 0.9, 0.4])
    assert m["Precisão"] == pytest.approx(0.5) and m["Recall"] == pytest.approx(0.5)
    assert m["Falsos Positivos"] == 1 and m["Falsos Negativos"] == 1


def test_custo_financeiro_soma_fraudes_perdidas_e_revisoes():
    tabela = custo_financeiro([1, 1, 0, 0], np.array([0.9, 0.2, 0.6, 0.1]), [100.0, 50.0, 10.0, 10.0],
                              custo_revisao=5, limiares=[0.5])
    linha = tabela.iloc[0]
    # marcadas: 0,9 (fraude) e 0,6 (legítima) -> 2 revisões = R$ 10; perdida: a fraude de R$ 50
    assert linha["Fraudes perdidas"] == 1 and linha["Alarmes falsos"] == 1
    assert linha["Custo total"] == pytest.approx(60)


def test_resumo_financeiro_economia_liquida():
    r = resumo_financeiro([1, 1, 0], [1, 0, 1], [100.0, 50.0, 10.0], custo_revisao=5)
    assert r["Valor das fraudes barradas"] == pytest.approx(100)
    assert r["% do valor barrado"] == pytest.approx(100 / 150)
    assert r["Economia líquida"] == pytest.approx(90)   # 100 barrados - 2 revisões x R$ 5


def test_aplicar_regra_com_regra_complementar_de_valor():
    proba = np.array([0.5, 0.01, 0.01, 0.001])
    valores = np.array([10.0, 300.0, 50.0, 300.0])
    assert aplicar_regra(proba, valores, 0.11).tolist() == [True, False, False, False]
    regra = {"corte_valor": 200, "limiar_valor": 0.005}
    assert aplicar_regra(proba, valores, 0.11, regra).tolist() == [True, True, False, False]


def test_bootstrap_e_reprodutivel_e_mantem_as_fraudes():
    y = np.array([1] * 10 + [0] * 90)
    marcado = np.array([True] * 8 + [False] * 2 + [True] * 5 + [False] * 85)
    a = bootstrap_decisoes(y, np.ones(100), {"r": marcado}, 1, n=50, semente=3)
    b = bootstrap_decisoes(y, np.ones(100), {"r": marcado}, 1, n=50, semente=3)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 50 and a["Recall"].between(0, 1).all()
    pr = bootstrap_pr_auc(y, np.r_[np.full(10, 0.9), np.full(90, 0.1)], n=20)
    assert pr.min() == pytest.approx(1.0)   # separação perfeita em toda amostra


# ---------- política de decisão ----------
def test_grade_de_limiares_vai_de_0_01_a_0_99():
    grade = grade_limiares({"inicio": 0.01, "fim": 0.99, "passo": 0.01})
    assert len(grade) == 99 and grade[0] == 0.01 and grade[-1] == 0.99


def test_limiar_de_custo_minimo():
    # 2 fraudes caras com probabilidade 0,3 e 0,8; legítimas com 0,1 e 0,2
    y, proba = np.array([1, 1, 0, 0]), np.array([0.3, 0.8, 0.1, 0.2])
    limiar, _ = escolher_limiar(y, proba, [1000.0, 1000.0, 5.0, 5.0], custo_revisao=5, limiares=[0.15, 0.25, 0.5])
    assert limiar == 0.25   # pega as duas fraudes sem revisar as legítimas


def test_regra_de_valor_so_e_adotada_se_aumentar_a_economia():
    y = np.array([1, 0, 0])
    proba = np.array([0.02, 0.5, 0.01])
    valores = np.array([900.0, 10.0, 20.0])
    cfg = {"cortes_valor": [500], "limiares_valor": [0.01]}
    regra, _, _ = escolher_regra_valor(y, proba, valores, limiar=0.11, custo_revisao=5, cfg=cfg)
    assert regra == {"corte_valor": 500.0, "limiar_valor": 0.01}      # a fraude de R$ 900 compensa
    regra_ruim, _, _ = escolher_regra_valor(np.array([0, 0, 0]), proba, valores, 0.11, 5, cfg)
    assert regra_ruim is None


# ---------- modelos ----------
def test_build_models_aplica_o_peso_da_classe():
    modelos = build_models(proporcao=600, balanceamento="class_weight", semente=42, n_jobs=1)
    assert set(modelos) == {"XGBoost", "Random Forest", "LightGBM", "Regressão Logística"}
    assert modelos["XGBoost"].get_params()["scale_pos_weight"] == 600
    assert modelos["Random Forest"].get_params()["class_weight"] == "balanced"
    assert modelos["Regressão Logística"].named_steps["escala"] is not None   # scaler dentro do Pipeline


def test_smote_fica_dentro_do_pipeline_e_sem_peso():
    modelos = build_models(proporcao=600, balanceamento="smote", semente=42, n_jobs=1)
    assert "smote" in modelos["XGBoost"].named_steps
    assert modelos["XGBoost"].named_steps["modelo"].get_params()["scale_pos_weight"] is None
    espaco = prefix_for_pipeline(modelos["XGBoost"], search_space("XGBoost", 600, "smote"))
    assert "modelo__max_depth" in espaco and espaco["modelo__scale_pos_weight"] == [1]


def test_balanceamento_invalido_gera_erro():
    with pytest.raises(ValueError):
        build_models(600, "oversampling", 42, 1)


# ---------- monitoramento ----------
def test_psi_zero_para_mesma_distribuicao_e_alto_para_deslocada():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 20_000)
    assert psi(base, rng.normal(0, 1, 20_000)) < 0.01
    assert psi(base, rng.normal(1.5, 1, 20_000)) > 0.25


def test_relatorio_drift_separa_estavel_de_mudanca():
    rng = np.random.default_rng(4)
    referencia = pd.DataFrame({"igual": rng.normal(size=5000), "deslocada": rng.normal(size=5000)})
    atual = pd.DataFrame({"igual": rng.normal(size=5000), "deslocada": rng.normal(1, 1, size=5000)})
    drift = relatorio_drift(referencia, atual, ["igual", "deslocada"])
    assert drift.loc["igual", "Leitura"] == "Estável" and drift.loc["deslocada", "Leitura"] == "Mudança"


def test_desempenho_lote_bate_com_o_sklearn():
    from sklearn.metrics import average_precision_score, precision_score, recall_score

    real = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    prob = np.array([0.1, 0.4, 0.8, 0.3, 0.7, 0.9, 0.05, 0.2])
    revisar = prob >= 0.5
    r = desempenho_lote(real, prob, revisar)
    assert r["Recall"] == recall_score(real, revisar) and r["Precisão da fila"] == precision_score(real, revisar)
    assert r["PR-AUC"] == average_precision_score(real, prob)


def test_painel_lotes_marca_o_sla_e_o_lote_sem_fraudes():
    real = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.1, 0.2, 0.1, 0.6, 0.2, 0.1, 0.3, 0.2])
    lotes = ["a"] * 4 + ["b"] * 3 + ["c"] * 3
    painel = painel_lotes(real, prob, prob >= 0.5, lotes, sla_pr_auc=0.7, sla_recall=0.7)
    assert list(painel["Situação"]) == ["dentro do SLA", "abaixo do SLA", "sem fraudes no lote"]
