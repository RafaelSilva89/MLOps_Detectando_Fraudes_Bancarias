"""Configuração central do projeto: caminhos, nomes de colunas e leitura do params.yaml.

Etapa do ciclo MLOps: **Configuration Management**.

Por que existe? No notebook, caminhos e constantes ficavam espalhados pelas células
(`sys.path.append('..')`, `'Model/modelo_fraude.pkl'`...). Aqui eles ficam num lugar só:
qualquer módulo importa `from src.config import PATHS, load_params` e ninguém escreve
um caminho "na mão". Para mudar uma pasta, muda-se uma linha.

Quem usa: todos os estágios do pipeline, a inferência, o app e os testes.
"""
import os
from pathlib import Path
from typing import Any

import yaml

# Raiz do projeto = pasta acima de src/. Usar caminhos absolutos a partir daqui faz o
# código funcionar de qualquer diretório (terminal, pytest, Airflow, Docker).
ROOT = Path(__file__).resolve().parents[1]


class Paths:
    """Todos os arquivos que o pipeline lê ou grava. Cada estágio lê o que o anterior gravou.

    A raiz pode ser trocada pela variável de ambiente FRAUD_PROJECT_ROOT. Os testes de integração
    usam isso para rodar o pipeline inteiro numa pasta temporária, sem tocar nos dados reais.
    """

    @property
    def root(self) -> Path:
        return Path(os.getenv("FRAUD_PROJECT_ROOT", ROOT))

    # Configuração
    params = property(lambda self: self.root / "params.yaml")
    # Dados (cada pasta = uma etapa do fluxo de dados)
    raw = property(lambda self: self.root / "data" / "raw" / "raw.csv")
    preprocessed_dir = property(lambda self: self.root / "data" / "preprocessed")
    processed_dir = property(lambda self: self.root / "data" / "processed")
    train_val_order = property(lambda self: self.root / "data" / "preprocessed" / "train_val_ids.csv")
    # Modelos e artefatos de suporte
    model_validation = property(lambda self: self.root / "models" / "model_validation.joblib")  # só treino: escolhe o limiar
    model_final = property(lambda self: self.root / "models" / "model.joblib")                 # treino + validação: produção
    model_card = property(lambda self: self.root / "models" / "model_card.json")
    decision_policy = property(lambda self: self.root / "artifacts" / "decision_policy.json")
    mlflow_run = property(lambda self: self.root / "artifacts" / "mlflow_run.json")
    # Métricas (lidas pelo DVC: `dvc metrics show`)
    metrics_dir = property(lambda self: self.root / "metrics")


PATHS = Paths()

# ------------------------------------------------------------------------------
# Esquema dos dados
# ------------------------------------------------------------------------------
# A base é o conjunto público "Credit Card Fraud Detection" (ULB/Kaggle) com as colunas
# renomeadas. 28 nomes "de negócio" são, na verdade, componentes PCA anônimas (V1..V28).
COLUNAS_ORIGINAIS = {
    "Timestamp": "Time", "pais": "V1", "cidade": "V2", "bairro": "V3", "cep": "V4", "ip": "V5", "dia": "V6",
    "hora": "V7", "minuto": "V8", "so": "V9", "valor": "V10", "antecedentes": "V11", "reclamacoes": "V12",
    "qtd_trans": "V13", "crt": "V14", "limite_global": "V15", "tipo_crt": "V16", "estabelecimento": "V17",
    "contas": "V18", "emprestimos": "V19", "navegador": "V20", "android": "V21", "ios": "V22", "compras": "V23",
    "hist_navegacao": "V24", "relacionamento": "V25", "ind_seguranca": "V26", "tempo_transacao": "V27",
    "lim_crt": "V28", "hist_saldo": "Amount", "Target": "Class",
}
COLUNAS_BRUTAS = list(COLUNAS_ORIGINAIS)                 # 31 colunas da base original
COMPONENTES_PCA = [c for c, o in COLUNAS_ORIGINAIS.items() if o.startswith("V")]
ALVO = "Target"
ID = "id_transacao"          # identificador criado no pré-processamento (posição na base sem duplicatas)
COLUNA_TEMPO = "Timestamp"
COLUNA_VALOR = "hist_saldo"                              # valor real da transação (R$)
# Colunas que o modelo recebe: as 28 componentes, o valor e a hora do dia (criada no feature engineering)
COLUNAS_MODELO = COMPONENTES_PCA + [COLUNA_VALOR, "hora_do_dia"]


def load_params(path: Path | None = None) -> dict[str, Any]:
    """Lê o params.yaml.

    Entrada: caminho do YAML (padrão: params.yaml da raiz).
    Saída: dicionário com todas as seções (seed, train, tune, ...).
    """
    with open(path or PATHS.params, encoding="utf-8") as f:
        return yaml.safe_load(f)
