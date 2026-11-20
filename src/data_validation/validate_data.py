"""Estágio 2 do pipeline: validar a qualidade, a integridade e a consistência dos dados.

Etapa do ciclo MLOps: **Data Validation**.

- Entrada: `data/raw/raw.csv`.
- Saída:   `metrics/data_validation.json` (relatório de cada checagem).
- Se alguma checagem *bloqueante* falhar, o estágio levanta um erro e o pipeline PARA:
  é melhor não treinar do que treinar com dados quebrados ("garbage in, garbage out").

Por que existe? No notebook, as checagens (nulos, tipos, duplicatas) eram feitas "olhando"
as saídas das células. Num pipeline automático ninguém olha: as regras precisam estar no
código. A mesma função `validate_dataframe` também protege a INFERÊNCIA (o app e a CLI
rejeitam arquivos sem as colunas certas), evitando que o modelo receba algo que nunca viu.

Executar: python -m src.data_validation.validate_data
"""
import logging

import numpy as np
import pandas as pd

from src.config import ALVO, COLUNA_TEMPO, COLUNA_VALOR, COLUNAS_BRUTAS, PATHS
from src.utils import read_csv, save_json

logger = logging.getLogger("src.data_validation.validate_data")


class DataValidationError(ValueError):
    """Erro levantado quando uma checagem bloqueante falha."""


def _check(nome: str, ok: bool, detalhe: str, bloqueante: bool = True) -> dict:
    return {"checagem": nome, "ok": bool(ok), "bloqueante": bloqueante, "detalhe": detalhe}


def validate_dataframe(df: pd.DataFrame, require_target: bool = True) -> dict:
    """Aplica as regras de qualidade à base e devolve um relatório.

    Entradas:
        df: base no formato bruto (Timestamp, 28 componentes, hist_saldo e, se exigido, Target).
        require_target: True no treino; False na inferência (transações novas não têm gabarito).
    Saída:
        {'valido': bool, 'linhas': int, 'checagens': [...], 'resumo': {...}}.
        'valido' é False se qualquer checagem bloqueante falhar.
    """
    esperadas = [c for c in COLUNAS_BRUTAS if require_target or c != ALVO]
    faltando = [c for c in esperadas if c not in df.columns]
    checagens = [_check("colunas obrigatórias", not faltando,
                        f"faltando: {faltando}" if faltando else f"{len(esperadas)} colunas presentes")]
    if faltando:   # sem as colunas, as demais regras não fazem sentido
        return {"valido": False, "linhas": len(df), "checagens": checagens, "resumo": {}}

    dados = df[esperadas]
    nao_numericas = [c for c in esperadas if not pd.api.types.is_numeric_dtype(dados[c])]
    checagens.append(_check("tipos numéricos", not nao_numericas,
                            f"não numéricas: {nao_numericas}" if nao_numericas else "todas numéricas"))
    if nao_numericas:
        return {"valido": False, "linhas": len(df), "checagens": checagens, "resumo": {}}

    nulos = int(dados.isna().sum().sum())
    infinitos = int(np.isinf(dados.to_numpy(dtype=float)).sum())
    checagens += [
        _check("base não vazia", len(dados) > 0, f"{len(dados):,} linhas"),
        _check("sem valores nulos", nulos == 0, f"{nulos} nulos"),
        _check("sem valores infinitos", infinitos == 0, f"{infinitos} infinitos"),
        _check("Timestamp >= 0", bool((dados[COLUNA_TEMPO] >= 0).all()),
               f"mínimo = {dados[COLUNA_TEMPO].min()}"),
        _check("valor da transação >= 0", bool((dados[COLUNA_VALOR] >= 0).all()),
               f"mínimo = {dados[COLUNA_VALOR].min()}"),
    ]
    resumo = {"linhas": len(dados), "colunas": dados.shape[1],
              "duplicatas": int(dados.duplicated().sum()),
              "janela_horas": round(float(dados[COLUNA_TEMPO].max()) / 3600, 2)}
    # Duplicatas não bloqueiam: são removidas no pré-processamento. Ficam registradas para auditoria.
    checagens.append(_check("duplicatas (removidas no próximo estágio)", True,
                            f"{resumo['duplicatas']} linhas duplicadas", bloqueante=False))

    if require_target:
        classes = set(dados[ALVO].unique().tolist())
        checagens.append(_check("alvo binário {0, 1}", classes <= {0, 1}, f"classes encontradas: {sorted(classes)}"))
        checagens.append(_check("as duas classes presentes", classes == {0, 1}, f"classes encontradas: {sorted(classes)}"))
        taxa = float(dados[ALVO].mean())
        resumo.update({"fraudes": int(dados[ALVO].sum()), "taxa_fraude": round(taxa, 6)})
        # Alerta (não bloqueia): taxa de fraude muito fora do esperado indica mudança na fonte
        checagens.append(_check("taxa de fraude entre 0,01% e 5%", 0.0001 <= taxa <= 0.05,
                                f"taxa = {taxa:.4%}", bloqueante=False))

    valido = all(c["ok"] for c in checagens if c["bloqueante"])
    return {"valido": valido, "linhas": len(df), "checagens": checagens, "resumo": resumo}


def main() -> None:
    """Valida data/raw/raw.csv, grava o relatório e interrompe o pipeline se houver erro."""
    logger.info(f"Validando {PATHS.raw}")
    relatorio = validate_dataframe(read_csv(PATHS.raw), require_target=True)
    save_json(relatorio, PATHS.metrics_dir / "data_validation.json")
    for c in relatorio["checagens"]:
        nivel = logging.INFO if c["ok"] else (logging.ERROR if c["bloqueante"] else logging.WARNING)
        logger.log(nivel, f"[{'OK' if c['ok'] else 'FALHOU'}] {c['checagem']}: {c['detalhe']}")
    if not relatorio["valido"]:
        raise DataValidationError("A validação dos dados falhou. Veja metrics/data_validation.json")
    logger.info("Validação dos dados concluída: base aprovada")


if __name__ == "__main__":
    main()
