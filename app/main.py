"""App Flask que serve o modelo: interface web (upload de CSV) e API JSON.

Etapa do ciclo MLOps: **Deployment / Serving** (o modelo sendo usado por pessoas e sistemas).

Rotas:
- GET  /         -> página para enviar um CSV de transações;
- POST /upload   -> recebe o CSV, devolve a tabela com probabilidade e decisão (HTML);
- POST /predict  -> API: recebe JSON {"transacoes": [{...}, ...]} e devolve as previsões em JSON;
- GET  /health   -> "estou vivo?" + qual modelo está carregado (usado por Docker/orquestradores).

Diferença para o projeto-exemplo: lá, `ModelService` reaplicava imputer, scaler e encoder à mão.
Aqui o app NÃO repete nenhuma transformação: ele delega tudo ao `FraudPredictor`
(src/inference/predict.py), que usa as mesmas funções do pipeline. Uma única fonte da verdade.

`create_app` é uma "fábrica": cria o app com o modelo indicado. Os testes passam um modelo de
teste; o gunicorn chama `create_app()` e carrega o modelo do params.yaml (local ou registry).

Executar localmente:  python -m app.main            (http://localhost:5001)
Docker (gunicorn):    gunicorn --bind=0.0.0.0:5001 "app.main:create_app()"
"""
import io
import logging

import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.data_validation.validate_data import DataValidationError
from src.inference.predict import FraudPredictor

logger = logging.getLogger("app.main")

LIMITE_LINHAS_TELA = 500   # a página mostra no máximo estas linhas (as demais ficam no download pela API)


def create_app(predictor: FraudPredictor | None = None) -> Flask:
    """Cria o app Flask com o modelo informado (ou o configurado no params.yaml)."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB por upload
    predictor = predictor or FraudPredictor.from_config()
    logger.info(f"App iniciado com o modelo ({predictor.origem})")

    @app.get("/health")
    def health():
        return jsonify(status="ok", modelo=predictor.origem, limiar=predictor.policy["limiar"],
                       regra_valor=predictor.policy["regra_valor"])

    @app.get("/")
    def index():
        return render_template("index.html", politica=predictor.policy)

    @app.post("/upload")
    def upload():
        arquivo = request.files.get("file")
        if arquivo is None or not arquivo.filename.lower().endswith(".csv"):
            return render_template("index.html", politica=predictor.policy, erro="Envie um arquivo .csv"), 400
        try:
            transacoes = pd.read_csv(io.BytesIO(arquivo.read()))
            resultado = transacoes.join(predictor.predict(transacoes))
        except DataValidationError as erro:
            return render_template("index.html", politica=predictor.policy, erro=str(erro)), 400
        except Exception as erro:   # qualquer outro problema vira mensagem, e o detalhe vai para o log
            logger.error(f"Erro ao processar {arquivo.filename}: {erro}", exc_info=True)
            return render_template("index.html", politica=predictor.policy, erro=f"Erro ao processar o arquivo: {erro}"), 500
        colunas = [c for c in ("hist_saldo", "Target", "probabilidade_fraude", "decisao") if c in resultado]
        resumo = {"total": len(resultado), "revisar": int((resultado["decisao"] == "revisar").sum())}
        tabela = (resultado[colunas].sort_values("probabilidade_fraude", ascending=False).head(LIMITE_LINHAS_TELA)
                  .to_html(classes="tabela", float_format=lambda v: f"{v:.4f}"))
        return render_template("index.html", politica=predictor.policy, resumo=resumo, tabela=tabela)

    @app.post("/predict")
    def predict():
        corpo = request.get_json(silent=True) or {}
        if not isinstance(corpo.get("transacoes"), list) or not corpo["transacoes"]:
            return jsonify(erro='Envie {"transacoes": [ {...}, ... ]} com ao menos uma transação'), 400
        try:
            previsoes = predictor.predict(pd.DataFrame(corpo["transacoes"]))
        except DataValidationError as erro:
            return jsonify(erro=str(erro)), 400
        return jsonify(modelo=predictor.origem, previsoes=previsoes.to_dict(orient="records"))

    return app


def main() -> None:
    """Servidor de desenvolvimento do Flask (no Windows; em produção, use o gunicorn no Docker)."""
    create_app().run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
