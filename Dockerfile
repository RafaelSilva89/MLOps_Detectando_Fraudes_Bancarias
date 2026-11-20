# ==============================================================================
# Dockerfile: imagem do APP que serve o modelo (Deployment).
#
# Uma imagem Docker empacota código + dependências + modelo num "pacote" que roda igual
# em qualquer máquina (seu PC, um servidor, a nuvem). Acaba o "na minha máquina funciona".
#
#   docker build -t fraude-api .
#   docker run -p 5001:5001 fraude-api        ->  http://localhost:5001
#
# O modelo vem de models/ + artifacts/ (gerados pelo `dvc repro`): rode o pipeline antes do build.
# ==============================================================================
FROM python:3.13-slim

# libgomp1: biblioteca de paralelismo (OpenMP) exigida por XGBoost e LightGBM no Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /fraud_detection

# 1) Dependências primeiro: esta camada fica em cache e só é refeita se o pyproject.toml mudar
COPY pyproject.toml ./
COPY src/__init__.py src/__init__.py
COPY app/__init__.py app/__init__.py
RUN pip install --no-cache-dir .

# 2) Código, configuração, modelo e política de decisão. O pacote é instalado em modo editável
#    (-e) para que `src` seja lido desta pasta, junto com params.yaml, models/ e artifacts/.
COPY . .
RUN pip install --no-cache-dir --no-deps -e .
ENV FRAUD_PROJECT_ROOT=/fraud_detection

EXPOSE 5001

# gunicorn = servidor de produção; "app.main:create_app()" chama a fábrica do app Flask
CMD ["gunicorn", "--bind=0.0.0.0:5001", "--workers=2", "--timeout=120", "app.main:create_app()"]
