# notebooks/: pesquisa x produção

O notebook que deu origem a este projeto **não foi copiado** para cá. Ele continua no projeto original:

`Ciencia de dados/Detectando_Fraudes_em_Transferencias_Bancarias/Notebooks/Detectando Fraudes em Transferências Bancárias - v2.ipynb`

## Por que separar?

| Notebook (pesquisa) | Pipeline (produção) |
|---|---|
| Explora: EDA, 6 hipóteses, testes estatísticos, gráficos | Executa: só os passos que geram o modelo |
| Compara alternativas: SMOTE x peso, Grid x Random x Optuna, calibração, peso pelo valor | Implementa a **alternativa escolhida** (peso da classe, Random Search, sem calibração) |
| Roda célula a célula, na ordem que a pessoa quiser | Roda sempre na mesma ordem (`dvc repro`), sem intervenção |
| O resultado depende do estado do kernel | O resultado depende só do código, dos dados e do `params.yaml` |

O notebook responde **"o que fazer e por quê"**. O pipeline responde **"como fazer de novo, igual, sempre"**.

Quando uma pergunta nova surgir (testar outro algoritmo, outra variável), o caminho é:
1. explorar no notebook;
2. se der certo, levar a decisão para o código em `src/` e para o `params.yaml`;
3. comparar com `dvc exp run` e registrar no MLflow.

Para usar um notebook *junto* com este projeto, crie-o nesta pasta e importe as funções de `src/`
(por exemplo, `from src.model_evaluation.metrics import resumo_financeiro`), em vez de copiar código.
