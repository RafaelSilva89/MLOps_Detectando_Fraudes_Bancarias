"""Conteúdo das Fases 8 a 12 (Parte 3 · Modelo; início da Parte 4 · Produção). Formato: ver conteudo_fases_1.py."""

P3 = "Parte 3 · Modelo"
P4 = "Parte 4 · Produção"

FASES = [
# ============================================================================================= F8
{
"n": 8, "grupo": P3, "curto": "Treinamento", "entregavel": "models.py + train_model.py → model_validation.joblib e model.joblib",
"titulo": "Treinar os algoritmos do notebook, escolher o melhor e gravar dois modelos",
"lede": "O treino replica exatamente o que o notebook decidiu: mesmos algoritmos, balanceamento, sementes e critério de escolha. Ele grava dois modelos: um treinado só no treino (para escolher o limiar sem vazamento) e o final, retreinado com treino + validação.",
"pastas": "src/model_training/ · models/ · metrics/train.json",
"ref": "<p><b>Implementado e validado.</b> Na validação, PR-AUC de 0,8913 (XGBoost), 0,8770 (RF), 0,8555 (LightGBM), 0,7876 (Reg. Logística); CV de 5 dobras: XGBoost 0,8482 ± 0,0276; Random Search 0,8410 × 0,8199 do padrão → modelo ajustado. Hiperparâmetros idênticos ao notebook. Tempo: ~44 min (4 núcleos, 8 GB).</p>",
"A": [
  ("p", "Implementar o estágio de treino separando <b>a definição dos modelos</b> (<code>models.py</code>) do <b>fluxo de treino</b> (<code>train_model.py</code>), e gravar os modelos treinados."),
  ("p", "<b>Problema que resolve:</b> treino irreprodutível e decisões de modelagem escondidas em células. Aqui cada decisão é um parâmetro, e o mesmo comando dá o mesmo modelo."),
],
"B": [
  ("Validação cruzada estratificada", "stratified k-fold cross-validation", "Divide o treino em k dobras com a mesma proporção de classes; mais estável que uma validação única."),
  ("Busca de hiperparâmetros", "hyperparameter tuning (random search)", "Sorteia combinações e escolhe a melhor pela métrica na CV."),
  ("Balanceamento", "class weighting / SMOTE", "Compensa classes raras: dando mais peso a elas (peso da classe) ou criando exemplos sintéticos (SMOTE, sempre dentro do Pipeline)."),
  ("Serialização do modelo", "model serialization", "Gravar o modelo treinado em arquivo (<code>joblib</code>) para usá-lo sem retreinar."),
  ("Semente", "random seed", "Fixa o sorteio: mesma semente, mesmo resultado."),
],
"C": [
  ("ol", [
    "Em <code>models.py</code>, escreva uma função que constrói os candidatos (com o balanceamento como parâmetro) e outra com os espaços de busca, copiados do notebook na mesma ordem.",
    "Em <code>train_model.py</code>, escreva o fluxo: comparar na validação → CV para escolher o algoritmo → busca no vencedor → regra \"ajustado só se a CV melhorar\" → retreino final.",
    "Calcule qualquer coisa que aprende com os dados (ex.: <code>scale_pos_weight</code>) só com o treino.",
    "Grave os dois modelos e um <code>metrics/train.json</code> com a comparação.",
    "Rode e compare cada número com o inventário da fase 2.",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "Algoritmos candidatos · estratégia de balanceamento · métrica de escolha · número de dobras e combinações (orçamento de tempo) · regra para aceitar o modelo ajustado."),
],
"D": [
  ("src/model_training/models.py", "<code>build_models</code>, <code>search_space</code>, <code>SKOPS_TRUSTED_TYPES</code>.", "E"),
  ("src/model_training/train_model.py", "<code>compare_models</code>, <code>cross_validate_models</code>, <code>tune_model</code>, <code>train_pipeline</code>; <code>main()</code> grava e registra no MLflow.", "RE"),
  ("models/model_validation.joblib", "Modelo escolhido, treinado só no treino. Usado pela fase 9 para escolher o limiar.", "R"),
  ("models/model.joblib", "Modelo final (treino + validação). É o de produção.", "R"),
  ("metrics/train.json", "Comparação, CV, busca e hiperparâmetros escolhidos.", "R"),
],
"E": [
  ("code", "models.py · candidatos com o balanceamento como parâmetro", """def build_models(proporcao, balanceamento, semente, n_jobs):
    peso = balanceamento == "class_weight"
    xgb_kw = {"scale_pos_weight": proporcao} if peso else {}
    cw_kw = {"class_weight": "balanced"} if peso else {}
    return {
        "XGBoost": _montar(xgb.XGBClassifier(random_state=semente, n_jobs=n_jobs, **xgb_kw), balanceamento, semente),
        "Random Forest": _montar(RandomForestClassifier(random_state=semente, n_jobs=n_jobs, **cw_kw), balanceamento, semente),
        "LightGBM": _montar(lgb.LGBMClassifier(random_state=semente, verbose=-1, **cw_kw), balanceamento, semente),
        "Regressão Logística": _montar(LogisticRegression(max_iter=3000, **cw_kw), balanceamento, semente, escala=True),
    }"""),
  ("code", "train_model.py · o fluxo (independe do algoritmo)", """proporcao = float((y_train == 0).sum() / y_train.sum())        # só o treino!
modelos = build_models(proporcao, cfg["balanceamento"], seed, cfg["n_jobs"])
validacao = compare_models(modelos, X_train, y_train, X_val, y_val)
cv = cross_validate_models(modelos, X_temp, y_temp, cfg["cv_folds_comparacao"], seed)
melhor = max(cv, key=lambda nome: cv[nome]["PR-AUC (média)"])
ajuste = tune_model(melhor, modelos[melhor], X_train, y_train, proporcao, params)
modelo_validacao = ajuste.pop("modelo")                          # treinado só no treino
modelo_final = clone(modelo_validacao).fit(X_temp, y_temp)       # treino + validação"""),
  ("cmd", "Rodar o estágio (≈ 44 min com os parâmetros do notebook)", """$ python -m src.model_training.train_model"""),
  ("note", "key", "Em outros problemas", "<b>Regressão</b>: troque os candidatos (ex.: <code>XGBRegressor</code>, <code>Ridge</code>), a estratificação (use <code>KFold</code> ou estratifique por faixas do alvo) e o <code>scoring</code> (<code>neg_mean_absolute_error</code>). O fluxo comparar → CV → busca → retreino continua igual."),
],
"F": [
  ("out", "Resposta esperada (projeto de referência, trechos do log)", """XGBoost              validação: PR-AUC 0.8913 | recall 0.851 | precisão 0.930 | treino PR-AUC 1.0000
Random Forest        validação: PR-AUC 0.8770 | recall 0.840 | precisão 0.919 | treino PR-AUC 1.0000
CV XGBoost              PR-AUC 0.8482 ± 0.0276
Melhor algoritmo na validação cruzada: XGBoost
Random Search: PR-AUC CV 0.8410 x padrão 0.8199 (30.4 min) -> ajustado
Modelo escolhido: XGBoost ajustado; final treinado em 226,980 transações"""),
  ("p", "Compare com os números do notebook (fase 2). Diferenças pequenas podem vir de versões de bibliotecas; diferenças grandes indicam lógica diferente (ordem das linhas, semente, espaço de busca)."),
  ("note", "warn", "Para iterar mais rápido", "Experimente com orçamento menor sem editar o arquivo: <code>python -m dvc exp run -S tune.n_iter=3 -S train.cv_folds_comparacao=3</code> (fase 13)."),
],
"G": ["Os dois modelos e o <code>metrics/train.json</code> são gravados.",
      "Os números da validação e da CV batem com o notebook.",
      "Rodar de novo com os mesmos parâmetros dá os mesmos números."],
"H": "O modelo devolve probabilidades; falta decidir a partir de qual valor agir e medir o resultado honestamente. É a <b>Fase 9</b>.",
"check": [
  ("Escrever <code>models.py</code>", "Candidatos e espaços como no notebook", "<code>models.py</code>", "<code>pytest tests/unit -k build_models</code>", "Peso da classe e Pipeline corretos"),
  ("Escrever <code>train_model.py</code>", "Fluxo reproduzível", "<code>train_model.py</code>", "Rodar o estágio", "Termina sem erro"),
  ("Gravar os dois modelos", "Separar escolha do limiar e produção", "<code>models/*.joblib</code>", "Arquivos existem", "Dois modelos gravados"),
  ("Conferir com o notebook", "Lógica preservada", "<code>metrics/train.json</code>", "Comparar PR-AUC e hiperparâmetros", "Números iguais"),
],
},
# ============================================================================================= F9
{
"n": 9, "grupo": P3, "curto": "Avaliação", "entregavel": "Regra de decisão (validação) + avaliação única no teste + SLA + model card",
"titulo": "Escolher a regra de decisão na validação e avaliar uma única vez no teste",
"lede": "Avaliar é duas coisas: decidir o que fazer com a previsão (a regra) e medir, sem trapaça, o quanto ela funciona. A regra é escolhida na validação; o teste é aberto uma única vez; o bootstrap mede a incerteza e vira o SLA de produção.",
"pastas": "src/decision_policy/ · src/model_evaluation/ · artifacts/ · metrics/ · models/model_card.json",
"ref": "<p><b>Implementado e validado.</b> Limiar de custo mínimo <b>0,11</b> e regra de valor <b>R$ 200 / 0,005</b> (validação). Teste: PR-AUC <b>0,8169</b>, 78 de 95 fraudes, 132 revisões, economia de <b>R$ 11.494,24</b> (IC 95% R$ 6.255 a R$ 17.940). SLA: PR-AUC ≥ 0,736 e recall ≥ 0,747. Tudo idêntico ao notebook.</p>",
"A": [
  ("p", "Criar dois estágios: <b>decision_policy</b> (escolhe a regra na validação) e <b>evaluate</b> (aplica modelo final + regra ao teste, uma vez, e documenta o modelo)."),
  ("p", "<b>Problemas que resolvem:</b> métrica errada para o negócio, limiar escolhido no teste (vazamento) e um modelo sem ficha (ninguém sabe como usá-lo nem os seus limites)."),
],
"B": [
  ("Regra de decisão / limiar", "decision threshold / policy", "Converte a probabilidade em ação. Escolhida pelo custo, não pelo 0,5 padrão."),
  ("Conjunto de teste intocado", "held-out test set", "Usado só no fim, uma vez. Se influenciar uma escolha, deixa de medir dados novos."),
  ("Bootstrap", "bootstrap resampling", "Reamostra o teste com reposição para obter intervalos de confiança."),
  ("SLA / nível de serviço", "service level agreement", "Desempenho mínimo aceitável em produção (aqui, o limite inferior do bootstrap)."),
  ("Ficha do modelo", "model card", "O que é o modelo, dados, métricas, regra, limites e avisos de uso."),
],
"C": [
  ("ol", [
    "Com o modelo treinado só no treino, calcule as probabilidades na validação.",
    "Para cada limiar candidato, calcule o custo de negócio; escolha o mínimo. (Regras extras, como a de valor, só entram se melhorarem a validação.)",
    "Grave a regra como artefato (<code>artifacts/decision_policy.json</code>).",
    "No estágio <code>evaluate</code>, aplique o modelo final + regra ao teste: métricas, resultado em reais, comparação com o limiar 0,5.",
    "Rode o bootstrap; derive o SLA dos limites inferiores; grave métricas e a ficha do modelo.",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "Critério do limiar (custo, F1, meta de recall) · premissas de custo · quantas amostras de bootstrap · como o SLA é definido."),
],
"D": [
  ("src/decision_policy/optimize_threshold.py", "<code>escolher_limiar</code>, <code>escolher_regra_valor</code>, <code>optimize_policy</code>.", "E"),
  ("src/model_evaluation/metrics.py", "Biblioteca: métricas, custo em reais, <code>aplicar_regra</code>, bootstrap.", "RE"),
  ("src/model_evaluation/evaluate_model.py", "<code>evaluate</code> e <code>build_model_card</code>.", "RE"),
  ("artifacts/decision_policy.json", "A regra que a inferência aplica. O modelo sozinho não decide.", "R"),
  ("metrics/evaluation.json · models/model_card.json", "Resultado no teste, intervalos, SLA e a ficha do modelo.", "R"),
],
"E": [
  ("code", "optimize_threshold.py · o limiar de menor custo", """def escolher_limiar(y, proba, valores, custo_revisao, limiares):
    custos = custo_financeiro(y, proba, valores, custo_revisao, limiares)
    # custo total = valor das fraudes perdidas + transações revisadas × custo da revisão
    return float(custos.loc[custos["Custo total"].idxmin(), "Limiar"]), custos"""),
  ("code", "metrics.py · a regra de decisão (usada na avaliação E na inferência)", """def aplicar_regra(proba, valores, limiar, regra_valor=None):
    marcado = np.asarray(proba, dtype=float) >= limiar
    if regra_valor:
        marcado |= (np.asarray(valores) >= regra_valor["corte_valor"]) & (proba >= regra_valor["limiar_valor"])
    return marcado"""),
  ("cmd", "Rodar os dois estágios", """$ python -m src.decision_policy.optimize_threshold
$ python -m src.model_evaluation.evaluate_model"""),
  ("note", "key", "Em outros problemas", "<b>Regressão</b>: não há limiar; a \"regra\" pode ser uma faixa de tolerância ou um ajuste por viés. Avalie MAE/RMSE no teste, erro por faixa do alvo e o bootstrap do MAE para o SLA. O princípio é o mesmo: escolha na validação, meça uma vez no teste."),
],
"F": [
  ("out", "Resposta esperada (projeto de referência)", """Limiar de custo mínimo: 0.11 | regra de valor: {'corte_valor': 200.0, 'limiar_valor': 0.005}
Validação, regra final : 131 revisões | economia R$ 8,037.66
TESTE (regra final): PR-AUC 0.8169 | recall 0.821 | precisão 0.591 | 78 de 95 fraudes
TESTE em reais: barrado R$ 12,154.24 (82.3%) | 132 revisões | economia líquida R$ 11,494.24
SLA para produção: PR-AUC >= 0.736 e recall >= 0.747"""),
  ("ul", ["Procure quem lê a parte de teste: <code>grep -rn \"\\\"test\\\"\" src/</code>. Só <code>evaluate_model.py</code> deve lê-la para medir (o <code>preprocess_data.py</code> só a grava, e o <code>monitor.py</code> usa a base inteira por tempo, sem escolher nada).",
          "Os testes <code>test_limiar_de_custo_minimo</code> e <code>test_regra_de_valor_so_e_adotada_se_aumentar_a_economia</code> passam."]),
],
"G": ["A regra é escolhida na validação e gravada como artefato.",
      "Só o estágio de avaliação lê o teste para medir; nenhuma escolha usa o teste.",
      "Existem intervalos de confiança, o SLA e a ficha do modelo.",
      "Os números batem com o notebook."],
"H": "Até aqui, os números ficam em arquivos JSON. A <b>Fase 10</b> registra cada execução no MLflow, para comparar versões e saber qual configuração gerou cada resultado.",
"check": [
  ("Escolher a regra na validação", "Decisão pelo negócio, sem vazamento", "<code>decision_policy.json</code>", "Rodar o estágio", "Limiar igual ao do notebook"),
  ("Avaliar no teste uma vez", "Medida honesta", "<code>evaluation.json</code>", "Rodar o <code>evaluate</code>", "PR-AUC e R$ iguais ao notebook"),
  ("Calcular bootstrap e SLA", "Saber a incerteza e o mínimo aceitável", "<code>intervalo_95_*</code>, <code>sla</code>", "Ler o JSON", "SLA definido"),
  ("Gerar a ficha do modelo", "Documentar uso e limites", "<code>models/model_card.json</code>", "Abrir o arquivo", "Ficha com regra, métricas e avisos"),
],
},
# ============================================================================================= F10
{
"n": 10, "grupo": P3, "curto": "Rastreamento", "entregavel": "Uma run do MLflow por pipeline, com parâmetros, métricas e modelo",
"titulo": "Registrar cada execução no MLflow: o que foi usado e o que saiu",
"lede": "Cada execução do pipeline vira uma run no MLflow com parâmetros, métricas, artefatos e o modelo. O treino abre a run; os estágios seguintes a reabrem, para que uma run represente um modelo completo.",
"pastas": "src/tracking.py · mlflow.db · mlruns/ · artifacts/mlflow_run.json",
"ref": "<p><b>Implementado e validado.</b> Experimento <code>deteccao_fraudes</code> num MLflow local com SQLite. A run <code>63249e6a…</code> tem 19 parâmetros (15 do treino e 4 da regra de decisão), as métricas de CV, validação, teste, SLA e monitoramento, o modelo com assinatura e os JSONs. Experimentos do DVC viram runs filhas (tag <code>dvc_exp</code>).</p>",
"A": [
  ("p", "Instrumentar o pipeline com <b>rastreamento de experimentos</b>: uma run por execução, com tudo que permite comparar e reproduzir."),
  ("p", "<b>Problema que resolve:</b> \"qual configuração deu aquele resultado?\". Sem rastreio, comparar execuções exige abrir JSONs de commits diferentes."),
],
"B": [
  ("Rastreamento de experimentos", "experiment tracking", "Registrar parâmetros, métricas e artefatos de cada execução."),
  ("Experimento × run", "experiment / run", "O experimento agrupa; a run é uma execução."),
  ("Servidor de rastreio", "tracking server / backend store", "Onde as runs ficam. Localmente, um arquivo SQLite; em equipe, um servidor (ou o DagsHub)."),
  ("Assinatura do modelo", "model signature", "Nomes e tipos das colunas de entrada e saída, gravados junto do modelo."),
  ("Run aninhada", "nested / child run", "Run filha de outra; usada para agrupar experimentos do DVC."),
],
"C": [
  ("ol", [
    "Escreva <code>setup_mlflow()</code>: define o servidor (variável <code>MLFLOW_TRACKING_URI</code> no <code>.env</code>) e o experimento.",
    "No treino, abra a run, grave o <code>run_id</code> num arquivo e registre parâmetros, métricas e o modelo (com assinatura).",
    "Nos estágios seguintes, reabra a mesma run pelo <code>run_id</code> e acrescente o que produzem.",
    "Quando rodar dentro de <code>dvc exp run</code>, crie a run como filha de uma run \"mãe\".",
    "Abra o <code>mlflow ui</code> e compare duas runs.",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "Servidor local (SQLite) ou remoto · nome do experimento · quais métricas registrar (as que você usaria para comparar versões)."),
],
"D": [
  ("src/tracking.py", "<code>setup_mlflow</code>, <code>start_training_run</code>, <code>resume_pipeline_run</code>.", "R"),
  (".env · MLFLOW_TRACKING_URI", "Onde está o servidor. Fora do código e do Git.", "R"),
  ("artifacts/mlflow_run.json", "O <code>run_id</code> da execução atual, saída do treino.", "R"),
  ("mlflow.db · mlruns/", "Banco e arquivos do MLflow local (fora do Git).", "R"),
],
"E": [
  ("code", "tracking.py · uma run por pipeline", """@contextmanager
def start_training_run():
    experiment_id = setup_mlflow()
    extra = {}
    if os.getenv("DVC_EXP_NAME"):                       # dentro de dvc exp run
        extra = {"parent_run_id": _parent_run_id_dvc(experiment_id),
                 "run_name": os.getenv("DVC_EXP_NAME"), "nested": True}
    with mlflow.start_run(**extra) as run:
        save_json({"run_id": run.info.run_id}, PATHS.mlflow_run)
        yield run

@contextmanager
def resume_pipeline_run():                              # usado por decision_policy, evaluate, monitor
    setup_mlflow()
    with mlflow.start_run(run_id=load_json(PATHS.mlflow_run)["run_id"]) as run:
        yield run"""),
  ("cmd", "Abrir a interface do MLflow", """$ mlflow ui --backend-store-uri sqlite:///mlflow.db"""),
  ("p", "Em <code>http://localhost:5000</code>: <i>Experiments → deteccao_fraudes</i>, selecione duas runs e use <i>Compare</i>."),
],
"F": [
  ("cmd", "Conferir a run pela API", """$ python -c "from src.tracking import setup_mlflow; from src.utils import load_json; from src.config import PATHS; from mlflow import MlflowClient; setup_mlflow(); r = MlflowClient().get_run(load_json(PATHS.mlflow_run)['run_id']); print(r.info.status, len(r.data.params), sorted(r.data.metrics)[:4])\""""),
  ("out", "Resposta esperada (projeto de referência)", """FINISHED 19 ['cv_pr_auc_lightgbm', 'cv_pr_auc_random_forest', 'cv_pr_auc_regressao_logistica', 'cv_pr_auc_xgboost']"""),
  ("note", "warn", "Erro real: o modelo não grava no MLflow", "O MLflow 3 grava modelos scikit-learn no formato seguro <code>skops</code>, que recusa tipos não declarados (<code>UntrustedTypesFoundException: xgboost.core.Booster</code>). Solução: passar <code>skops_trusted_types</code> com os tipos das bibliotecas usadas (veja a fase 11)."),
],
"G": ["Cada execução do pipeline cria uma run com status FINISHED.",
      "A run tem parâmetros, métricas de validação e de teste, o modelo e os artefatos.",
      "Os estágios 6 a 8 do pipeline acrescentam informações na mesma run."],
"H": "A run guarda TODOS os modelos já treinados, inclusive os ruins. A <b>Fase 11</b> separa os modelos promovidos num registro com versões e um apelido para o de produção.",
"check": [
  ("Configurar o servidor no <code>.env</code>", "Rastreio fora do código", "<code>.env</code>, <code>setup_mlflow</code>", "<code>mlflow ui</code> abre", "Experimento aparece"),
  ("Abrir a run no treino", "1 run por pipeline", "<code>artifacts/mlflow_run.json</code>", "Comando do 10.F", "Status FINISHED"),
  ("Reabrir a run nos estágios seguintes", "Run completa (treino + decisão + teste)", "Métricas <code>test_*</code> na run", "Comando do 10.F", "Métricas de teste presentes"),
  ("Comparar duas runs no <code>mlflow ui</code>", "Saber usar o rastreio", "—", "Tela Compare", "Diferenças visíveis"),
],
},
# ============================================================================================= F11
{
"n": 11, "grupo": P3, "curto": "Registro de modelos", "entregavel": "Versão no Model Registry com alias @champion",
"titulo": "Versionar e registrar o modelo: versões numeradas e um apelido para o de produção",
"lede": "Dois mecanismos complementares: o DVC versiona os arquivos do modelo junto com o código (dvc.lock), e o Model Registry do MLflow guarda as versões promovidas com um alias (@champion) que a produção consulta. Promover e voltar atrás é mover o alias.",
"pastas": "src/model_registry/ · dvc.lock · MLflow Model Registry",
"ref": "<p><b>Implementado e validado.</b> <code>fraude_xgboost</code> <b>v1</b>, alias <code>@champion</code>, com tags (<code>test_pr_auc</code>, <code>limiar</code>). A inferência pelo registry deu previsões <b>idênticas</b> às da origem local. Lição real: a lista <code>SKOPS_TRUSTED_TYPES</code> precisou incluir XGBoost, LightGBM, imblearn e as árvores do scikit-learn, e há um teste que grava e recarrega cada algoritmo.</p>",
"A": [
  ("p", "Criar o <b>registro do modelo</b>: uma versão numerada por modelo promovido e um alias que aponta para o de produção."),
  ("p", "<b>Problema que resolve:</b> \"qual modelo está em produção? como volto ao anterior?\". O app pede <i>o champion</i> e não precisa saber o <code>run_id</code>; rollback é mover o alias."),
],
"B": [
  ("Registro de modelos", "model registry", "Catálogo de modelos promovidos, com versões."),
  ("Versão do modelo", "model version", "v1, v2, ... Cada uma aponta para a run que a gerou."),
  ("Alias", "model alias", "Nome móvel (<code>@champion</code>) que aponta para uma versão."),
  ("Reversão", "rollback", "Voltar à versão anterior movendo o alias."),
  ("Serialização segura", "safe serialization (skops)", "Formato que não executa código ao carregar e exige declarar os tipos confiáveis."),
],
"C": [
  ("ol", [
    "Grave o modelo na run com assinatura e com <code>skops_trusted_types</code> (fase 10).",
    "Escreva <code>register_model.py</code>: escolhe a run (a do pipeline ou a melhor filha dos experimentos), cria a versão, grava tags e move o alias.",
    "Rode-o <b>depois</b> do <code>dvc repro</code> (não é um estágio do DVC: cria uma versão nova a cada execução).",
    "Teste a inferência lendo do registry (fase 12).",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "Nome do modelo registrado · nome do alias · critério para promover (ex.: maior <code>test_pr_auc</code> entre os experimentos) · quem pode mover o alias."),
],
"D": [
  ("src/model_registry/register_model.py", "<code>choose_run</code> e <code>register_model</code>.", "R"),
  ("src/model_training/models.py · SKOPS_TRUSTED_TYPES", "Tipos que o skops pode carregar (dependem das bibliotecas dos modelos).", "RE"),
  ("tests/unit/test_mlflow_serialization.py", "Grava e recarrega cada algoritmo candidato no MLflow.", "R"),
  ("dvc.lock", "Hash do <code>model.joblib</code> de cada commit: o outro lado do versionamento.", "R"),
],
"E": [
  ("code", "register_model.py · versão + tags + alias", """def register_model(run_id, nome, alias) -> int:
    client = MlflowClient()
    versao = mlflow.register_model(f"runs:/{run_id}/model", nome)
    client.set_model_version_tag(nome, versao.version, "test_pr_auc", ...)
    client.set_registered_model_alias(nome, alias, versao.version)
    return int(versao.version)"""),
  ("code", "Gravar o modelo com tipos confiáveis declarados", """SKOPS_TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier",
                       "lightgbm.basic.Booster", "lightgbm.sklearn.LGBMClassifier", ...,
                       "sklearn.tree._tree.Tree"]
mlflow.sklearn.log_model(modelo, name="model", signature=assinatura,
                         skops_trusted_types=SKOPS_TRUSTED_TYPES)"""),
  ("cmd", "Registrar e fazer rollback", """$ python -m src.model_registry.register_model
$ python -c "from src.tracking import setup_mlflow; from mlflow import MlflowClient; setup_mlflow(); MlflowClient().set_registered_model_alias('fraude_xgboost', 'champion', 1)\""""),
],
"F": [
  ("out", "Resposta esperada (projeto de referência)", """Modelo 'fraude_xgboost' versão 1 registrado a partir da run 63249e6a2cae4f139c86d21b05d3d5d4; alias 'champion' -> v1
Registro do modelo concluído"""),
  ("cmd", "Teste automático da serialização", """$ pytest tests/unit/test_mlflow_serialization.py -q"""),
  ("out", "Resposta esperada", """2 passed"""),
  ("note", "warn", "Por que este teste existe", "No teste de integração (base sintética) o vencedor era a Regressão Logística, cujos tipos são todos do scikit-learn: o problema do XGBoost só apareceu no treino real, depois de 44 minutos. Teste todos os caminhos possíveis, não só o que os dados de teste percorrem."),
],
"G": ["Existe uma versão registrada com o alias apontando para ela.",
      "A versão tem tags que permitem saber de onde veio e o quanto é boa.",
      "Todos os algoritmos candidatos gravam e recarregam no MLflow (teste passando)."],
"H": "Com o modelo versionado e apontado por um alias, a <b>Fase 12</b> cria a inferência: a peça que transforma transações novas em decisões, pela linha de comando, por uma API e num container.",
"check": [
  ("Escrever <code>register_model.py</code>", "Promover modelos com versão", "<code>register_model.py</code>", "Rodar o módulo", "Versão criada e alias movido"),
  ("Declarar os tipos confiáveis", "Gravar qualquer candidato no MLflow", "<code>SKOPS_TRUSTED_TYPES</code>", "<code>pytest tests/unit/test_mlflow_serialization.py</code>", "2 passed"),
  ("Testar o rollback", "Saber voltar atrás", "—", "Mover o alias para outra versão e voltar", "Alias move sem erro"),
],
},
# ============================================================================================= F12
{
"n": 12, "grupo": P4, "curto": "Inferência", "entregavel": "FraudPredictor + CLI + API Flask + imagem Docker",
"titulo": "Servir o modelo: uma classe de inferência usada pela CLI, pela API e pelo container",
"lede": "A inferência junta, na ordem do treino, tudo que é preciso para prever: validar a entrada, criar as features com a mesma função, calcular a probabilidade e aplicar a regra. A CLI, a API Flask e o container Docker só chamam essa classe.",
"pastas": "src/inference/ · app/ · Dockerfile · .dockerignore",
"ref": "<p><b>Implementado e validado.</b> No lote de exemplo (15 fraudes, 35 legítimas), 12 transações para revisão, todas fraudes. Local e registry: resultados idênticos. Flask: <code>/health</code>, <code>/predict</code>, <code>/upload</code> e erro 400 testados com o servidor. Imagem <code>fraude-api</code> (2,02 GB) com gunicorn: mesmas respostas. Lição real: o MLflow 3.18 exige <code>gunicorn&lt;27</code> no Linux.</p>",
"A": [
  ("p", "Criar uma <b>classe de inferência</b> e três formas de usá-la: CLI (lotes), API (sistemas) e container (implantação)."),
  ("p", "<b>Problema que resolve:</b> prever não pode exigir abrir o notebook, nem reescrever a transformação no app (training-serving skew)."),
],
"B": [
  ("Inferência", "inference / serving", "Usar o modelo treinado para prever dados novos."),
  ("Lote × tempo real", "batch vs online inference", "Lote: um arquivo de cada vez (CLI). Tempo real: uma requisição de cada vez (API)."),
  ("API REST", "REST API", "Interface HTTP: o sistema envia JSON e recebe JSON."),
  ("Fábrica de app", "app factory", "<code>create_app(predictor)</code>: os testes passam um modelo de teste; o servidor carrega o real."),
  ("Container", "container image", "Pacote com código + dependências + modelo, que roda igual em qualquer máquina."),
],
"C": [
  ("ol", [
    "Escreva a classe com <code>from_local</code>, <code>from_registry</code> e <code>predict</code>; o <code>predict</code> valida, cria features, prevê e aplica a regra.",
    "Escreva a CLI (<code>--input</code>, <code>--output</code>, <code>--source</code>).",
    "Escreva o app Flask com uma fábrica e as rotas <code>/health</code>, <code>/predict</code>, <code>/upload</code>.",
    "Escreva o <code>Dockerfile</code> (dependências numa camada, código em outra) e o <code>.dockerignore</code>.",
    "Teste os três caminhos com um lote de exemplo.",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "Contrato de entrada (formato bruto ou já com features?) e de saída · origem padrão do modelo (local ou registry) · o que entra na imagem."),
],
"D": [
  ("src/inference/predict.py", "<code>FraudPredictor</code> e a CLI.", "RE"),
  ("app/main.py · app/templates/index.html", "API e página de upload. Não repete nenhuma transformação.", "RE"),
  ("Dockerfile · .dockerignore", "Imagem do app (python:3.13-slim + gunicorn).", "R"),
  ("docs/exemplo_lote.csv", "Lote de exemplo para testar (15 fraudes e 35 legítimas do teste).", "E"),
],
"E": [
  ("code", "predict.py · a ordem do treino, numa classe", """def predict(self, transacoes: pd.DataFrame) -> pd.DataFrame:
    relatorio = validate_dataframe(transacoes, require_target=False)   # fase 6
    if not relatorio["valido"]:
        raise DataValidationError(...)
    X = build_features(transacoes)[COLUNAS_MODELO]                      # fase 7
    proba = self.model.predict_proba(X)[:, 1].astype(float)             # fase 8
    revisar = aplicar_regra(proba, X[COLUNA_VALOR], self.policy["limiar"], self.policy["regra_valor"])  # fase 9
    return pd.DataFrame({"probabilidade_fraude": proba.round(6),
                         "decisao": pd.Series(revisar, index=transacoes.index).map({True: "revisar", False: "liberar"})})"""),
  ("cmd", "CLI, API e Docker", """$ python -m src.inference.predict --input docs/exemplo_lote.csv --output previsoes.csv
$ python -m app.main
$ docker build -t fraude-api .
$ docker run -p 5001:5001 fraude-api"""),
  ("note", "warn", "Docker: dependências que mudam por sistema", "O <code>pip install</code> da imagem falhou com <code>ResolutionImpossible: mlflow 3.18.0 depends on gunicorn&lt;27</code>. No Windows o MLflow não depende do gunicorn, por isso a versão errada passou. Resolva as dependências para Linux antes de fixar: <code>uv pip compile pyproject.toml --python-platform x86_64-manylinux_2_28</code>."),
],
"F": [
  ("cmd", "Testar a API (com o servidor rodando)", """$ curl -s localhost:5001/health
$ curl -s -F "file=@docs/exemplo_lote.csv" localhost:5001/upload | grep -o "[0-9]* transações avaliadas"
$ curl -s -X POST localhost:5001/predict -H "Content-Type: application/json" -d '{"transacoes":[{"Timestamp":1}]}' -o /dev/null -w "%{http_code}\\n\""""),
  ("out", "Resposta esperada (projeto de referência)", """{"limiar":0.11,"modelo":"local: model.joblib","regra_valor":{"corte_valor":200.0,"limiar_valor":0.005},"status":"ok"}
50 transações avaliadas
400"""),
  ("p", "O 400 é o comportamento certo: a entrada sem as colunas é recusada pela mesma validação do pipeline."),
],
"G": ["CLI, API e container devolvem as mesmas previsões para o mesmo lote.",
      "Origem local e registry dão resultados idênticos.",
      "Entrada inválida é recusada (erro claro / HTTP 400), nunca prevista.",
      "O app não reimplementa nenhuma transformação."],
"H": "Cada estágio roda sozinho com <code>python -m</code>. A <b>Fase 13</b> amarra tudo num pipeline que o DVC reproduz e que o Airflow pode agendar.",
"check": [
  ("Escrever a classe de inferência", "Uma fonte da verdade para prever", "<code>predict.py</code>", "<code>pytest tests/unit/test_inference.py</code>", "Testes passam"),
  ("Escrever a CLI", "Previsões em lote", "<code>previsoes.csv</code>", "Comando do 12.E", "Arquivo com probabilidade e decisão"),
  ("Escrever o app Flask", "Previsões por HTTP", "<code>app/main.py</code>", "Comandos do 12.F", "200 nas rotas, 400 na entrada inválida"),
  ("Construir e rodar a imagem", "Implantação reprodutível", "Imagem Docker", "<code>docker run</code> + <code>curl /health</code>", "Mesmas respostas do local"),
  ("Comparar local × registry", "Garantir a mesma versão", "—", "<code>--source local</code> × <code>--source registry</code>", "Previsões idênticas"),
],
},
]
