"""Conteúdo das Fases 13 a 17 (Parte 4 · Produção; Parte 5 · Fechamento). Formato: ver conteudo_fases_1.py."""

P4 = "Parte 4 · Produção"
P5 = "Parte 5 · Fechamento"

# Resultado real da validação do Airflow no projeto de referência (atualizar se mudar)
AIRFLOW_REF = (
    "<p><b>Airflow: parcialmente validado.</b> A imagem (Airflow 2.10.5 + o projeto num ambiente Python separado) foi construída; a DAG "
    "<code>fraud_detection_pipeline</code> carrega sem erros, com as 10 tarefas na ordem certa; as 4 primeiras tarefas "
    "(<code>dvc_load_data</code>, <code>dvc_validate_data</code>, <code>dvc_preprocess_data</code>, <code>dvc_engineer_features</code>) "
    "terminaram com SUCCESS via <code>airflow tasks test</code>, com os mesmos números do notebook. A DAG completa (com os ~44 min de treino) "
    "não foi executada no container: a VM do Docker tem 2,8 GB. Três lições reais: pastas do OneDrive chegam como somente leitura "
    "(<code>chmod -R u+w</code>); o <code>python</code> dos estágios era o do Airflow, sem scikit-learn (a DAG põe o ambiente do projeto "
    "na frente do <code>PATH</code>); e o estágio foi encerrado por falta de memória (código 137) disputando a VM com o Airflow.</p>"
)

FASES = [
# ============================================================================================= F13
{
"n": 13, "grupo": P4, "curto": "Orquestração", "entregavel": "dvc.yaml (8 estágios) + DAG do Airflow",
"titulo": "Orquestrar: o DVC reproduz o pipeline; o Airflow o agenda",
"lede": "Duas camadas. O dvc.yaml declara os estágios com dependências, parâmetros e saídas, e o DVC só refaz o que mudou. A DAG do Airflow lê esses estágios e os executa numa agenda, com novas tentativas e histórico.",
"pastas": "dvc.yaml · dvc.lock · dags/ · Dockerfile.airflow · docker-compose.airflow.yaml",
"ref": "<p><b>DVC: implementado e validado.</b> 8 estágios; o <code>dvc repro</code> completo reproduziu o notebook, e o segundo <code>dvc repro</code> respondeu <i>Data and pipelines are up to date</i>.</p>" + AIRFLOW_REF,
"A": [
  ("p", "Declarar o pipeline no <b>dvc.yaml</b> e criar a <b>DAG</b> que o agenda."),
  ("p", "<b>Problema que resolve:</b> etapas rodadas à mão, fora de ordem, ou refeitas sem necessidade. O DVC garante a ordem e o cache; o Airflow garante a execução recorrente e a visibilidade."),
],
"B": [
  ("Orquestração", "pipeline orchestration", "Executar etapas na ordem certa, de forma automática."),
  ("Estágio do DVC", "DVC stage", "<code>cmd</code> + <code>deps</code> + <code>params</code> + <code>outs</code> + <code>metrics</code>."),
  ("Arquivo de trava", "dvc.lock", "Hashes da última execução. É ele que faz o resultado ser reproduzível e auditável."),
  ("DAG e tarefa", "DAG / task (Airflow)", "O fluxo agendado e cada passo dele."),
  ("Operador", "operator (BashOperator)", "O tipo de tarefa. Aqui, um comando de terminal."),
],
"C": [
  ("ol", [
    "Escreva um estágio do <code>dvc.yaml</code> para cada módulo, declarando o próprio código também como <code>dep</code>.",
    "Marque as métricas com <code>cache: false</code> (ficam no Git para comparar commits).",
    "Rode <code>dvc repro</code>, depois de novo (deve dizer <i>up to date</i>), e faça o commit do <code>dvc.lock</code>.",
    "Faça um experimento com <code>dvc exp run -S</code> e compare com <code>dvc exp show</code>.",
    "Só então: escreva a DAG (lendo o <code>dvc.yaml</code>), o <code>Dockerfile.airflow</code> e o <code>docker-compose.airflow.yaml</code>; suba e teste uma tarefa.",
  ]),
  ("fig", "orquestracao", "Quem faz o quê: Airflow (quando), DVC (se e como), <code>src/</code> (o quê), MLflow ao lado."),
  ("note", "tip", "Decisões a tomar nesta fase", "Granularidade dos estágios · o que é métrica (Git) e o que é saída (cache) · agenda (ex.: <code>@monthly</code>, o plano de retreino) · se o Airflow é necessário agora ou fica para depois."),
],
"D": [
  ("dvc.yaml", "O pipeline: um bloco por estágio.", "RE"),
  ("dvc.lock", "Hashes da última execução (Git).", "R"),
  ("dags/ml_pipeline_dag.py", "Uma tarefa por estágio do <code>dvc.yaml</code> + registro + build da imagem.", "R"),
  ("Dockerfile.airflow · Dockerfile.airflow.dockerignore", "Airflow com o projeto num ambiente Python separado.", "R"),
  ("docker-compose.airflow.yaml", "Postgres + Docker-in-Docker + Airflow.", "R"),
],
"E": [
  ("code", "dvc.yaml · um estágio completo", """  train:
    cmd: python -m src.model_training.train_model
    deps:
      - src/model_training/train_model.py
      - src/model_training/models.py
      - data/processed/train.csv
      - data/processed/val.csv
    params:
      - seed
      - train
      - tune
    outs:
      - models/model_validation.joblib
      - models/model.joblib
    metrics:
      - metrics/train.json:
          cache: false"""),
  ("code", "dags/ml_pipeline_dag.py · a DAG nasce do dvc.yaml", """def get_dvc_stages():
    with open(PROJECT_ROOT / "dvc.yaml", encoding="utf-8") as f:
        return list(yaml.safe_load(f)["stages"])

dvc_tasks = [BashOperator(task_id=f"dvc_{s}", cwd=str(PROJECT_ROOT),
                          bash_command=f"{PROJECT_PYTHON} -m dvc repro {s}") for s in get_dvc_stages()]
for anterior, seguinte in zip(dvc_tasks, dvc_tasks[1:]):
    anterior >> seguinte
dvc_tasks[-1] >> register_model >> build_app_image"""),
  ("cmd", "DVC: reproduzir, conferir e experimentar", """$ python -m dvc repro
$ python -m dvc repro
$ python -m dvc dag
$ python -m dvc exp run -S tune.n_iter=5
$ python -m dvc exp show"""),
  ("cmd", "Airflow em containers", """$ docker compose -f docker-compose.airflow.yaml build
$ docker compose -f docker-compose.airflow.yaml up -d
$ docker compose -f docker-compose.airflow.yaml exec airflow airflow dags list
$ docker compose -f docker-compose.airflow.yaml exec airflow airflow tasks test fraud_detection_pipeline dvc_load_data 2026-01-01"""),
],
"F": [
  ("out", "Resposta esperada (projeto de referência)", """Stage 'train' didn't change, skipping
...
Data and pipelines are up to date.

fraud_detection_pipeline | /home/airflow/fraud_detection/dags/ml_pipeline_dag.py | mlops | True
<Task(BashOperator): dvc_load_data>
    <Task(BashOperator): dvc_validate_data>
        ... (8 estágios) ...
                                <Task(BashOperator): register_model>
                                    <Task(BashOperator): build_app_image>"""),
  ("note", "warn", "Erros reais desta fase", "<code>Unable to read RWLock-file '.dvc/tmp/rwlock'</code>: um <code>dvc repro</code> interrompido deixou a trava vazia; confira que não há processo do DVC rodando e apague o arquivo. <code>output is already tracked by SCM</code>: saída do DVC commitada no Git (fase 4). <code>No possible cache link types</code> no container: pasta sem permissão de escrita. <code>ModuleNotFoundError: sklearn</code> numa tarefa: o <code>python</code> do estágio é o do Airflow; ponha o ambiente do projeto na frente do <code>PATH</code>."),
],
"G": ["<code>dvc repro</code> roda o pipeline inteiro do zero, e a segunda execução diz <i>up to date</i>.",
      "<code>dvc.lock</code> e <code>metrics/*.json</code> estão commitados.",
      "Um experimento com <code>-S</code> aparece no <code>dvc exp show</code>.",
      "A DAG carrega sem erros e pelo menos uma tarefa roda dentro do Airflow."],
"H": "O pipeline roda sozinho, então qualquer mudança pode quebrá-lo sem ninguém ver. A <b>Fase 14</b> cria os testes que avisam antes.",
"check": [
  ("Escrever o <code>dvc.yaml</code>", "Pipeline declarativo", "<code>dvc.yaml</code>", "<code>python -m dvc dag</code>", "Grafo com todos os estágios"),
  ("Reproduzir e commitar o <code>dvc.lock</code>", "Resultado auditável", "<code>dvc.lock</code>, métricas", "<code>dvc repro</code> 2×", "Segunda vez: up to date"),
  ("Fazer um experimento", "Comparar sem editar arquivos", "Experimento no DVC", "<code>dvc exp show</code>", "Experimento listado"),
  ("Escrever a DAG e subir o Airflow", "Agendamento", "<code>dags/</code>, compose", "<code>airflow dags list</code>", "DAG sem erro de importação"),
  ("Rodar uma tarefa no Airflow", "Provar a integração", "Log da tarefa", "<code>airflow tasks test ...</code>", "Tarefa termina com sucesso"),
],
},
# ============================================================================================= F14
{
"n": 14, "grupo": P4, "curto": "Testes", "entregavel": "Testes unitários + integração (pipeline completo em pasta temporária) + CI",
"titulo": "Testar: funções isoladas, o pipeline de ponta a ponta e a integração contínua",
"lede": "Testes unitários conferem cada função pura com dados pequenos. O teste de integração roda os 8 estágios + registro + API numa pasta temporária, com uma base sintética do mesmo esquema. A CI roda tudo a cada push.",
"pastas": "tests/conftest.py · tests/unit/ · tests/integration/ · .github/workflows/tests.yml",
"ref": "<p><b>Testes: implementados e validados.</b> 39 passando (33 unitários + 6 de integração, ~5 min). <b>CI: escrito, não executado</b> (o repositório é só local, sem GitHub). Lição real: o teste de integração não pegou um erro de serialização porque, na base sintética, o algoritmo vencedor era outro (ver fase 11).</p>",
"A": [
  ("p", "Criar a <b>rede de segurança</b>: testes que falham quando uma mudança quebra uma regra, e uma CI que os roda sozinha."),
  ("p", "<b>Problema que resolve:</b> o pipeline roda sem ninguém olhando; um erro só seria descoberto em produção (ou depois de 44 minutos de treino)."),
],
"B": [
  ("Teste unitário", "unit test", "Uma função, uma entrada pequena e conhecida, uma saída conferida."),
  ("Teste de integração", "integration test", "Várias peças juntas (aqui: o pipeline inteiro)."),
  ("Fixture", "pytest fixture", "Dado ou recurso preparado para os testes (a base sintética)."),
  ("Dados sintéticos", "synthetic data", "Base com o mesmo esquema da real, pequena e com um sinal fácil."),
  ("Integração contínua", "continuous integration (CI)", "Rodar os testes automaticamente a cada mudança."),
],
"C": [
  ("ol", [
    "Escreva <code>conftest.py</code> com a base sintética (mesmas colunas, poucas linhas, um sinal simples, algumas duplicatas).",
    "Escreva testes unitários para cada regra importante: validação reprova, split estratifica, features na ordem certa, custo em reais, regra de decisão, PSI.",
    "Escreva o teste de integração: aponte a raiz do projeto e o MLflow para pastas temporárias, gere a base, rode todos os <code>main()</code>.",
    "Teste <b>todos</b> os caminhos possíveis do modelo (cada algoritmo candidato), não só o vencedor.",
    "Escreva o workflow de CI.",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "O que é regra de negócio que precisa de teste · tamanho da base sintética (rápido, mas com classes suficientes para CV) · orçamento reduzido nos testes (<code>n_iter=3</code>, <code>bootstrap_n=50</code>)."),
],
"D": [
  ("tests/conftest.py", "<code>make_synthetic_base</code> e a fixture <code>base_sintetica</code>.", "RE"),
  ("tests/unit/test_data.py", "Esquema, validação, split, features.", "RE"),
  ("tests/unit/test_metrics_and_policy.py", "Métricas, custo, regra, modelos, drift.", "RE"),
  ("tests/unit/test_inference.py", "Classe de inferência e app.", "RE"),
  ("tests/unit/test_mlflow_serialization.py", "Cada algoritmo grava e recarrega no MLflow.", "R"),
  ("tests/integration/test_pipeline.py", "Os 8 estágios + registro + inferência + Flask em pasta temporária.", "R"),
  (".github/workflows/tests.yml", "CI no GitHub Actions.", "R"),
],
"E": [
  ("code", "test_pipeline.py · o pipeline inteiro numa pasta temporária", """@pytest.fixture(scope="module")
def projeto(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("projeto")
    mp = pytest.MonkeyPatch()
    mp.setenv("FRAUD_PROJECT_ROOT", str(raiz))                         # todos os caminhos vão para tmp
    mp.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(raiz / 'mlflow.db').as_posix()}")
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    params["tune"].update({"n_iter": 3, "cv_folds": 3})                # orçamento de teste
    ...
    for estagio in (load_data, validate_data, preprocess_data, engineer_features, train_model,
                    optimize_threshold, evaluate_model, monitor, register_model):
        estagio.main()
    yield raiz"""),
  ("note", "key", "Por que FRAUD_PROJECT_ROOT", "Os caminhos de <code>config.PATHS</code> são calculados a cada acesso a partir dessa variável. Assim o teste roda o pipeline real, com o código real, sem tocar nos dados nem no <code>mlflow.db</code> do projeto."),
],
"F": [
  ("cmd", "Rodar os testes", """$ pytest
$ pytest tests/unit -q
$ pytest -m integration -q"""),
  ("out", "Resposta esperada (projeto de referência)", """39 passed in 299.20s (0:04:59)"""),
  ("p", "Teste do teste: quebre de propósito uma regra (ex.: comente a remoção de duplicatas) e confirme que algum teste falha. Depois desfaça."),
],
"G": ["Os testes unitários cobrem cada regra de negócio e cada função pública.",
      "O teste de integração roda o pipeline inteiro sem tocar nos dados reais.",
      "Todos os algoritmos candidatos são testados.",
      "O workflow de CI existe (e roda, se houver repositório remoto)."],
"H": "Os testes protegem o código. A <b>Fase 15</b> protege a operação: logs para saber o que aconteceu e monitoramento para saber se o modelo continua bom.",
"check": [
  ("Escrever a base sintética", "Testes rápidos e portáteis", "<code>conftest.py</code>", "Usada pelos testes", "Mesmo esquema da base real"),
  ("Escrever os testes unitários", "Proteger as regras", "<code>tests/unit/</code>", "<code>pytest tests/unit</code>", "Todos passam"),
  ("Escrever o teste de integração", "Proteger o encadeamento", "<code>tests/integration/</code>", "<code>pytest -m integration</code>", "Todos passam"),
  ("Testar todos os candidatos", "Evitar erro só no caminho real", "<code>test_mlflow_serialization.py</code>", "<code>pytest</code>", "Passa para cada algoritmo"),
  ("Escrever a CI", "Rodar a cada mudança", "<code>.github/workflows/tests.yml</code>", "Push para o GitHub", "Workflow verde (se houver remoto)"),
],
},
# ============================================================================================= F15
{
"n": 15, "grupo": P4, "curto": "Logging e monitoramento", "entregavel": "logs/pipeline.log + estágio monitor (PSI, K-S, SLA por lote)",
"titulo": "Saber o que aconteceu (logs) e se o modelo continua bom (monitoramento)",
"lede": "O logging registra cada execução com horário, módulo e nível. O monitoramento mede dois tipos de mudança: nas entradas (data drift, sem gabarito) e no desempenho (concept drift, quando o gabarito chega), comparando com o SLA.",
"pastas": "src/__init__.py · logs/ · src/monitoring/ · metrics/monitoring.json",
"ref": "<p><b>Implementado e validado (monitoramento simulado).</b> PSI dia 2 × dia 1: <code>lim_crt</code> 1,05 (mudança), <code>cep</code> 0,195 (atenção), score 0,035 (estável). Validação temporal: PR-AUC 0,798, precisão 0,268. Lotes de 4 h: 1 de 3 abaixo do SLA. Iguais ao notebook. Simulado com os 2 dias da base; em produção rodaria sobre transações novas.</p>",
"A": [
  ("p", "Garantir <b>observabilidade</b>: logs de cada execução e um estágio que mede deriva e desempenho contra o SLA."),
  ("p", "<b>Problema que resolve:</b> modelos envelhecem (fraudadores mudam de tática) e ninguém percebe. Sem logs, também não se sabe por que uma execução falhou."),
],
"B": [
  ("Registro de eventos", "logging", "Mensagens com nível (INFO, WARNING, ERROR), horário e origem."),
  ("Deriva de dados", "data drift", "A distribuição das entradas muda. Medida sem gabarito."),
  ("Deriva de conceito", "concept drift", "A relação entre entradas e alvo muda. Só aparece nas métricas, quando o gabarito chega."),
  ("PSI", "population stability index", "&lt; 0,10 estável; 0,10–0,25 atenção; &gt; 0,25 mudança relevante."),
  ("Teste K-S", "Kolmogorov–Smirnov", "Maior distância entre duas distribuições acumuladas (0 a 1). Com muitos dados, leia a distância, não o p-valor."),
],
"C": [
  ("ol", [
    "Use <code>logger = logging.getLogger(__name__)</code> em cada módulo; nunca <code>print</code> no pipeline.",
    "Escreva as funções de drift (PSI, K-S) e de desempenho por lote.",
    "Escreva o estágio <code>monitor</code>: referência × atual (aqui, dia 1 × dia 2), validação temporal e lotes × SLA.",
    "Defina o plano de ação: o que fazer com cada alerta (investigar, retreinar).",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "Quais variáveis monitorar (as mais importantes + o score) · janela de referência · tamanho do lote · SLA · gatilhos de retreino."),
],
"D": [
  ("src/__init__.py · logs/pipeline.log", "Configuração única do logging; arquivo com o histórico.", "R"),
  ("src/monitoring/drift.py", "<code>psi</code>, <code>relatorio_drift</code>, <code>painel_lotes</code>.", "R"),
  ("src/monitoring/monitor.py", "Drift entre períodos, validação temporal, lotes × SLA.", "E"),
  ("metrics/monitoring.json", "Resultado e alertas.", "R"),
],
"E": [
  ("code", "drift.py · PSI por faixas da distribuição de referência", """def psi(esperado, atual, faixas=10) -> float:
    cortes = np.unique(np.quantile(esperado, np.linspace(0, 1, faixas + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p_esp = np.histogram(esperado, cortes)[0] / len(esperado)
    p_atu = np.histogram(atual, cortes)[0] / len(atual)
    p_esp, p_atu = np.clip(p_esp, 1e-6, None), np.clip(p_atu, 1e-6, None)
    return float(np.sum((p_atu - p_esp) * np.log(p_atu / p_esp)))"""),
  ("cmd", "Rodar o monitoramento e ler o log", """$ python -m src.monitoring.monitor
$ tail -n 20 logs/pipeline.log"""),
],
"F": [
  ("out", "Resposta esperada (projeto de referência)", """Drift dia 2 x dia 1 | lim_crt          PSI 1.0513 | K-S 0.361 | Mudança
Drift dia 2 x dia 1 | score do modelo  PSI 0.0347 | K-S 0.061 | Estável
Validação temporal (a partir de 36 h): PR-AUC 0.798 | recall 0.798 | precisão 0.268
Lote 44 a 48 h: PR-AUC 0.734 | recall 0.773 | abaixo do SLA"""),
  ("note", "key", "Lendo o resultado", "O PSI de <code>lim_crt</code> é alto, mas a PR-AUC dos lotes não cai junto: <b>data drift nem sempre derruba o desempenho</b>. Por isso os dois monitoramentos são necessários. Um lote isolado abaixo do SLA, com só 22 fraudes, pede investigação, não retreino automático."),
],
"G": ["Todos os módulos logam pelo <code>logging</code>; <code>logs/pipeline.log</code> cresce a cada execução.",
      "O estágio de monitoramento grava PSI/K-S, desempenho por lote e alertas.",
      "Existe um plano de ação escrito para os alertas."],
"H": "O sistema está completo. A <b>Fase 16</b> garante que outra pessoa (ou você daqui a seis meses) consiga entender, rodar e reproduzir tudo.",
"check": [
  ("Usar logging em todos os módulos", "Histórico das execuções", "<code>logs/pipeline.log</code>", "<code>grep -L \"getLogger\" src/*/*.py</code>", "Só aparecem bibliotecas sem <code>main()</code> (no exemplo: <code>metrics.py</code>, <code>models.py</code>, <code>drift.py</code>)"),
  ("Escrever as funções de drift", "Medir mudanças nas entradas", "<code>drift.py</code>", "<code>pytest -k psi</code>", "PSI ~0 igual, &gt;0,25 deslocado"),
  ("Escrever o estágio <code>monitor</code>", "Deriva + desempenho × SLA", "<code>monitoring.json</code>", "<code>python -m src.monitoring.monitor</code>", "JSON com alertas"),
  ("Escrever o plano de ação", "Saber o que fazer com um alerta", "Seção no README", "Ler", "Gatilhos e ações definidos"),
],
},
# ============================================================================================= F16
{
"n": 16, "grupo": P5, "curto": "Documentação", "entregavel": "README didático, docstrings, diagrama e garantias de reprodutibilidade",
"titulo": "Documentar e garantir a reprodutibilidade",
"lede": "Documentação em três níveis: o docstring de cada módulo (por que existe, entradas, saídas, etapa do ciclo), o README (o projeto inteiro) e o diagrama. A reprodutibilidade é conferida, não prometida.",
"pastas": "README.md · docs/ · docstrings em src/ · dvc.lock · pyproject.toml",
"ref": "<p><b>Implementado e validado.</b> README com as 15 seções pedidas + tabela notebook → código + resultados reais + limitações. Todo módulo de <code>src/</code> abre com um docstring no mesmo padrão. Diagrama em <code>docs/arquitetura.svg</code>. Garantias: sementes, versões fixas, CSV em round trip, ordem preservada, <code>dvc.lock</code>.</p>",
"A": [
  ("p", "Deixar o projeto <b>compreensível e reproduzível</b> por quem não participou dele."),
  ("p", "<b>Problema que resolve:</b> \"na minha máquina funcionava\" e o conhecimento que só existe na cabeça de quem fez."),
],
"B": [
  ("Docstring", "docstring", "Texto no início de um módulo, classe ou função explicando responsabilidade, entradas e saídas."),
  ("Reprodutibilidade", "reproducibility", "Mesmo código + dados + parâmetros + ambiente = mesmo resultado."),
  ("Ficha do modelo", "model card", "Documento do modelo para quem vai usá-lo (fase 9)."),
  ("Registro de decisões", "decision log", "Por que cada escolha foi feita (a tabela notebook → código)."),
],
"C": [
  ("ol", [
    "Escreva o docstring de cada módulo: por que existe · entradas · saídas · etapa do ciclo · como executar.",
    "Escreva o README: problema, MLOps no projeto, arquitetura, pastas e arquivos, fluxo de dados e de treino, versionamento, como executar, testar, prever e reproduzir, tecnologias, glossário.",
    "Atualize o diagrama para refletir o que foi construído.",
    "Confira as garantias de reprodutibilidade (abaixo) e escreva as limitações.",
  ]),
  ("table", ["Garantia", "Como foi implementada"], [
    ("Mesmo ambiente", "versões fixas no <code>pyproject.toml</code>; imagem Docker"),
    ("Mesmos sorteios", "<code>seed: 42</code> em split, modelos, busca e bootstrap"),
    ("Mesmos números lidos", "<code>read_csv(..., float_precision=\"round_trip\")</code>"),
    ("Mesma ordem de linhas", "<code>train_val_ids.csv</code> (a CV depende da ordem)"),
    ("Mesmos dados e saídas", "<code>dvc.lock</code> com o hash de cada arquivo"),
  ]),
],
"D": [
  ("README.md", "A documentação do projeto inteiro.", "RE"),
  ("docs/arquitetura.svg", "O desenho da arquitetura construída.", "E"),
  ("Docstrings em src/", "Explicação local de cada módulo.", "R"),
  ("notebooks/README.md", "Pesquisa × produção.", "R"),
],
"E": [
  ("code", "O padrão de docstring de módulo (projeto de referência)", '''"""Estágio 3 do pipeline: limpar os dados e separar treino, validação e teste.

Etapa do ciclo MLOps: **Data Preprocessing**.

- Entrada: `data/raw/raw.csv`.
- Saídas:  `data/preprocessed/{train,val,test}.csv` e `metrics/preprocessing.json`.

O que faz (decisões do notebook v2, Fase 02 item 2.3 e Fase 03 item 4.1): ...
Por que ...

Executar: python -m src.data_preprocessing.preprocess_data
"""'''),
  ("cmd", "Conferir a reprodutibilidade", """$ python -m dvc repro
$ python -m dvc status
$ git status --short"""),
],
"F": [
  ("out", "Resposta esperada", """Data and pipelines are up to date.
Data and pipelines are up to date."""),
  ("p", "O teste definitivo: clone o repositório numa pasta nova, siga só o README e veja se chega ao mesmo resultado."),
],
"G": ["Todo módulo de <code>src/</code> tem docstring no padrão.",
      "O README permite rodar o projeto do zero sem perguntar nada a ninguém.",
      "O diagrama corresponde ao que existe.",
      "<code>dvc repro</code> e <code>dvc status</code> respondem <i>up to date</i> depois da execução."],
"H": "Falta a prova final: rodar tudo de ponta a ponta e comparar com o notebook. É a <b>Fase 17</b>.",
"check": [
  ("Escrever os docstrings", "Explicação local", "Docstrings", "<code>python -c \"import src.monitoring.monitor as m; print(m.__doc__[:80])\"</code>", "Todo módulo documentado"),
  ("Escrever o README", "Explicação global", "<code>README.md</code>", "Seguir o README do zero", "Funciona sem ajuda"),
  ("Atualizar o diagrama", "Visão da arquitetura", "<code>docs/arquitetura.svg</code>", "Comparar com as pastas", "Corresponde ao construído"),
  ("Conferir a reprodutibilidade", "Garantir o mesmo resultado", "—", "<code>dvc repro</code> + <code>dvc status</code>", "up to date"),
],
},
# ============================================================================================= F17
{
"n": 17, "grupo": P5, "curto": "Validação final", "entregavel": "Execução completa conferida contra o notebook e entregáveis no ar",
"titulo": "Validação final: tudo de ponta a ponta, comparado com o notebook",
"lede": "A última fase não cria nada novo: executa e confere. O critério de sucesso da migração é reproduzir os números do notebook (os anotados na fase 2) e ter cada entregável funcionando.",
"pastas": "todas",
"ref": "<p><b>Validado, com uma ressalva.</b> Todos os números do notebook reproduzidos exatamente (tabela no README); 39 testes passando; registro, CLI, Flask e Docker funcionando. Ressalva: o Airflow foi validado parcialmente (fase 13) e a CI não foi executada (sem remoto).</p>",
"A": [
  ("p", "Rodar a sequência completa e conferir cada entregável contra um critério objetivo."),
  ("p", "<b>Problema que resolve:</b> declarar \"pronto\" sem prova. Aqui, pronto = cada linha da tabela de verificação conferida."),
],
"B": [
  ("Teste de aceitação", "acceptance test", "Verificação final contra o critério de sucesso combinado."),
  ("Paridade com a referência", "parity check", "Os números do pipeline batem com os do notebook."),
  ("Execução de ponta a ponta", "end-to-end run", "Do dado bruto à previsão servida."),
],
"C": [
  ("ol", [
    "Rode os testes.",
    "Rode o pipeline do zero (<code>dvc repro -f</code>) e depois sem forçar (<i>up to date</i>).",
    "Compare as métricas com os números da fase 2, linha a linha.",
    "Registre o modelo, preveja pela CLI e pela API, suba o container.",
    "Suba o Airflow e rode ao menos uma tarefa.",
    "Atualize o README com os resultados reais e as limitações; faça o commit final.",
  ]),
],
"D": [
  ("README.md · seção \"Resultados da execução\"", "A tabela notebook × pipeline com os números reais.", "E"),
],
"E": [
  ("cmd", "A sequência completa", """$ pytest
$ python -m dvc repro -f
$ python -m dvc repro
$ python -m dvc metrics show
$ python -m src.model_registry.register_model
$ python -m src.inference.predict --input docs/exemplo_lote.csv --source registry
$ docker build -t fraude-api . && docker run -p 5001:5001 fraude-api"""),
  ("table", ["Conferência (projeto de referência)", "Notebook v2", "Pipeline"], [
    ("Partes treino / val / teste", "170.235 / 56.745 / 56.746", "idêntico"),
    ("CV 5 dobras · XGBoost", "0,848 ± 0,028", "0,8482 ± 0,0276"),
    ("Random Search × padrão", "0,8410 × 0,8199", "0,8410 × 0,8199"),
    ("Limiar / regra de valor", "0,11 / R$ 200 · 0,005", "idêntico"),
    ("Teste: PR-AUC / economia", "0,8169 / R$ 11.494,24", "idêntico"),
    ("SLA PR-AUC / recall", "0,736 / 0,747", "idêntico"),
  ]),
],
"F": [
  ("p", "Para cada linha: o número bate? Se não bater, volte à fase do estágio correspondente e procure a diferença de lógica (semente, ordem das linhas, espaço de busca, versão de biblioteca). Não ajuste nada só para forçar o número."),
],
"G": ["Testes passando.", "Pipeline reproduz o notebook (ou a diferença está explicada).",
      "Modelo registrado e servido pela CLI, pela API e pelo container.",
      "README com resultados reais e limitações; commit final feito."],
"H": "O projeto está pronto. Para aplicar o mesmo método a outro problema, siga o <b>Anexo A</b>: ele diz o que copiar, o que reescrever e como adaptar a classificação, regressão e outros tipos de problema.",
"check": [
  ("Rodar os testes", "Código íntegro", "—", "<code>pytest</code>", "Todos passam"),
  ("Reproduzir do zero", "Pipeline autossuficiente", "<code>dvc.lock</code>", "<code>dvc repro -f</code> e depois <code>dvc repro</code>", "Termina; depois up to date"),
  ("Comparar com o notebook", "Lógica preservada", "Tabela de resultados", "Linha a linha", "Números iguais ou diferença explicada"),
  ("Servir o modelo", "Entrega funcionando", "Registry, CLI, API, imagem", "Comandos do 17.E", "Previsões iguais nos três caminhos"),
  ("Fechar a documentação", "Entregar", "README atualizado, commit", "<code>git log</code>", "Commit final"),
],
},
]
