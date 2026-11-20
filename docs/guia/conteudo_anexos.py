"""Seções de abertura (mapa e reuso) e anexos do guia. Formato dos blocos: ver conteudo_fases_1.py."""

VG = "Visão geral"
AX = "Anexos"

MAPA = {
"id": "mapa", "rotulo": "MAPA", "grupo": VG, "curto": "O roteiro e o seu progresso",
"titulo": "O roteiro inteiro e onde você está",
"lede": "17 fases em 5 partes. Cada fase entrega pastas e arquivos que a seguinte usa. O mapa abaixo é pintado com o SEU progresso, calculado a partir dos checklists.",
"blocos": [
  ("note", "tip", "Os 4 estados (iguais em todo o guia, nos capítulos e nos diagramas)",
   "<span class=\"estado d-done\">■ Já feito</span> todos os itens do checklist da fase marcados, e cada item só é marcado depois de rodar o teste e ver o critério atendido. "
   "<span class=\"estado d-now\">■ Esta fase</span> a primeira fase ainda não concluída: é a que você executa agora. "
   "<span class=\"estado d-next\">■ Próxima fase</span> a que vem logo depois da atual. "
   "<span class=\"estado d-todo\">■ Depois</span> prevista, sem execução imediata."),
  ("ul", [
    "<b>Os estados são calculados, não escolhidos.</b> Documentar uma fase não a conclui: só o checklist validado conclui. Por isso, num projeto novo, o guia começa com a Fase 1 em <span class=\"estado d-now\">■ Esta fase</span> e todo o resto em <span class=\"estado d-todo\">■ Depois</span>.",
    "<b>Um progresso por projeto.</b> Digite o nome do projeto no campo <i>Projeto</i> da barra do topo: o progresso é salvo no navegador separadamente para cada nome. Use um nome por projeto que estiver migrando.",
    "<b>Referência × seu projeto.</b> O quadro <i>No projeto de referência</i>, no topo de cada fase, conta o que foi implementado e validado em <code>mlops_detectando_fraudes_bancarias</code>, com os números reais. Ele não altera os seus estados.",
    "<b>Duas visões do mapa.</b> Este mapa mostra o seu progresso. O mapa no fim de cada fase mostra a posição daquela fase no roteiro (o que vem antes, ela, a próxima, o resto).",
  ]),
  ("note", "key", "Como cada fase está organizada",
   "<b>A</b> Objetivo · <b>B</b> Conceitos (português e inglês) · <b>C</b> O que fazer na prática · <b>D</b> Estrutura de arquivos · <b>E</b> Implementação (código e comandos reais) · "
   "<b>F</b> Como testar (resposta esperada e erros comuns) · <b>G</b> Critério de conclusão · <b>H</b> Próxima etapa · <b>✓</b> Checklist com atividade, objetivo, entregáveis, teste, critério e estado."),
],
}

REUSO = {
"id": "reuso", "rotulo": "REUSO", "grupo": VG, "curto": "Reutilizável × específico",
"titulo": "O que se reaproveita e o que muda de um projeto para outro",
"lede": "O método é o mesmo para qualquer problema de Machine Learning; o que muda são as peças do problema. Em todo o guia, as marcas REUTILIZÁVEL e ESPECÍFICO dizem a qual lado cada pasta ou arquivo pertence.",
"blocos": [
  ("fig", "reutilizavel_especifico", "À esquerda, o esqueleto que se copia; à direita, o que se reescreve; as setas mostram onde cada peça específica se encaixa."),
  ("table", ["Componente", "Tipo", "O que muda num novo projeto"], [
    ("Pastas, <code>pyproject.toml</code>, <code>.gitignore</code>, <code>utils.py</code>, logging", "<span class=\"tag tag-r\">REUTILIZÁVEL</span>", "Nome do pacote e as bibliotecas de modelagem."),
    ("<code>config.py</code> (caminhos) e <code>params.yaml</code> (estrutura)", "<span class=\"tag tag-r\">REUTILIZÁVEL</span> <span class=\"tag tag-e\">AJUSTAR</span>", "Nomes das colunas, do alvo e os valores dos parâmetros."),
    ("Molde dos estágios (função + <code>main()</code>) e <code>dvc.yaml</code>", "<span class=\"tag tag-r\">REUTILIZÁVEL</span>", "Os estágios específicos podem ganhar ou perder etapas."),
    ("Validação de dados", "<span class=\"tag tag-r\">REUTILIZÁVEL</span> <span class=\"tag tag-e\">AJUSTAR</span>", "As regras (faixas, domínios); a mecânica é a mesma."),
    ("Limpeza, separação e features", "<span class=\"tag tag-e\">ESPECÍFICO</span>", "O que o notebook do novo problema decidiu."),
    ("Modelos e espaços de busca", "<span class=\"tag tag-e\">ESPECÍFICO</span>", "Algoritmos, balanceamento, métrica de escolha."),
    ("Fluxo do treino (comparar → CV → busca → retreino)", "<span class=\"tag tag-r\">REUTILIZÁVEL</span>", "Nada; muda só o que entra nele."),
    ("Regra de decisão e métricas de negócio", "<span class=\"tag tag-e\">ESPECÍFICO</span>", "Limiar e custo (classificação) ou nenhum limiar (regressão)."),
    ("MLflow (<code>tracking.py</code>, <code>register_model.py</code>)", "<span class=\"tag tag-r\">REUTILIZÁVEL</span>", "Nome do experimento, do modelo registrado e a métrica de promoção."),
    ("Inferência, Flask, Dockerfile", "<span class=\"tag tag-r\">REUTILIZÁVEL</span> <span class=\"tag tag-e\">AJUSTAR</span>", "Contrato de entrada e saída."),
    ("Testes (base sintética, integração em pasta temporária)", "<span class=\"tag tag-r\">REUTILIZÁVEL</span> <span class=\"tag tag-e\">AJUSTAR</span>", "A base sintética segue o novo esquema."),
    ("DAG do Airflow e compose", "<span class=\"tag tag-r\">REUTILIZÁVEL</span>", "Nada: a DAG lê os estágios do <code>dvc.yaml</code>."),
    ("Monitoramento", "<span class=\"tag tag-r\">REUTILIZÁVEL</span> <span class=\"tag tag-e\">AJUSTAR</span>", "Variáveis monitoradas, SLA, janelas."),
  ]),
],
}

ANEXOS = [
# ============================================================================================= A
{
"id": "anexo-a", "rotulo": "ANEXO A", "grupo": AX, "curto": "Aplicar a um novo projeto",
"titulo": "Aplicar o roteiro a um novo projeto de Ciência de Dados",
"lede": "O caminho para repetir tudo com outro problema e outros dados: o que copiar do projeto de referência, o que reescrever e o que muda por tipo de problema.",
"blocos": [
  ("ol", [
    "<b>Escolha o projeto.</b> Prefira um notebook que já tenha um modelo final escolhido e números de avaliação: a migração precisa de uma referência para conferir.",
    "<b>Abra este guia com um nome de projeto novo</b> (campo <i>Projeto</i> no topo). A Fase 1 fica em <span class=\"estado d-now\">■ Esta fase</span>.",
    "<b>Fases 1 e 2:</b> preencha a ficha de diagnóstico e faça o inventário do notebook (decisões, estudos, problemas, números de referência).",
    "<b>Separe o específico do padrão</b> usando a seção <a href=\"#reuso\">Reutilizável × específico</a>: liste as peças da coluna da direita para o novo problema.",
    "<b>Copie o esqueleto</b> do projeto de referência (comando abaixo) e renomeie: pacote no <code>pyproject.toml</code>, variável de raiz (<code>FRAUD_PROJECT_ROOT</code>), experimento e modelo no MLflow.",
    "<b>Reescreva as peças específicas</b>, fase a fase: esquema em <code>config.py</code>, regras de validação, limpeza e split, <code>build_features</code>, <code>models.py</code>, métrica e regra de decisão, variáveis monitoradas, base sintética dos testes.",
    "<b>Substitua modelos e pré-processamento sem quebrar a arquitetura:</b> o que aprende com os dados vai para dentro do Pipeline do modelo; o resto vai para <code>build_features</code>. As assinaturas das funções (o que entra e o que sai de cada estágio) continuam iguais.",
    "<b>Execute e valide</b> fase a fase, marcando os checklists. A Fase 17 compara com os números do notebook novo.",
  ]),
  ("cmd", "Copiar o esqueleto reutilizável (ajuste os caminhos)", """$ mkdir mlops_novo_projeto && cd mlops_novo_projeto
$ git init
$ cp -r ../mlops_detectando_fraudes_bancarias/{src,app,dags,tests,docs/guia} .
$ cp ../mlops_detectando_fraudes_bancarias/{pyproject.toml,params.yaml,dvc.yaml,Dockerfile,.dockerignore,Dockerfile.airflow,Dockerfile.airflow.dockerignore,docker-compose.airflow.yaml,.gitignore,.gitattributes,.env.example} .
$ mkdir -p data/{external,raw,preprocessed,processed} models artifacts metrics logs"""),
  ("note", "warn", "Depois de copiar", "Apague dos módulos tudo que é de fraude antes de rodar: <code>COLUNAS_ORIGINAIS</code>, <code>hora_do_dia</code>, a regra de valor, as faixas de validação, os nomes <code>fraude_xgboost</code>/<code>deteccao_fraudes</code>. Rode <code>pytest</code>: os testes de fraude vão falhar, e isso é o mapa do que falta adaptar."),
  ("table", ["Tipo de problema", "Separação", "Métrica principal", "Balanceamento / pesos", "Regra de decisão", "Monitorar"], [
    ("Classificação binária desbalanceada (ex.: fraude)", "estratificada", "PR-AUC + custo", "peso da classe ou SMOTE no Pipeline", "limiar pelo custo", "PSI + PR-AUC/recall por lote"),
    ("Classificação balanceada / multiclasse", "estratificada", "F1 macro, AUC-ROC, log loss", "em geral, não precisa", "classe de maior probabilidade (ou limiar por classe)", "PSI + acurácia/F1 por lote"),
    ("Regressão (ex.: preço de aluguel)", "aleatória ou por faixas do alvo", "MAE ou RMSE na unidade do negócio", "não se aplica", "sem limiar; cenários pior/provável/melhor", "PSI + MAE por lote; viés"),
    ("Séries temporais / previsão", "<b>temporal</b> (nunca aleatória)", "MAE/MAPE por horizonte", "não se aplica", "previsão + intervalo", "erro por janela; sazonalidade"),
    ("Agrupamento (clusters)", "sem alvo: amostras de ajuste e de checagem", "silhueta, estabilidade dos grupos", "não se aplica", "atribuição ao grupo", "tamanho e perfil dos grupos"),
  ]),
  ("note", "key", "O que NÃO muda em nenhum tipo", "Dados versionados · validação que bloqueia · separação antes de qualquer ajuste · teste lido uma vez · uma run por pipeline · registro com alias · uma classe de inferência que reusa as funções do treino · testes com base sintética · <code>dvc repro</code> reproduzindo tudo."),
],
},
# ============================================================================================= B
{
"id": "anexo-b", "rotulo": "ANEXO B", "grupo": AX, "curto": "Projeto de referência",
"titulo": "O projeto de referência: arquitetura construída e resultados reais",
"lede": "mlops_detectando_fraudes_bancarias, a partir do notebook \"Detectando Fraudes em Transferências Bancárias - v2\". O diagrama mostra a arquitetura efetivamente implementada.",
"blocos": [
  ("fig", "../arquitetura", "Arquitetura implementada (docs/arquitetura.svg): pipeline do DVC, MLflow, inferência, Airflow, versionamento, testes e configuração."),
  ("table", ["Componente", "Situação no projeto de referência", "Evidência"], [
    ("Pipeline DVC (8 estágios)", "implementado e validado", "<code>dvc repro</code> completo; 2ª execução <i>up to date</i>"),
    ("Paridade com o notebook", "validada", "todos os números iguais (README, Resultados da execução)"),
    ("MLflow tracking + registry", "implementado e validado", "run <code>63249e6a…</code>; <code>fraude_xgboost</code> v1 <code>@champion</code>"),
    ("Inferência CLI / Flask / Docker", "implementado e validado", "lote de exemplo: 12 revisões; local = registry; container respondendo"),
    ("Testes", "implementado e validado", "39 passando (33 unitários + 6 integração)"),
    ("CI (GitHub Actions)", "escrito, não executado", "sem repositório remoto"),
    ("Airflow", "implementado e validado", "DAG completa no container: 10/10 tarefas SUCCESS (VM do Docker com 5 GB); imagem <code>fraude-api</code> gerada no DinD"),
    ("DagsHub (remoto)", "não implementado", "documentado no README; exige conta e token"),
    ("Monitoramento", "implementado (simulado)", "PSI/K-S e lotes × SLA com os 2 dias da base"),
  ]),
],
},
# ============================================================================================= C
{
"id": "anexo-c", "rotulo": "ANEXO C", "grupo": AX, "curto": "Quando der errado",
"titulo": "Quando der errado: os erros reais da construção do projeto de referência",
"lede": "Cada linha aconteceu de verdade durante a implementação. Eles se repetem em outros projetos, principalmente no Windows e com pastas sincronizadas.",
"blocos": [
  ("table", ["Sintoma", "Causa", "Solução", "Fase"], [
    ("<code>dvc.exe: Permission denied</code>", "Política de controle de aplicativos do Windows bloqueia o executável", "Use <code>python -m dvc</code> (equivalente)", "5"),
    ("<code>output 'models/...' is already tracked by SCM</code>", "Um <code>git add -A</code> incluiu saídas do DVC antes de existirem <code>.gitignore</code> para elas", "<code>git rm --cached</code> nos arquivos; declare as saídas no <code>.gitignore</code> da raiz", "4, 13"),
    ("<code>UntrustedTypesFoundException: xgboost.core.Booster</code>", "MLflow 3 grava em <code>skops</code> e recusa tipos não declarados", "<code>skops_trusted_types</code> + teste que grava cada algoritmo", "10, 11"),
    ("<code>ResolutionImpossible ... gunicorn&lt;27</code> no <code>docker build</code>", "Dependência do MLflow que só existe no Linux", "Resolver para Linux: <code>uv pip compile ... --python-platform x86_64-manylinux_2_28</code>", "4, 12"),
    ("<code>Unable to read RWLock-file '.dvc/tmp/rwlock'</code>", "<code>dvc repro</code> interrompido deixou a trava vazia", "Confirmar que nenhum DVC roda e apagar <code>.dvc/tmp/rwlock</code>", "13"),
    ("<code>No possible cache link types</code> / <code>Permission denied</code> no container do Airflow", "Pastas copiadas do OneDrive chegam como somente leitura", "<code>RUN chmod -R u+w</code> após o <code>COPY</code>", "13"),
    ("<code>ModuleNotFoundError: No module named 'sklearn'</code> numa tarefa do Airflow", "O <code>cmd</code> do estágio chama <code>python</code>, que no container é o Python do Airflow", "Na DAG, <code>export PATH=\"/opt/project-venv/bin:$PATH\"</code> antes do <code>dvc repro</code>", "13"),
    ("Tarefa termina com código 137 (sem traceback)", "Processo encerrado por falta de memória: a VM do Docker (3 GB) é dividida com o Airflow", "<code>memory=5GB</code> no <code>.wslconfig</code> + <code>wsl --shutdown</code>, ou rodar os estágios pesados fora do container", "13"),
    ("<code>No response from gunicorn master within 120 seconds</code> e o container reinicia", "Com a CPU ocupada, o webserver do Airflow demora a subir e se desliga, levando a tarefa junto", "<code>AIRFLOW__WEBSERVER__WEB_SERVER_MASTER_TIMEOUT=600</code> e <code>AIRFLOW__WEBSERVER__WORKERS=2</code>; esperar o <code>/health</code> antes de disparar", "13"),
    ("<code>UniqueViolation ... xcom_pkey</code> numa tarefa que fez o trabalho", "A mesma tarefa rodou duas vezes ao mesmo tempo (o primeiro processo continuava vivo)", "Conferir o estado (<code>airflow tasks states-for-dag-run</code>) antes de rodar de novo", "13"),
    ("Mesmo código e versões, números um pouco diferentes no container", "Outra plataforma (Linux × Windows, número de threads) muda a ordem das somas em ponto flutuante", "Comparar números sempre na mesma plataforma; fixar o ambiente com o container", "8, 13"),
    ("Processo longo encerrado sem erro no código", "A máquina ficou sem memória (treino + Docker em 8 GB)", "Rodar o treino sozinho; subir o Docker depois", "8, 13"),
    ("Teste de integração passa, pipeline real falha", "A base sintética percorre outro caminho (outro algoritmo vence)", "Testar todos os candidatos, não só o vencedor", "14"),
    ("Números diferentes do notebook por pouco", "Ordem das linhas na CV, leitura de floats, versões", "<code>train_val_ids.csv</code>, <code>float_precision=\"round_trip\"</code>, versões fixas", "6, 8"),
  ]),
],
},
# ============================================================================================= D
{
"id": "anexo-d", "rotulo": "ANEXO D", "grupo": AX, "curto": "Glossário",
"titulo": "Glossário de MLOps (português · inglês)",
"lede": "Os termos usados no guia, na ordem em que aparecem no ciclo de vida.",
"blocos": [
  ("table", ["Termo", "Em inglês", "Significado"], [
    ("MLOps", "Machine Learning Operations", "Práticas para levar um modelo à operação e mantê-lo funcionando: versionar, automatizar, testar, servir e monitorar."),
    ("Pipeline", "pipeline", "Sequência automatizada de etapas, em que a saída de uma é a entrada da próxima."),
    ("Estágio", "stage", "Uma etapa do pipeline do DVC, com comando, dependências e saídas."),
    ("DAG", "directed acyclic graph", "A ordem e as dependências das etapas, sem ciclos (no DVC e no Airflow)."),
    ("Artefato", "artifact", "Arquivo produzido pelo pipeline: modelo, regra, métricas, ficha."),
    ("Versionamento de dados", "data versioning", "Guardar versões de dados grandes fora do Git, referenciadas por hash (DVC)."),
    ("Validação de dados", "data validation", "Regras que a base precisa cumprir; as bloqueantes param o pipeline."),
    ("Vazamento de dados", "data leakage", "Informação que não existiria na hora de prever entrando no treino."),
    ("Desvio treino-produção", "training-serving skew", "Transformação diferente entre treino e produção."),
    ("Validação cruzada", "cross-validation", "Avaliar em várias dobras do treino para um resultado estável."),
    ("Rastreamento de experimentos", "experiment tracking", "Registro de parâmetros, métricas e artefatos de cada execução (MLflow)."),
    ("Run", "run", "Uma execução registrada no MLflow."),
    ("Registro de modelos", "model registry", "Catálogo de modelos promovidos, com versões."),
    ("Alias", "alias (@champion)", "Nome que aponta para uma versão; promover e reverter = mover o alias."),
    ("Serialização", "serialization", "Gravar o modelo em arquivo para usá-lo sem retreinar."),
    ("Inferência", "inference / serving", "Usar o modelo para prever dados novos (lote ou tempo real)."),
    ("Container", "container", "Pacote isolado com app e dependências, que roda igual em qualquer lugar (Docker)."),
    ("Orquestração", "orchestration", "Executar e agendar as etapas na ordem certa (DVC, Airflow)."),
    ("Integração contínua", "continuous integration (CI)", "Testes automáticos a cada mudança no código."),
    ("Deriva de dados", "data drift", "Mudança na distribuição das entradas (PSI, K-S)."),
    ("Deriva de conceito", "concept drift", "Mudança na relação entre entradas e alvo; aparece nas métricas."),
    ("SLA", "service level agreement", "Desempenho mínimo aceitável em produção."),
    ("Ficha do modelo", "model card", "Documento do modelo: dados, métricas, regra, limites e avisos."),
    ("Reprodutibilidade", "reproducibility", "Rodar de novo e obter o mesmo resultado."),
  ]),
],
},
]
