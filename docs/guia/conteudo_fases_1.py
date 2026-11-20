"""Conteúdo das Fases 1 a 7 (Parte 1 · Diagnóstico e planejamento; Parte 2 · Dados).

Cada fase segue a mesma estrutura: A objetivo · B conceitos · C prática · D arquivos · E implementação
· F teste · G critério de conclusão · H próxima etapa · checklist. Os blocos (listas de tuplas) são
desenhados pelo gerar_guia.py: ("p", html) · ("cmd", rótulo, comando) · ("code", rótulo, código)
· ("out", rótulo, saída) · ("bad", rótulo, saída) · ("note", tip|warn|key, título, html) · ("ol"/"ul", [itens])
· ("table", cabeçalho, linhas) · ("fig", nome_do_svg, legenda) · ("details", título, blocos).
D: (caminho, descrição, "R" reutilizável | "E" específico | "RE" reutilizável com ajuste).
check: (atividade, objetivo, entregáveis, como testar, critério de conclusão).
"""

P1 = "Parte 1 · Diagnóstico e planejamento"
P2 = "Parte 2 · Dados"

FASES = [
# ============================================================================================= F1
{
"n": 1, "grupo": P1, "curto": "Diagnóstico do problema", "entregavel": "Ficha de diagnóstico (problema, alvo, métrica, custo do erro)",
"titulo": "Entender o problema antes de mexer em qualquer arquivo",
"lede": "MLOps não começa no código: começa em saber o que o modelo decide, quem usa a decisão, como medir se ela é boa e quanto custa errar. Sem isso, não há como escolher métrica, regra de decisão, SLA nem monitoramento.",
"pastas": "README.md (seção do problema)",
"ref": "<p><b>Implementado e validado.</b> Problema: barrar transferências fraudulentas. Alvo binário <code>Target</code> com 0,17% de fraudes. Métrica principal: <b>PR-AUC</b> (a acurácia de \"nunca é fraude\" é 99,83%). Decisão: revisar ou liberar, pelo <b>menor custo em reais</b> (fraude perdida = valor da transação; revisão = R$ 5). Registrado no <code>README.md</code>, seção 2.</p>",
"A": [
  ("p", "Produzir uma <b>ficha de diagnóstico</b> curta que responde: qual decisão o modelo apoia, qual é o alvo, como o resultado será medido e qual é o custo de cada tipo de erro."),
  ("p", "<b>Por que é necessário:</b> toda escolha técnica das próximas fases depende dessas respostas. A métrica define como comparar modelos (fase 8); o custo do erro define o limiar (fase 9); o que é \"funcionar\" define o SLA e o monitoramento (fase 15)."),
  ("p", "<b>Problema que resolve no MLOps:</b> evita automatizar e versionar com perfeição um modelo que otimiza a coisa errada."),
],
"B": [
  ("Problema de negócio", "business problem", "A decisão que o modelo apoia (aprovar, revisar, prever demanda). É o início do ciclo de vida de ML."),
  ("Alvo / variável resposta", "target", "O que o modelo prevê. Define o tipo de problema: classificação, regressão, ranking, previsão de série temporal."),
  ("Métrica de avaliação", "evaluation metric", "Como o modelo é comparado. Precisa refletir o negócio, não a conveniência (acurácia em classes raras engana)."),
  ("Custo do erro", "cost of errors", "Quanto custa um falso positivo e um falso negativo. Base da regra de decisão e do limiar."),
  ("Critério de sucesso", "success criteria", "O valor mínimo que torna o modelo útil. Vira o SLA em produção."),
],
"C": [
  ("ol", [
    "Leia o README e a introdução do notebook; anote o problema com uma frase: <i>\"o modelo ajuda X a decidir Y\"</i>.",
    "Identifique o alvo e a sua distribuição (proporção de cada classe, ou faixa de valores).",
    "Escolha a métrica principal e as de apoio. Justifique por que a métrica óbvia (acurácia, R²) serve ou não.",
    "Descreva o custo de cada erro, mesmo que como premissa (\"revisão = R$ 5\").",
    "Defina o que é um modelo bom o bastante: o critério de sucesso.",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "Tipo de problema · métrica principal · métricas de apoio · premissas de custo · critério de sucesso. Escreva as premissas: elas viram parâmetros no <code>params.yaml</code> (fase 4) e podem ser mudadas sem tocar no código."),
],
"D": [
  ("README.md · seção \"Problema de negócio\"", "Registra o diagnóstico para quem chega depois. É lido por pessoas, não por código.", "RE"),
  ("params.yaml · decision_policy.custo_revisao", "No exemplo, a premissa de custo vira um parâmetro (criado na fase 4).", "E"),
],
"E": [
  ("p", "Modelo de ficha (copie para o README do novo projeto e preencha):"),
  ("code", "Ficha de diagnóstico", """| Pergunta                      | Resposta (exemplo: fraude)                                   |
|-------------------------------|--------------------------------------------------------------|
| Que decisão o modelo apoia?   | Revisar ou liberar cada transferência                        |
| Alvo e distribuição           | Target binário; 473 fraudes em 283.726 transações (0,17%)    |
| Tipo de problema              | Classificação binária com classes muito desbalanceadas       |
| Métrica principal             | PR-AUC (sorteio = 0,0017)                                    |
| Métricas de apoio             | Recall, precisão, F1, economia líquida em R$                 |
| Custo de cada erro            | Fraude perdida = valor da transação; revisão = R$ 5          |
| Critério de sucesso           | Melhor que o sorteio e que o limiar 0,5 em R$; SLA a definir |"""),
  ("note", "key", "Adaptando a outros problemas", "<b>Regressão</b> (ex.: preço de aluguel): métrica MAE ou RMSE na unidade do negócio; custo do erro costuma ser assimétrico (subestimar x superestimar). <b>Classificação balanceada</b>: F1 ou AUC-ROC podem bastar. <b>Previsão de séries</b>: MAPE/MAE por horizonte e validação sempre temporal."),
],
"F": [
  ("p", "Este teste é de leitura, não de código. Peça a alguém (ou releia no dia seguinte) e verifique se a ficha responde, sem ambiguidade, às 7 perguntas."),
  ("ul", ["Dá para dizer qual métrica decide entre dois modelos? Se não, a fase 8 vai travar.",
          "Dá para calcular o custo de um erro a partir da ficha? Se não, a fase 9 não terá regra de decisão.",
          "A métrica funciona com a distribuição do alvo? (classe rara + acurácia = sinal de alerta)"]),
],
"G": ["A ficha de diagnóstico está no README com as 7 perguntas respondidas.",
      "A métrica principal está justificada em relação à distribuição do alvo.",
      "As premissas de custo estão escritas e marcadas como premissas."],
"H": "Com o problema claro, a <b>Fase 2</b> lê o notebook original procurando as decisões que já foram tomadas sobre ele: o que vira código, o que fica como pesquisa e o que precisa ser corrigido.",
"check": [
  ("Escrever o problema numa frase", "Saber qual decisão o modelo apoia", "Frase no README", "Ler a frase: ela cita a decisão e quem a usa", "A frase existe e não fala de algoritmo"),
  ("Descrever o alvo e a distribuição", "Definir o tipo de problema", "Tabela com contagem/proporção do alvo", "<code>df[alvo].value_counts(normalize=True)</code>", "Tipo de problema identificado"),
  ("Escolher métrica principal e de apoio", "Ter um critério único de comparação", "Métricas no README", "Comparar com a métrica ingênua (sorteio, média)", "Métrica justificada"),
  ("Registrar custos e critério de sucesso", "Base da regra de decisão e do SLA", "Premissas no README", "Calcular o custo de 1 erro de cada tipo", "Premissas escritas"),
],
},
# ============================================================================================= F2
{
"n": 2, "grupo": P1, "curto": "Análise do notebook", "entregavel": "Inventário: decisões → código · pesquisa · problemas a corrigir",
"titulo": "Ler o notebook como um engenheiro: o que é decisão, o que é pesquisa, o que está quebrado",
"lede": "O notebook mistura exploração, testes de alternativas e a solução escolhida. Antes de criar pastas, faça o inventário: cada decisão vira uma função, cada estudo fica no notebook e cada problema de reprodutibilidade vira uma tarefa.",
"pastas": "notebooks/README.md · tabela no README.md",
"ref": "<p><b>Implementado e validado.</b> As 210 células do notebook v2 viraram uma tabela \"decisão → onde está no código\" (README, seção <i>Do notebook ao código</i>). Ficaram no notebook: EDA, 6 hipóteses, SMOTE × peso, seleção de variáveis, Grid × Random × Optuna, calibração, SHAP. Problemas encontrados: <code>requirements.txt</code> sem versões, caminhos relativos a <code>Notebooks/</code>, modelo salvo como <code>.pkl</code> solto, nenhum rastreio.</p>",
"A": [
  ("p", "Transformar o notebook num <b>inventário de três listas</b>: (1) decisões que viram código de produção, (2) estudos que ficam como pesquisa, (3) problemas que impedem a reprodução."),
  ("p", "<b>Por que é necessário:</b> o requisito central da migração é <b>preservar a lógica</b>. Sem o inventário, é fácil trocar um algoritmo, esquecer uma remoção de duplicatas ou levar para produção um experimento que foi descartado."),
],
"B": [
  ("Pesquisa × produção", "research vs production code", "O notebook responde \"o que fazer e por quê\"; o pipeline responde \"como fazer de novo, igual, sempre\"."),
  ("Vazamento de dados", "data leakage", "Informação do teste (ou do futuro) entrando no treino. Procure: escala ajustada antes do split, duplicatas, colunas de tempo, escolha de hiperparâmetros no teste."),
  ("Reprodutibilidade", "reproducibility", "Rodar de novo e obter o mesmo resultado. Inimigos: versões soltas, sementes ausentes, caminhos fixos, estado do kernel."),
  ("Dependências", "dependencies", "Bibliotecas e versões usadas. No notebook ficam implícitas; no projeto viram o <code>pyproject.toml</code>."),
],
"C": [
  ("ol", [
    "Extraia o texto do notebook (código + markdown + saídas) para um arquivo legível (comando no 2.E).",
    "Percorra as seções e classifique cada célula: <b>dados</b>, <b>EDA</b>, <b>features</b>, <b>modelagem</b>, <b>avaliação</b>, <b>implantação</b>.",
    "Para cada decisão (\"remover duplicatas\", \"usar peso da classe\"), anote: o que é, onde está (célula) e o número que a sustenta.",
    "Liste o que é estudo descartado (fica no notebook) e o que é a escolha final (vai para o código).",
    "Liste os problemas de reprodução: caminhos, versões, sementes, dados ausentes, avaliação no próprio treino.",
    "Anote os números de referência (PR-AUC, contagens): eles serão o critério para saber se o pipeline reproduz o notebook (fase 17).",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "O que é produção e o que é pesquisa. Na dúvida, leve para o código só o que o notebook usa no modelo final."),
],
"D": [
  ("notebooks/README.md", "Explica a separação pesquisa × produção e onde está o notebook original (que não é copiado).", "R"),
  ("README.md · tabela \"Do notebook ao código\"", "Liga cada decisão do notebook à função que a implementa. É a prova de que a lógica foi preservada.", "RE"),
],
"E": [
  ("p", "Extrair o notebook para texto (funciona em qualquer projeto; ajuste o caminho):"),
  ("code", "Python · extrair código, markdown e saídas de um .ipynb", """import json
nb = json.load(open("Notebooks/meu_notebook.ipynb", encoding="utf-8"))
with open("notebook.txt", "w", encoding="utf-8") as saida:
    for i, celula in enumerate(nb["cells"], 1):
        saida.write(f"### [{i}] {celula['cell_type']}\\n" + "".join(celula["source"]) + "\\n")
        for o in celula.get("outputs", []):
            texto = o.get("text") or o.get("data", {}).get("text/plain")
            if texto:
                saida.write("--> " + "".join(texto)[:800] + "\\n")"""),
  ("p", "Trecho do inventário real (projeto de referência):"),
  ("table", ["Decisão no notebook v2", "Número que a sustenta", "Destino"], [
    ("Remover duplicatas antes do split", "1.081 duplicatas, 19 fraudes", "<code>preprocess_data.remover_duplicatas</code>"),
    ("Trocar <code>Timestamp</code> por <code>hora_do_dia</code>", "fraude 3,5× maior de madrugada", "<code>engineer_features.build_features</code>"),
    ("Peso da classe em vez de SMOTE", "PR-AUC 0,891 × 0,893 (empate)", "<code>models.build_models</code>"),
    ("Escolher algoritmo pela CV de 5 dobras", "XGBoost 0,848 ± 0,028", "<code>train_model.cross_validate_models</code>"),
    ("Random Search, 20 combinações", "0,8410 × 0,8199 do padrão", "<code>train_model.tune_model</code>"),
    ("Limiar de custo mínimo + regra de valor", "0,11 · R$ 200 / 0,005", "<code>optimize_threshold</code>"),
    ("SMOTE × peso, Grid × Optuna, calibração, SHAP", "estudos comparativos", "ficam no notebook"),
  ]),
],
"F": [
  ("p", "Conferências objetivas:"),
  ("ul", ["Toda célula do notebook está em uma das três listas.",
          "Toda decisão da lista 1 tem um número de referência anotado.",
          "Nenhum estudo descartado aparece na lista de produção."]),
  ("note", "warn", "Sinal de alerta", "Métricas \"perfeitas\" (100%) no notebook costumam ser avaliação nos dados de treino. No projeto de referência, o \"LightGBM com 100% em tudo\" da versão original era exatamente isso."),
],
"G": ["As três listas (produção, pesquisa, problemas) estão completas.",
      "Cada decisão de produção tem um número de referência para conferir a reprodução.",
      "<code>notebooks/README.md</code> explica onde está o notebook e por que ele não é copiado."],
"H": "Com as decisões listadas, a <b>Fase 3</b> desenha a arquitetura: quais estágios existirão, em que ordem, e qual ferramenta resolve cada problema encontrado aqui.",
"check": [
  ("Extrair o notebook para texto", "Ler tudo, inclusive as saídas", "<code>notebook.txt</code>", "Abrir e conferir o número de células", "Arquivo com todas as células"),
  ("Classificar as células", "Separar produção de pesquisa", "Lista de decisões e de estudos", "Cada célula tem uma categoria", "Nenhuma célula sem destino"),
  ("Anotar números de referência", "Ter critério para validar a reprodução", "Tabela de números", "Conferir os números nas saídas do notebook", "Cada decisão com um número"),
  ("Listar problemas de reprodução", "Saber o que corrigir na migração", "Lista de problemas", "Caminhos, versões, sementes, vazamentos verificados", "Lista revisada"),
],
},
# ============================================================================================= F3
{
"n": 3, "grupo": P1, "curto": "Planejamento", "entregavel": "Desenho dos estágios, fluxo de dados e ferramentas",
"titulo": "Desenhar a arquitetura: estágios, fluxo de dados e uma ferramenta por problema",
"lede": "Planejar é decidir quais estágios existirão, o que cada um lê e grava, e qual ferramenta resolve cada problema da fase 2. Uma ferramenta só entra se resolve um problema concreto.",
"pastas": "docs/arquitetura.svg",
"ref": "<p><b>Implementado e validado.</b> 8 estágios no DVC (load → validate → preprocess → features → train → decision_policy → evaluate → monitor), MLflow para rastreio e registro, Flask + Docker para servir, Airflow para agendar. Desenho em <code>docs/arquitetura.svg</code>. TensorFlow (do projeto-exemplo do curso) ficou de fora: o notebook não usa rede neural.</p>",
"A": [
  ("p", "Produzir o <b>desenho da arquitetura</b>: a lista de estágios com entradas e saídas, a ordem entre eles e a tabela problema → ferramenta."),
  ("p", "<b>Por que é necessário:</b> o desenho evita dois erros opostos: um pipeline monolítico (um script que faz tudo, impossível de testar) e uma arquitetura inflada (ferramentas sem problema para resolver)."),
],
"B": [
  ("Estágio", "pipeline stage", "Uma etapa com uma responsabilidade: lê arquivos, transforma, grava arquivos."),
  ("Grafo acíclico dirigido", "DAG", "A ordem e as dependências entre estágios. Sem ciclos: um estágio nunca depende de algo que vem depois dele."),
  ("Artefato", "artifact", "Tudo que um estágio produz: dados, modelo, regra de decisão, métricas, ficha do modelo."),
  ("Separação de responsabilidades", "separation of concerns", "Cada módulo faz uma coisa. Facilita testar, trocar e entender."),
],
"C": [
  ("ol", [
    "A partir do inventário, liste os estágios na ordem natural: carga → validação → limpeza/split → features → treino → decisão → avaliação → monitoramento.",
    "Para cada estágio, escreva: entradas, saídas, parâmetros usados.",
    "Marque quais estágios podem ler o conjunto de teste (só a avaliação) — veja o diagrama da fase 6.",
    "Monte a tabela problema → ferramenta (abaixo) e corte qualquer ferramenta que não resolva um problema da sua lista.",
    "Desenhe (papel, draw.io ou SVG) e guarde em <code>docs/</code>.",
  ]),
  ("table", ["Problema", "Ferramenta (no exemplo)", "Fase"], [
    ("\"Qual base gerou este modelo?\"", "DVC (versionamento de dados)", "5"),
    ("Etapas rodadas à mão, fora de ordem", "DVC (dvc.yaml) e, para agendar, Airflow", "13"),
    ("\"Qual configuração deu aquele resultado?\"", "MLflow Tracking", "10"),
    ("\"Qual modelo está em produção? Como volto ao anterior?\"", "MLflow Model Registry", "11"),
    ("Prever exige abrir o notebook", "Classe de inferência + Flask + Docker", "12"),
    ("Mudança que quebra algo sem ninguém ver", "pytest + CI", "14"),
  ]),
],
"D": [
  ("docs/arquitetura.svg", "O desenho da arquitetura efetivamente construída (atualizado no fim do projeto).", "RE"),
  ("README.md · seção \"Arquitetura geral\"", "A versão em texto do desenho, com as ideias que organizam o projeto.", "RE"),
],
"E": [
  ("fig", "orquestracao", "As três camadas de execução do projeto de referência: o Airflow decide quando, o DVC decide se e como, os módulos de <code>src/</code> fazem o trabalho."),
  ("p", "Lista de estágios do exemplo, como ela ficou no <code>dvc.yaml</code> (fase 13):"),
  ("code", "Estágios, entradas e saídas", """load_data          zip               → data/raw/raw.csv
validate_data      raw.csv           → metrics/data_validation.json (para se falhar)
preprocess_data    raw.csv           → data/preprocessed/{train,val,test}.csv
engineer_features  preprocessed/*    → data/processed/{train,val,test}.csv
train              processed/train,val → models/model_validation.joblib, models/model.joblib
decision_policy    model_validation + val → artifacts/decision_policy.json
evaluate           model + policy + test  → metrics/evaluation.json, models/model_card.json
monitor            base inteira + modelos  → metrics/monitoring.json"""),
],
"F": [
  ("ul", ["Cada estágio tem pelo menos uma saída que outro estágio (ou a inferência) usa.",
          "Só um estágio lê o conjunto de teste.",
          "Cada ferramenta da tabela aponta para um problema do inventário da fase 2.",
          "Não há ciclo: seguindo as setas, nunca se volta a um estágio anterior."]),
],
"G": ["Lista de estágios com entradas, saídas e parâmetros.",
      "Tabela problema → ferramenta sem ferramenta \"sobrando\".",
      "Desenho salvo em <code>docs/</code>."],
"H": "Com o desenho pronto, a <b>Fase 4</b> cria a estrutura de pastas, o ambiente com versões fixas e as peças que todo estágio usa: configuração, parâmetros e logging.",
"check": [
  ("Listar os estágios com entradas e saídas", "Ter o contrato de cada etapa", "Lista de estágios", "Cada saída é usada por alguém", "Nenhum estágio órfão"),
  ("Definir quem lê o teste", "Prevenir vazamento por desenho", "Regra anotada", "Só a avaliação lê o teste", "Regra no desenho"),
  ("Montar a tabela problema → ferramenta", "Evitar complexidade sem motivo", "Tabela", "Cada ferramenta resolve um problema listado", "Sem ferramenta sobrando"),
  ("Desenhar a arquitetura", "Comunicar o plano", "<code>docs/arquitetura.svg</code> (ou rascunho)", "Abrir o arquivo", "Desenho com setas e estágios"),
],
},
# ============================================================================================= F4
{
"n": 4, "grupo": P1, "curto": "Estrutura e ambiente", "entregavel": "Pastas, pyproject com versões fixas, config.py, params.yaml, logging, Git",
"titulo": "Criar o esqueleto: pastas, ambiente com versões fixas, configuração, parâmetros e logging",
"lede": "Esta fase cria o que todos os estágios vão usar: a estrutura de pastas, o ambiente Python com versões fixas, o lugar único dos caminhos (config.py), dos parâmetros (params.yaml) e o logging. É a parte mais reutilizável do roteiro.",
"pastas": "src/ · params.yaml · pyproject.toml · .gitignore · .env",
"ref": "<p><b>Implementado e validado.</b> <code>pyproject.toml</code> com 14 dependências fixadas, mais o pytest nos extras de desenvolvimento (Python 3.13). <code>src/config.py</code> com a raiz configurável por <code>FRAUD_PROJECT_ROOT</code> (usada pelos testes). Logging em <code>src/__init__.py</code> → <code>logs/pipeline.log</code>. Git local com commits por etapa. Lição real: o <code>.gitignore</code> precisa listar as saídas do DVC explicitamente (ver Anexo C).</p>",
"A": [
  ("p", "Montar o <b>esqueleto do projeto</b>: pastas por etapa, ambiente isolado e reprodutível, configuração central e logging."),
  ("p", "<b>Problema que resolve:</b> no notebook, caminhos, constantes e versões ficam espalhados e implícitos. Aqui eles ganham um lugar único: para mudar uma pasta ou um parâmetro, muda-se uma linha."),
],
"B": [
  ("Ambiente virtual", "virtual environment", "Uma instalação de Python isolada por projeto, para as versões de um não quebrarem o outro."),
  ("Versões fixadas", "pinned dependencies", "<code>pandas==3.1.1</code> em vez de <code>pandas</code>: o mesmo código instala o mesmo ambiente daqui a um ano."),
  ("Gestão de configuração", "configuration management", "Separar o que muda (parâmetros, caminhos, credenciais) do código que usa."),
  ("Pacote editável", "editable install", "<code>pip install -e .</code>: o código de <code>src/</code> é importável de qualquer lugar e mudanças valem sem reinstalar."),
  ("Registro de eventos", "logging", "Mensagens com horário, módulo e nível, gravadas em arquivo. Quando o pipeline roda sozinho, é o log que conta o que aconteceu."),
],
"C": [
  ("ol", [
    "Crie a pasta do projeto e rode <code>git init</code>.",
    "Crie as pastas (2.D) com um <code>.gitkeep</code> nas que começam vazias.",
    "Escreva o <code>pyproject.toml</code>, crie o ambiente e instale; depois fixe as versões que foram resolvidas.",
    "Escreva <code>src/config.py</code> (caminhos e esquema), <code>params.yaml</code> (parâmetros) e <code>src/__init__.py</code> (logging + .env).",
    "Escreva o <code>.gitignore</code> já com as pastas de dados e modelos (que serão do DVC).",
    "Faça o primeiro commit.",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "Versão do Python · onde fica o ambiente virtual · nomes das pastas (siga o projeto-exemplo do curso) · quais parâmetros vão para o <code>params.yaml</code> (todo número \"mágico\" do notebook)."),
],
"D": [
  ("data/{external,raw,preprocessed,processed}/", "Dados em camadas: cada camada é a saída de um estágio (fases 5 a 7).", "R"),
  ("src/&lt;etapa&gt;/&lt;modulo&gt;.py", "Um subpacote por etapa do ciclo (data_loading, data_validation...). Cada um com <code>__init__.py</code>.", "R"),
  ("src/config.py", "Caminhos de todos os arquivos (a partir da raiz), nomes das colunas e <code>load_params()</code>.", "RE"),
  ("src/utils.py", "Ler e gravar JSON e CSV sempre do mesmo jeito.", "R"),
  ("src/__init__.py", "Carrega o <code>.env</code> e configura o logging uma única vez.", "R"),
  ("params.yaml", "Todos os parâmetros por estágio. O DVC usa para saber o que refazer (fase 13).", "RE"),
  ("pyproject.toml", "Nome do pacote e dependências com versões fixas.", "RE"),
  (".gitignore · .env.example · .gitattributes", "O que o Git ignora; o modelo do .env; finais de linha padronizados.", "R"),
  ("models/ · artifacts/ · metrics/ · logs/", "Saídas: modelos, artefatos de suporte, métricas por estágio, logs.", "R"),
],
"E": [
  ("cmd", "Terminal, na pasta do projeto: ambiente e instalação", """$ git init
$ python -m venv .venv
$ .venv\\Scripts\\activate
$ pip install -e ".[dev]"
$ copy .env.example .env"""),
  ("code", "src/config.py · o essencial (projeto de referência)", """ROOT = Path(__file__).resolve().parents[1]

class Paths:
    @property
    def root(self) -> Path:                       # trocável nos testes
        return Path(os.getenv("FRAUD_PROJECT_ROOT", ROOT))
    raw = property(lambda self: self.root / "data" / "raw" / "raw.csv")
    model_final = property(lambda self: self.root / "models" / "model.joblib")
    ...

PATHS = Paths()
ALVO = "Target"                                    # ESPECÍFICO do problema
COLUNAS_MODELO = COMPONENTES_PCA + ["hist_saldo", "hora_do_dia"]

def load_params(path=None) -> dict:
    with open(path or PATHS.params, encoding="utf-8") as f:
        return yaml.safe_load(f)"""),
  ("code", "params.yaml · um trecho", """seed: 42
preprocess_data:
  test_size: 0.2
  val_size: 0.25
train:
  balanceamento: class_weight   # class_weight | smote | none
tune:
  n_iter: 20
decision_policy:
  custo_revisao: 5.0"""),
  ("code", "src/__init__.py · logging do projeto inteiro", """logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(_LOGS / "pipeline.log", encoding="utf-8")],
)"""),
  ("code", ".gitignore · as saídas do DVC declaradas explicitamente", """/data/external/transferencias.zip
/data/raw/*.csv
/data/preprocessed/*.csv
/data/processed/*.csv
/models/*.joblib
mlruns/
mlflow.db
.env"""),
  ("note", "warn", "Por que fixar as versões depois de instalar", "Instale sem versão, veja o que foi resolvido (<code>pip freeze</code>) e copie para o <code>pyproject.toml</code>. Atenção ao Docker: o ambiente Linux pode resolver diferente do Windows. No projeto, o MLflow 3.18 exige <code>gunicorn&lt;27</code> só no Linux; a versão correta saiu de <code>uv pip compile pyproject.toml --python-platform x86_64-manylinux_2_28</code>."),
],
"F": [
  ("cmd", "Conferir o ambiente e a configuração", """$ python -c "from src.config import PATHS, load_params; print(PATHS.raw); print(load_params()['seed'])"
$ git status --short"""),
  ("out", "Resposta esperada", """C:\\...\\mlops_seu_projeto\\data\\raw\\raw.csv
42"""),
  ("note", "warn", "Erros comuns", "<code>ModuleNotFoundError: src</code>: faltou o <code>pip install -e .</code> ou o comando não está na raiz. <code>FileNotFoundError: params.yaml</code>: o <code>config.py</code> não está calculando a raiz a partir do próprio arquivo."),
],
"G": ["<code>pip install -e \".[dev]\"</code> roda sem erro num ambiente novo.",
      "<code>from src.config import PATHS, load_params</code> funciona de qualquer pasta.",
      "Uma chamada a <code>logging.getLogger(...).info()</code> aparece no terminal e em <code>logs/pipeline.log</code>.",
      "O primeiro commit não contém dados nem modelos."],
"H": "Com o esqueleto pronto, a <b>Fase 5</b> traz os dados para dentro do projeto e os põe sob versionamento com o DVC.",
"check": [
  ("Criar pastas e <code>git init</code>", "Ter a estrutura padrão", "Árvore de pastas + <code>.git</code>", "<code>git status</code>", "Repositório criado"),
  ("Escrever o <code>pyproject.toml</code> e instalar", "Ambiente reprodutível", "<code>pyproject.toml</code> com versões fixas", "<code>pip install -e \".[dev]\"</code> num ambiente novo", "Instala sem erro"),
  ("Escrever <code>config.py</code> e <code>params.yaml</code>", "Um lugar para caminhos e parâmetros", "<code>src/config.py</code>, <code>params.yaml</code>", "Comando do 4.F", "Imprime o caminho e a semente"),
  ("Configurar o logging", "Registrar a execução", "<code>src/__init__.py</code>", "Logar uma mensagem de teste", "Mensagem no terminal e no <code>pipeline.log</code>"),
  ("Escrever o <code>.gitignore</code> e o 1º commit", "Não versionar dados no Git", "<code>.gitignore</code>, commit", "<code>git show --stat HEAD</code>", "Nenhum CSV ou modelo no commit"),
],
},
# ============================================================================================= F5
{
"n": 5, "grupo": P2, "curto": "Dados versionados", "entregavel": "Base sob DVC + estágio load_data → data/raw/raw.csv",
"titulo": "Trazer os dados para o projeto e versioná-los com o DVC",
"lede": "Dados grandes não vão para o Git: o DVC guarda o conteúdo num cache e o Git guarda só um \"ponteiro\" com o hash. O primeiro estágio, load_data, lê a fonte e grava o dado bruto, intacto.",
"pastas": "data/external/ · data/raw/ · .dvc/ · src/data_loading/",
"ref": "<p><b>Implementado e validado.</b> <code>transferencias.zip</code> (68 MB) em <code>data/external/</code>, versionado com <code>dvc add</code>. O estágio <code>load_data</code> gera <code>data/raw/raw.csv</code> com <b>284.807 linhas × 31 colunas</b>. Particularidade desta máquina: <code>dvc.exe</code> é bloqueado por política do Windows, então todos os comandos usam <code>python -m dvc</code>.</p>",
"A": [
  ("p", "Colocar a base original sob <b>versionamento de dados</b> e criar o estágio que gera o <b>dado bruto</b>."),
  ("p", "<b>Problema que resolve:</b> \"qual base gerou este modelo?\". Com o hash no Git, cada commit sabe exatamente que dados usou, e <code>git checkout</code> + <code>dvc checkout</code> voltam código e dados juntos."),
],
"B": [
  ("Versionamento de dados", "data versioning", "Guardar versões de arquivos grandes fora do Git, referenciadas por hash."),
  ("Cache do DVC", "DVC cache", "Onde o conteúdo fica de fato (<code>.dvc/cache</code> ou outra pasta). Pode ser enviado a um remoto com <code>dvc push</code>."),
  ("Arquivo .dvc", "DVC pointer file", "Arquivo pequeno com o hash do dado; é ele que vai para o Git."),
  ("Dado bruto", "raw data", "A cópia intacta da fonte. Nunca é editada: toda limpeza acontece nos estágios seguintes."),
  ("Remoto", "remote storage", "Um armazenamento compartilhado (S3, Google Drive, DagsHub) para o cache. Opcional neste roteiro."),
],
"C": [
  ("ol", [
    "Copie a base original para <code>data/external/</code>.",
    "Rode <code>dvc init</code> e <code>dvc add</code> na base; faça o commit do <code>.dvc</code>.",
    "Escreva <code>src/data_loading/load_data.py</code> no molde <i>função + main()</i>.",
    "Rode o estágio e confira o dado bruto.",
    "(Opcional) Configure um remoto e rode <code>dvc push</code>.",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "Onde fica o cache (dentro do projeto ou fora, se a pasta é sincronizada pelo OneDrive) · formato do dado bruto (CSV é simples e legível; Parquet é menor) · se haverá remoto."),
],
"D": [
  ("data/external/&lt;base&gt;", "A fonte original. Entra no Git só o <code>&lt;base&gt;.dvc</code>.", "E"),
  ("data/raw/raw.csv", "O dado bruto: saída do estágio 1, entrada da validação.", "R"),
  ("src/data_loading/load_data.py", "<code>fetch_data(fonte)</code> e <code>save_data(df, destino)</code>; <code>main()</code> lê o caminho do <code>params.yaml</code>.", "RE"),
  (".dvc/ · .dvcignore", "Configuração do DVC (vai para o Git); o cache não vai.", "R"),
],
"E": [
  ("cmd", "Terminal, na raiz: iniciar o DVC e versionar a base", """$ python -m dvc init
$ python -m dvc config core.autostage true
$ python -m dvc add data/external/transferencias.zip
$ git add data/external/transferencias.zip.dvc .dvc .dvcignore
$ git commit -m "Base original sob DVC\""""),
  ("code", "src/data_loading/load_data.py · o molde de todo estágio", """def fetch_data(source: Path) -> pd.DataFrame:
    logger.info(f"Lendo a base original de {source}")
    return pd.read_csv(source)               # o pandas lê o CSV de dentro do zip

def save_data(data: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)

def main() -> None:
    params = load_params()
    raw = fetch_data(PATHS.root / params["load_data"]["source"])
    save_data(raw, PATHS.raw)

if __name__ == "__main__":
    main()"""),
  ("note", "key", "Caso a pasta esteja no OneDrive", "Pastas sincronizadas duplicam o cache do DVC e deixam tudo lento. No projeto de referência, o cache foi para fora: <code>python -m dvc cache dir --local %LOCALAPPDATA%\\dvc-cache\\projeto</code> (o <code>--local</code> grava em <code>.dvc/config.local</code>, que não vai para o Git)."),
],
"F": [
  ("cmd", "Rodar o estágio e conferir", """$ python -m src.data_loading.load_data
$ python -m dvc status"""),
  ("out", "Resposta esperada (projeto de referência)", """... src.data_loading.load_data:32 - INFO - Base lida: 284,807 linhas x 31 colunas
... src.data_loading.load_data:39 - INFO - Gravando os dados brutos em ...\\data\\raw\\raw.csv
Data and pipelines are up to date."""),
  ("note", "warn", "Erros comuns", "<code>dvc: Permission denied</code> no Windows: use <code>python -m dvc</code>. <code>output is already tracked by SCM</code>: o arquivo entrou no Git antes; tire com <code>git rm --cached</code> e declare no <code>.gitignore</code>."),
],
"G": ["A base original está em <code>data/external/</code> e o <code>.dvc</code> está commitado.",
      "<code>python -m src.data_loading.load_data</code> gera <code>data/raw/raw.csv</code> com o número de linhas e colunas da fonte.",
      "<code>git ls-files</code> não lista nenhum CSV nem o zip."],
"H": "O dado bruto ainda pode ter colunas faltando, nulos ou duplicatas. A <b>Fase 6</b> cria a validação que impede o pipeline de seguir com dados ruins e faz a limpeza e a separação treino/validação/teste.",
"check": [
  ("Versionar a base com <code>dvc add</code>", "Saber qual base gerou cada modelo", "<code>&lt;base&gt;.dvc</code> commitado", "<code>git ls-files | grep .dvc</code>", "Ponteiro no Git, conteúdo no cache"),
  ("Escrever <code>load_data.py</code>", "Primeiro estágio no molde padrão", "<code>src/data_loading/load_data.py</code>", "<code>python -m src.data_loading.load_data</code>", "Roda sem erro"),
  ("Conferir o dado bruto", "Garantir que nada foi alterado", "<code>data/raw/raw.csv</code>", "Linhas × colunas no log", "Mesmo formato da fonte"),
  ("Conferir o Git", "Dados fora do Git", "—", "<code>git ls-files</code>", "Nenhum CSV/zip listado"),
],
},
# ============================================================================================= F6
{
"n": 6, "grupo": P2, "curto": "Validação e pré-processamento", "entregavel": "validate_data (bloqueante) + preprocess_data → train/val/test",
"titulo": "Validar os dados (e parar se estiverem ruins), limpar e separar treino, validação e teste",
"lede": "Num pipeline automático ninguém olha as saídas das células: as regras de qualidade precisam estar no código e interromper a execução. Em seguida, a limpeza e a separação, feitas de forma a não vazar informação do teste.",
"pastas": "src/data_validation/ · src/data_preprocessing/ · data/preprocessed/ · metrics/",
"ref": "<p><b>Implementado e validado.</b> <code>validate_dataframe</code> com 11 checagens (bloqueantes e alertas), reutilizada pela inferência. Pré-processamento: 1.081 duplicatas removidas, separação estratificada <b>170.235 / 56.745 / 56.746</b> transações com <b>284 / 94 / 95</b> fraudes, idêntica ao notebook. <code>id_transacao</code> e <code>train_val_ids.csv</code> preservam a ordem usada pela validação cruzada.</p>",
"A": [
  ("p", "Criar dois estágios: <b>validação</b> (para o pipeline se a base estiver fora do contrato) e <b>pré-processamento</b> (limpeza e separação)."),
  ("p", "<b>Problemas que resolvem:</b> \"garbage in, garbage out\" silencioso, e vazamento de dados pela separação mal feita (duplicatas dos dois lados, proporção de classes diferente entre as partes)."),
],
"B": [
  ("Validação de dados", "data validation", "Regras que a base precisa cumprir: colunas, tipos, nulos, faixas, domínio do alvo."),
  ("Checagem bloqueante × alerta", "blocking check vs warning", "Bloqueante para o pipeline (coluna faltando); alerta só registra (taxa de fraude estranha)."),
  ("Separação estratificada", "stratified split", "Cada parte mantém a proporção das classes. Essencial com classes raras."),
  ("Treino / validação / teste", "train / validation / test", "Treino ajusta; validação escolhe; teste mede uma única vez."),
  ("Contrato de dados", "data contract", "O esquema esperado. A mesma regra vale no treino e na inferência."),
],
"C": [
  ("ol", [
    "Liste as regras de qualidade a partir da fase 2 (o que o notebook conferia \"olhando\").",
    "Escreva <code>validate_dataframe(df, require_target)</code>, que devolve um relatório; o <code>main()</code> grava o JSON e levanta erro se algo bloqueante falhar.",
    "Escreva a limpeza decidida no notebook (no exemplo: remover duplicatas).",
    "Escreva a separação com as mesmas sementes e proporções do notebook, estratificando pelo alvo.",
    "Grave as partes e um resumo (contagens por parte) em <code>metrics/</code>.",
  ]),
  ("fig", "divisao_dados", "Quem lê cada parte dos dados no projeto de referência. É este desenho que garante que nenhuma escolha vê o teste."),
  ("note", "tip", "Decisões a tomar nesta fase", "Quais regras bloqueiam · tipo de separação: aleatória estratificada (exemplo), temporal (séries, dados com ordem) ou por grupo (vários registros do mesmo cliente) · proporções · semente."),
],
"D": [
  ("src/data_validation/validate_data.py", "<code>validate_dataframe</code> + <code>DataValidationError</code>. A mecânica é reutilizável; as regras são do problema.", "RE"),
  ("src/data_preprocessing/preprocess_data.py", "<code>remover_duplicatas</code>, <code>split_data</code>, <code>resumo_split</code>.", "RE"),
  ("data/preprocessed/{train,val,test}.csv", "As três partes, com <code>id_transacao</code>.", "R"),
  ("data/preprocessed/train_val_ids.csv", "A ordem de treino + validação (a validação cruzada depende dela).", "E"),
  ("metrics/data_validation.json · preprocessing.json", "Relatório das checagens e contagens por parte.", "R"),
],
"E": [
  ("code", "validate_data.py · o padrão de checagem", """def _check(nome, ok, detalhe, bloqueante=True):
    return {"checagem": nome, "ok": bool(ok), "bloqueante": bloqueante, "detalhe": detalhe}

def validate_dataframe(df, require_target=True) -> dict:
    esperadas = [c for c in COLUNAS_BRUTAS if require_target or c != ALVO]
    faltando = [c for c in esperadas if c not in df.columns]
    checagens = [_check("colunas obrigatórias", not faltando, f"faltando: {faltando}")]
    ...
    checagens.append(_check("taxa de fraude entre 0,01% e 5%", 0.0001 <= taxa <= 0.05,
                            f"taxa = {taxa:.4%}", bloqueante=False))   # alerta, não para
    valido = all(c["ok"] for c in checagens if c["bloqueante"])
    return {"valido": valido, "checagens": checagens, "resumo": resumo}

def main():
    relatorio = validate_dataframe(read_csv(PATHS.raw))
    save_json(relatorio, PATHS.metrics_dir / "data_validation.json")
    if not relatorio["valido"]:
        raise DataValidationError("A validação dos dados falhou")   # o DVC para aqui"""),
  ("code", "preprocess_data.py · limpar antes de separar, estratificar sempre", """def split_data(df, test_size, val_size, seed):
    temp, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df[ALVO])
    train, val = train_test_split(temp, test_size=val_size, random_state=seed, stratify=temp[ALVO])
    return {"train": train, "val": val, "test": test, "train_val": temp}

def main():
    limpo, removidas = remover_duplicatas(read_csv(PATHS.raw))   # ANTES do split
    limpo.insert(0, ID, limpo.index)
    ..."""),
  ("note", "key", "Reprodutibilidade fina: ler o CSV em round trip", "<code>pd.read_csv(path, float_precision=\"round_trip\")</code> devolve cada número exatamente como foi gravado. Sem isso, o último dígito de alguns valores muda e o modelo deixa de ser idêntico entre execuções (<code>src/utils.read_csv</code>)."),
],
"F": [
  ("cmd", "Rodar os dois estágios", """$ python -m src.data_validation.validate_data
$ python -m src.data_preprocessing.preprocess_data
$ pytest tests/unit/test_data.py -q"""),
  ("out", "Resposta esperada (projeto de referência)", """[OK] colunas obrigatórias: 31 colunas presentes
[OK] sem valores nulos: 0 nulos
[OK] duplicatas (removidas no próximo estágio): 1081 linhas duplicadas
Validação dos dados concluída: base aprovada
Duplicatas removidas: 1,081 | restam 283,726 transações (473 fraudes)
train: 170,235 transações, 284 fraudes (0.167%)
  val: 56,745 transações, 94 fraudes (0.166%)
 test: 56,746 transações, 95 fraudes (0.167%)
12 passed"""),
  ("p", "Teste de que a validação <b>bloqueia</b> de verdade: apague uma coluna de uma cópia da base e rode <code>validate_dataframe</code>; <code>valido</code> deve ser <code>False</code> (é o que <code>test_validacao_reprova_coluna_faltando</code> faz)."),
],
"G": ["A validação reprova uma base com coluna faltando, nulos, valor fora da faixa ou alvo fora do domínio, e o estágio termina com erro.",
      "As três partes têm a proporção de classes da base (estratificação) e nenhum ID em comum.",
      "As contagens por parte batem com o notebook.",
      "Os testes unitários de dados passam."],
"H": "Com as partes separadas e limpas, a <b>Fase 7</b> cria as features com uma função sem estado, que será reaproveitada na inferência.",
"check": [
  ("Escrever <code>validate_dataframe</code>", "Parar o pipeline com dados ruins", "<code>validate_data.py</code>, <code>data_validation.json</code>", "Base com coluna removida → <code>valido=False</code>", "Bloqueia e registra"),
  ("Escrever a limpeza do notebook", "Preservar a lógica (ex.: duplicatas)", "<code>preprocess_data.py</code>", "Log com a contagem removida", "Número igual ao do notebook"),
  ("Separar estratificado", "Partes comparáveis e sem vazamento", "<code>data/preprocessed/*.csv</code>", "<code>pytest tests/unit/test_data.py</code>", "Proporção igual e IDs disjuntos"),
  ("Conferir com o notebook", "Provar que a lógica foi preservada", "<code>metrics/preprocessing.json</code>", "Comparar contagens", "Contagens idênticas"),
],
},
# ============================================================================================= F7
{
"n": 7, "grupo": P2, "curto": "Engenharia de atributos", "entregavel": "build_features sem estado, usada no treino e na inferência",
"titulo": "Criar as features com uma função que o treino e a produção compartilham",
"lede": "A transformação de features precisa ser idêntica no treino e na produção. Transformações sem estado ficam numa função compartilhada; as que aprendem com os dados (escala, codificação) ficam dentro do Pipeline do modelo.",
"pastas": "src/feature_engineering/ · data/processed/",
"ref": "<p><b>Implementado e validado.</b> <code>build_features</code> cria <code>hora_do_dia</code> e remove o <code>Timestamp</code>, mantendo as 30 variáveis na ordem de <code>COLUNAS_MODELO</code>. A <b>mesma</b> função é chamada por <code>FraudPredictor</code> na inferência. A escala da Regressão Logística fica dentro do Pipeline (fase 8).</p>",
"A": [
  ("p", "Escrever a engenharia de atributos como uma <b>função reutilizável</b> e gravar as partes prontas para o modelo."),
  ("p", "<b>Problema que resolve:</b> o <i>training-serving skew</i>, quando a produção transforma os dados de um jeito ligeiramente diferente do treino e o modelo erra sem ninguém saber por quê."),
],
"B": [
  ("Engenharia de atributos", "feature engineering", "Criar e transformar variáveis para o modelo."),
  ("Transformação sem estado", "stateless transform", "Não aprende nada dos dados (ex.: hora a partir do timestamp). Pode rodar em qualquer parte sem vazamento."),
  ("Transformação com estado", "stateful transform / fit", "Aprende com os dados (média, desvio, categorias). Deve ser ajustada só no treino: dentro do Pipeline."),
  ("Desvio treino-produção", "training-serving skew", "Diferença entre a transformação do treino e a da produção."),
],
"C": [
  ("ol", [
    "Separe as transformações do notebook em <b>sem estado</b> (vão para <code>build_features</code>) e <b>com estado</b> (vão para o Pipeline do modelo, fase 8).",
    "Escreva <code>build_features(df)</code>: recebe o formato bruto, devolve as colunas do modelo numa ordem fixa.",
    "Aplique às três partes e grave em <code>data/processed/</code>.",
    "Garanta que a função funciona sem a coluna alvo (na inferência ela não existe).",
  ]),
  ("note", "tip", "Decisões a tomar nesta fase", "O que é sem estado e o que é com estado · a lista e a ordem final das colunas do modelo (vai para <code>config.COLUNAS_MODELO</code>)."),
],
"D": [
  ("src/feature_engineering/engineer_features.py", "<code>hora_do_dia</code> e <code>build_features</code>; <code>main()</code> processa as três partes.", "E"),
  ("data/processed/{train,val,test}.csv", "As colunas do modelo + <code>id_transacao</code> + <code>Target</code>.", "R"),
  ("src/config.py · COLUNAS_MODELO", "A ordem exata das colunas que o modelo recebe.", "E"),
],
"E": [
  ("code", "engineer_features.py (projeto de referência)", """def hora_do_dia(timestamp: pd.Series) -> pd.Series:
    return (timestamp % 86_400) / 3600                  # 0 a 24 h, repete todo dia

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.assign(hora_do_dia=hora_do_dia(df[COLUNA_TEMPO]))[COLUNAS_MODELO]
    if ID in df.columns:
        features.insert(0, ID, df[ID])
    if ALVO in df.columns:                              # na inferência não há alvo
        features = features.assign(**{ALVO: df[ALVO]})
    return features"""),
  ("note", "key", "Em outros problemas", "Codificação de categorias, imputação e escala são <b>com estado</b>: use <code>ColumnTransformer</code>/<code>Pipeline</code> do scikit-learn dentro do modelo. Assim o objeto salvo já carrega a transformação, e a inferência não precisa repeti-la."),
],
"F": [
  ("cmd", "Rodar e testar", """$ python -m src.feature_engineering.engineer_features
$ pytest tests/unit/test_data.py -q -k features"""),
  ("out", "Resposta esperada", """... Features criadas: 30 variáveis (hist_saldo, hora_do_dia no fim)
... Engenharia de atributos concluída
1 passed"""),
],
"G": ["<code>build_features</code> devolve exatamente <code>COLUNAS_MODELO</code>, na mesma ordem, com e sem a coluna alvo.",
      "Nenhuma transformação que aprende com os dados está fora de um Pipeline.",
      "<code>data/processed/</code> tem as três partes."],
"H": "Os dados estão prontos. A <b>Fase 8</b> treina os algoritmos do notebook, escolhe o melhor e grava os modelos.",
"check": [
  ("Classificar as transformações", "Evitar vazamento e skew", "Lista sem estado × com estado", "Revisar contra o notebook", "Nenhuma com estado fora do Pipeline"),
  ("Escrever <code>build_features</code>", "Uma função para treino e produção", "<code>engineer_features.py</code>", "Teste unitário de colunas e ordem", "Passa com e sem alvo"),
  ("Gerar <code>data/processed/</code>", "Entrada do treino", "3 CSVs", "<code>python -m src.feature_engineering.engineer_features</code>", "3 arquivos com as colunas do modelo"),
],
},
]
