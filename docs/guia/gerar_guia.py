"""Gera o Guia-MLOps.html (na raiz do projeto) a partir das fontes desta pasta.

Por que um gerador? O guia tem 17 fases com a MESMA estrutura (A a H + checklist + mapa). Escrever
esse HTML à mão seria repetir centenas de linhas de marcação e errar em alguma. Aqui:
  conteudo_*.py  -> o texto de cada fase (o que você edita no dia a dia)
  *.svg          -> os diagramas (editáveis em qualquer editor de SVG ou de texto)
  estilo_base.css + ESTILO_EXTRA -> o visual (o do guia original, mais os 4 estados)
  guia.js        -> progresso, estados e mapa
O resultado é UM arquivo HTML autocontido, que abre direto no navegador.

Executar (na raiz do projeto):  python docs/guia/gerar_guia.py
"""
import re
from html import escape
from pathlib import Path

PASTA = Path(__file__).resolve().parent
RAIZ = PASTA.parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(PASTA))
from conteudo_anexos import ANEXOS, MAPA, REUSO  # noqa: E402
from conteudo_fases_1 import FASES as F1  # noqa: E402
from conteudo_fases_2 import FASES as F2  # noqa: E402
from conteudo_fases_3 import FASES as F3  # noqa: E402

FASES = F1 + F2 + F3
NOMES = {f["n"]: f["curto"] for f in FASES}

ESTILO_EXTRA = """
  /* ---------- 4 estados de acompanhamento ---------- */
  .estado { display: inline-block; font-family: "IBM Plex Mono", monospace; font-size: 11.5px; font-weight: 600;
            border-radius: 4px; padding: 2px 7px; white-space: nowrap; }
  .estado.d-done { color: #155232; background: #DEF0E6; }
  .estado.d-now  { color: #B45309; background: #FFE7C7; box-shadow: inset 0 0 0 2px #FF8A00; }
  .estado.d-next { color: #0A6A72; background: #DBECED; box-shadow: inset 0 0 0 1.5px #0A8A94; }
  .estado.d-todo { color: #5A6B70; background: #ECEFF0; }
  [data-theme="dark"] .estado.d-todo { color: #9FB0B5; background: #1D2A2E; }
  .leg-estados { display: flex; flex-wrap: wrap; gap: 8px 16px; margin: 10px 0 0; align-items: center; font-size: 13.5px; }
  .toc a { position: relative; }
  .toc a.st-done b { color: #2E9E5B; } .toc a.st-now b { color: #FF8A00; } .toc a.st-next b { color: #0A8A94; }
  .toc a.st-done::after, .toc a.st-now::after, .toc a.st-next::after { content: ""; position: absolute; right: 2px; top: 50%;
      width: 8px; height: 8px; margin-top: -4px; border-radius: 2px; }
  .toc a.st-done::after { background: #2E9E5B; } .toc a.st-now::after { background: #FF8A00; }
  .toc a.st-next::after { background: transparent; box-shadow: inset 0 0 0 2px #0A8A94; }
  .fase-meta { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; margin-top: 8px; font-size: 13px; color: var(--ink-2); }
  .fase-meta code { font-size: 12px; }
  /* ---------- projeto de referência ---------- */
  .ref { margin: 16px 0 0; border: 1px solid var(--line); border-left: 4px solid #5B6B8C; border-radius: 8px;
         padding: 12px 16px; background: var(--surface); font-size: 14.5px; }
  .ref > b:first-child { display: block; font-family: "IBM Plex Mono", monospace; font-size: 11px; letter-spacing: .12em;
         text-transform: uppercase; color: #5B6B8C; margin-bottom: 4px; }
  [data-theme="dark"] .ref > b:first-child { color: #A9B6D3; }
  .ref p { margin: 0; } .ref p + p { margin-top: 6px; }
  .tag { display: inline-block; font-family: "IBM Plex Mono", monospace; font-size: 10.5px; font-weight: 600; letter-spacing: .04em;
         border-radius: 3px; padding: 1px 6px; vertical-align: 1px; white-space: nowrap; }
  .tag-r { color: #0B5394; background: #DCEBFA; } .tag-e { color: #8A3A0A; background: #FCE8D8; }
  [data-theme="dark"] .tag-r { color: #9CC8F2; background: #12304A; } [data-theme="dark"] .tag-e { color: #F2B98C; background: #3A2414; }
  .sub { font-family: "Archivo", sans-serif; font-size: 12px; letter-spacing: .1em; text-transform: uppercase; color: var(--accent); margin: 0 0 4px; }
  .step ol.passos, .step ul.lista { margin: 8px 0 0; padding-left: 1.4em; }
  .step ol.passos li + li, .step ul.lista li + li { margin-top: 5px; }
  .criterios { list-style: none; padding: 0; margin: 8px 0 0; }
  .criterios li { padding-left: 26px; position: relative; } .criterios li + li { margin-top: 6px; }
  .criterios li::before { content: "✔"; position: absolute; left: 4px; color: var(--ok); font-weight: 700; }
  /* ---------- checklist em tabela ---------- */
  table.checklist td { vertical-align: top; font-size: 13.5px; }
  table.checklist { table-layout: fixed; min-width: 760px; }
  table.checklist th, table.checklist td { padding: 10px 10px; }
  table.checklist th:nth-child(1) { width: 40px; }
  table.checklist th:nth-child(2) { width: 24%; }
  table.checklist th:nth-child(3) { width: 20%; }
  table.checklist th:nth-child(4) { width: 22%; }
  table.checklist th:nth-child(6) { width: 112px; }
  table.checklist td:first-child { min-width: 0; text-align: center; }
  table.checklist td { overflow-wrap: anywhere; }
  table.checklist input { width: 18px; height: 18px; accent-color: #2E9E5B; cursor: pointer; }
  table.checklist tr:has(input:checked) td { background: color-mix(in srgb, #2E9E5B 7%, transparent); }
  table.checklist code { font-size: 11.5px; }
  table.checklist .obj { font-size: 12.5px; color: var(--ink-2); }
  table.checklist td:last-child { white-space: nowrap; }
  /* ---------- diagramas ---------- */
  .rot-host, .fig { position: relative; margin: 14px 0 0; border: 1px solid var(--line); border-radius: 10px; background: #FFFFFF; padding: 8px; overflow-x: auto; }
  .rot-host svg, .fig svg { display: block; width: 100%; height: auto; min-width: 760px; }
  .fig-cap { font-size: 13px; color: var(--ink-2); margin: 6px 2px 0; }
  .voce-esta { margin: 12px 0 0; font-size: 15px; }
  .arch-dlg-body svg { width: 100%; height: auto; }
"""


# ------------------------------------------------------------------ blocos
def esc(t):
    return escape(t, quote=False)


def bloco_cmd(label, code, copiar=True):
    linhas = []
    for linha in code.strip("\n").split("\n"):
        if linha.startswith("$ "):
            linhas.append('<span class="p">$ </span>' + esc(linha[2:]))
        else:
            linhas.append(esc(linha))
    botao = '<button class="copy" type="button">copiar</button>' if copiar else ""
    return f'<div class="cmd"><span class="cmd-label">{label}</span>{botao}<pre>' + "\n".join(linhas) + "</pre></div>"


def render_blocos(blocos):
    out = []
    for b in blocos:
        tipo = b[0]
        if tipo == "p":
            out.append(f"<p>{b[1]}</p>")
        elif tipo == "cmd":
            out.append(bloco_cmd(b[1], b[2]))
        elif tipo == "code":       # trecho de código (copiável, sem prompt)
            out.append(bloco_cmd(b[1], b[2]))
        elif tipo in ("out", "bad"):
            classe = "good" if tipo == "out" else "bad"
            out.append(f'<div class="out {classe}"><span class="out-label">{b[1]}</span><pre>{esc(b[2].strip(chr(10)))}</pre></div>')
        elif tipo == "note":
            out.append(f'<div class="note n-{b[1]}"><b>{b[2]}</b><p>{b[3]}</p></div>')
        elif tipo == "ol":
            out.append('<ol class="passos">' + "".join(f"<li>{i}</li>" for i in b[1]) + "</ol>")
        elif tipo == "ul":
            out.append('<ul class="lista">' + "".join(f"<li>{i}</li>" for i in b[1]) + "</ul>")
        elif tipo == "table":
            out.append(tabela(b[1], b[2]))
        elif tipo == "fig":
            out.append(figura(b[1], b[2]))
        elif tipo == "details":
            out.append(f'<details class="fallback"><summary>{b[1]}</summary>' + render_blocos(b[2]) + "</details>")
        else:
            raise ValueError(f"bloco desconhecido: {tipo}")
    return "\n".join(out)


def tabela(cab, linhas, classe=""):
    th = "".join(f"<th>{c}</th>" for c in cab)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in l) + "</tr>" for l in linhas)
    return f'<div class="tablewrap"><table class="{classe}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def svg_inline(nome):
    s = (PASTA / f"{nome}.svg").read_text(encoding="utf-8")
    return re.sub(r"^<\?xml[^>]*>\s*", "", s)


def figura(nome, legenda):
    return f'<div class="fig">{svg_inline(nome)}</div><p class="fig-cap">{legenda}</p>'


TAG = {"R": '<span class="tag tag-r">REUTILIZÁVEL</span>', "E": '<span class="tag tag-e">ESPECÍFICO</span>',
       "RE": '<span class="tag tag-r">REUTILIZÁVEL</span> <span class="tag tag-e">AJUSTAR</span>'}
LEGENDA = ('<div class="leg-estados"><span class="estado d-done">■ Já feito</span><span class="estado d-now">■ Esta fase</span>'
           '<span class="estado d-next">■ Próxima fase</span><span class="estado d-todo">■ Depois</span></div>')


def passo(n, letra, titulo, onde, corpo):
    pid = f"passo-{n}-{letra.lower()}"
    return (f'<div class="step" id="{pid}"><div class="step-top"><a class="step-id" href="#{pid}">{n}.{letra}</a>'
            f'<h3>{titulo}</h3><span class="where {onde[0]}">{onde[1]}</span></div>{corpo}</div>')


def render_fase(f):
    n = f["n"]
    conceitos = tabela(["Conceito", "Em inglês", "O que é e onde entra no ciclo de ML"],
                       [(f"<b>{a}</b>", f"<i>{b}</i>", c) for a, b, c in f["B"]])
    arquivos = tabela(["Pasta ou arquivo", "Para que serve · responsabilidade", "Tipo"],
                      [(f"<code>{p}</code>", d, TAG[t]) for p, d, t in f["D"]])
    linhas = []
    for i, (ativ, obj, entreg, teste, crit) in enumerate(f["check"]):
        linhas.append(f'<tr data-item="{i}"><td><input type="checkbox" aria-label="{escape(re.sub("<[^>]+>", "", ativ))}"></td>'
                      f'<td><b>{ativ}</b><br><span class="obj">Objetivo: {obj}</span></td><td>{entreg}</td><td>{teste}</td><td>{crit}</td>'
                      f'<td><span class="estado estado-linha"></span></td></tr>')
    checklist = ('<div class="tablewrap"><table class="checklist"><thead><tr><th>✓</th><th>Atividade · objetivo</th>'
                 '<th>Entregáveis</th><th>Como testar</th><th>Critério de conclusão</th><th>Estado</th></tr></thead><tbody>'
                 + "".join(linhas) + "</tbody></table></div>")
    prox = n + 1
    prox_link = (f'<a href="#f{prox}">Fase {prox} · {NOMES[prox]}</a>' if prox in NOMES
                 else '<a href="#anexo-a">Anexo A · Aplicar o roteiro a um novo projeto</a>')
    partes = [
        f'<section class="phase" id="f{n}" data-fase="{n}" data-group="{f["grupo"]}" data-title="{f["curto"]}">',
        '<div class="phase-head">', f'<span class="phase-n">FASE {n}</span>', '<div>',
        f'<h2>{f["titulo"]}</h2>', f'<p>{f["lede"]}</p>',
        f'<div class="fase-meta"><span>No seu projeto: <span class="estado estado-fase d-todo">■ Depois</span></span>'
        f'<span>Pastas e arquivos: <code>{f["pastas"]}</code></span></div>',
        '</div>', f'<span class="phase-count" data-count-for="f{n}"></span>', '</div>',
        f'<div class="ref"><b>No projeto de referência · mlops_detectando_fraudes_bancarias</b>{f["ref"]}</div>',
        passo(n, "A", "Objetivo da etapa", ("w-wsl", "POR QUÊ"), render_blocos(f["A"])),
        passo(n, "B", "Conceitos envolvidos", ("w-ssm", "CONCEITOS"), conceitos),
        passo(n, "C", "O que fazer na prática", ("w-console", "ROTEIRO"), render_blocos(f["C"])),
        passo(n, "D", "Estrutura de arquivos", ("w-web", "PASTAS"), arquivos),
        passo(n, "E", "Implementação", ("w-wsl", "CÓDIGO"), render_blocos(f["E"])),
        passo(n, "F", "Como testar", ("w-ssm", "TESTE"), render_blocos(f["F"])),
        passo(n, "G", "Critério de conclusão", ("w-kiro", "PRONTO SE"),
              '<ul class="criterios">' + "".join(f"<li>{c}</li>" for c in f["G"]) + "</ul>"),
        passo(n, "H", "Próxima etapa", ("w-web", "DEPOIS"), f'<p>{f["H"]}</p>'),
        f'<div class="step" id="check-{n}"><div class="step-top"><span class="step-id">{n}.✓</span><h3>Checklist da fase {n}</h3>'
        '<span class="where w-kiro">ACOMPANHAMENTO</span></div><div class="why"><b>Como usar</b> Marque cada item só depois de rodar '
        'o teste e ver o critério atendido. A fase vira <span class="estado d-done">■ Já feito</span> quando todos os itens '
        f'estiverem marcados; o estado de cada linha acompanha a fase.</div>{checklist}</div>',
        f'<div class="recap"><div class="recap-head">Onde esta fase fica no roteiro</div>'
        f'<div class="rot-host" data-modo="fase" data-fase="{n}" data-titulo="Roteiro · ao trabalhar na Fase {n}"></div>{LEGENDA}'
        f'<p class="recap-next"><b>Próxima</b> {prox_link}</p></div>',
        "</section>",
    ]
    return "\n".join(partes)


def render_secao(s):
    """Seções livres (mapa, reuso, anexos): cabeçalho + blocos."""
    head = (f'<section class="phase" id="{s["id"]}" data-group="{s["grupo"]}" data-title="{s["curto"]}">'
            f'<div class="phase-head"><span class="phase-n">{s["rotulo"]}</span><div><h2>{s["titulo"]}</h2>'
            f'<p>{s["lede"]}</p></div></div>')
    return head + render_blocos(s["blocos"]) + "</section>"


def main():
    css = (PASTA / "estilo_base.css").read_text(encoding="utf-8") + ESTILO_EXTRA
    js = (PASTA / "guia.js").read_text(encoding="utf-8")
    painel = tabela(["Fase", "Parte", "Entregável principal", "Estado no seu projeto"],
                    [(f'<a href="#f{f["n"]}">F{f["n"]} · {f["curto"]}</a>', f["grupo"], f["entregavel"],
                      f'<span class="estado d-todo" data-estado-de="{f["n"]}">■ Depois</span>') for f in FASES])
    # Seção MAPA: o roteiro pintado com o SEU progresso, a explicação dos estados e o painel de acompanhamento
    vivo = ('<div class="rot-host" data-modo="progresso" data-titulo="Roteiro · o seu progresso"></div>' + LEGENDA +
            '<p class="voce-esta" id="voce-esta"></p>')
    mapa_html = (f'<section class="phase" id="{MAPA["id"]}" data-group="{MAPA["grupo"]}" data-title="{MAPA["curto"]}">'
                 f'<div class="phase-head"><span class="phase-n">{MAPA["rotulo"]}</span><div><h2>{MAPA["titulo"]}</h2>'
                 f'<p>{MAPA["lede"]}</p></div></div>' + vivo + render_blocos(MAPA["blocos"]) +
                 '<h3 class="sub" style="margin-top:22px">Painel de acompanhamento</h3>' + painel + "</section>")

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Guia de laboratório para transformar um projeto de Ciência de Dados em um projeto com MLOps: 17 fases, cada uma com objetivo, conceitos, prática, arquivos, implementação, teste, critério de conclusão e checklist, usando como referência real o projeto de detecção de fraudes bancárias.">
<meta name="color-scheme" content="light dark">
<title>Guia MLOps</title>
<script>(function () {{ try {{ var t = localStorage.getItem("guia-mlops-tema"); if (t === "dark" || t === "light") {{ document.documentElement.setAttribute("data-theme", t); }} }} catch (e) {{}} }})();</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&amp;family=IBM+Plex+Mono:wght@400;500;600&amp;family=IBM+Plex+Sans:wght@400;500;600&amp;display=swap">
<style>{css}</style>
</head>
<body>
<div class="topbar"><div class="topbar-in">
  <div class="topbar-id"><strong>Guia MLOps · 17 fases</strong></div>
  <label class="topbar-proj" style="display:flex;gap:6px;align-items:center;font-family:'IBM Plex Mono',monospace;font-size:11.5px">Projeto
    <input id="projeto" list="projetos" style="width:120px;font:inherit;padding:3px 6px;border:1px solid var(--line);border-radius:4px;background:var(--surface);color:var(--ink)" title="O progresso é salvo separadamente para cada nome de projeto"></label>
  <datalist id="projetos"></datalist>
  <div class="prog"><span class="prog-num" id="progNum">0 / 0</span><div class="prog-track"><div class="prog-fill" id="progFill"></div></div></div>
  <button class="btn-ghost" id="tema" type="button">Tema: claro</button>
  <button class="btn-ghost" id="reset" type="button">Limpar progresso</button>
</div></div>
<div class="shell">
<nav class="toc" id="toc" aria-label="Fases do roteiro"><div class="toc-label">Roteiro</div></nav>
<main>
<header class="hero">
  <div class="eyebrow">Manual de laboratório · MLOps: da Ciência de Dados à Operação · projeto de referência: detecção de fraudes bancárias</div>
  <h1>Do notebook à operação: um roteiro de MLOps em 17 fases</h1>
  <p class="lede">Um roteiro para pegar um projeto de Ciência de Dados que vive num notebook e transformá-lo num sistema
  reprodutível, versionado, testado e servindo previsões. Cada fase traz: <b>objetivo</b>, <b>conceitos</b>, <b>o que fazer</b>,
  <b>arquivos</b>, <b>implementação</b>, <b>como testar</b>, <b>critério de conclusão</b> e um <b>checklist</b> que marca o seu progresso
  com quatro estados. O exemplo real é o projeto <code>mlops_detectando_fraudes_bancarias</code>; cada parte diz o que é
  <span class="tag tag-r">REUTILIZÁVEL</span> em qualquer projeto e o que é <span class="tag tag-e">ESPECÍFICO</span> do problema de fraude.</p>
  <div class="facts">
    <span class="fact"><b>referência</b> mlops_detectando_fraudes_bancarias</span>
    <span class="fact"><b>origem</b> notebook "Detectando Fraudes … - v2"</span>
    <span class="fact"><b>pipeline</b> DVC · 8 estágios</span>
    <span class="fact"><b>rastreio</b> MLflow · Model Registry</span>
    <span class="fact"><b>serviço</b> Flask · gunicorn · Docker</span>
    <span class="fact"><b>orquestração</b> Airflow</span>
    <span class="fact"><b>testes</b> pytest · CI</span>
    <span class="fact"><b>Python</b> 3.13</span>
  </div>
</header>
<template id="rot-tpl">{svg_inline("roteiro")}</template>
{mapa_html}
{render_secao(REUSO)}
{chr(10).join(render_fase(f) for f in FASES)}
{chr(10).join(render_secao(a) for a in ANEXOS)}
</main>
</div>
<script>
{js}
</script>
</body>
</html>
"""
    destino = RAIZ / "Guia-MLOps.html"
    destino.write_text(html, encoding="utf-8")
    print(f"Guia gerado: {destino} ({len(html) / 1024:.0f} KB, {len(FASES)} fases, "
          f"{sum(len(f['check']) for f in FASES)} itens de checklist)")


if __name__ == "__main__":
    main()
