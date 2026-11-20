/* Guia-MLOps: progresso por projeto, os 4 estados e o mapa do roteiro.
 *
 * Os estados NÃO são escolhidos à mão: saem do checklist de cada fase.
 *   Já feito     = todos os itens do checklist da fase marcados (cada item é um critério de validação)
 *   Esta fase    = a primeira fase que ainda não está feita
 *   Próxima fase = a fase seguinte a ela
 *   Depois       = as demais fases ainda não feitas
 * O progresso fica no navegador (localStorage), separado por nome de projeto: dá para usar o mesmo
 * guia em vários projetos e trocar entre eles no campo "Projeto" da barra do topo.
 */
(function () {
  var PREFIXO = "guia-mlops-v1::";
  var CHAVE_PROJETO = "guia-mlops-projeto-atual";
  var CHAVE_LISTA = "guia-mlops-projetos";
  var ESTADOS = {
    "d-done": "Já feito", "d-now": "Esta fase", "d-next": "Próxima fase", "d-todo": "Depois"
  };

  function ler(k, padrao) {
    try { var v = localStorage.getItem(k); return v === null ? padrao : v; } catch (e) { return padrao; }
  }
  function gravar(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }

  var sections = Array.prototype.slice.call(document.querySelectorAll("section.phase"));
  var fases = sections.filter(function (s) { return s.dataset.fase; });
  var campoProjeto = document.getElementById("projeto");
  var listaProjetos = document.getElementById("projetos");
  var projeto = ler(CHAVE_PROJETO, "meu-projeto");
  var state = {};
  var boxes = [];

  function carregarEstado() {
    try { state = JSON.parse(ler(PREFIXO + projeto, "{}")) || {}; } catch (e) { state = {}; }
    boxes.forEach(function (b) { b.checked = !!state[b.dataset.k]; });
  }
  function salvarEstado() { gravar(PREFIXO + projeto, JSON.stringify(state)); }
  function atualizarListaProjetos() {
    var nomes = [];
    try { nomes = JSON.parse(ler(CHAVE_LISTA, "[]")); } catch (e) { nomes = []; }
    if (nomes.indexOf(projeto) < 0) { nomes.push(projeto); gravar(CHAVE_LISTA, JSON.stringify(nomes)); }
    if (listaProjetos) {
      listaProjetos.innerHTML = "";
      nomes.forEach(function (n) { var o = document.createElement("option"); o.value = n; listaProjetos.appendChild(o); });
    }
  }

  /* ---- checkboxes ---- */
  sections.forEach(function (sec) {
    Array.prototype.forEach.call(sec.querySelectorAll('input[type="checkbox"]'), function (box, i) {
      box.dataset.k = sec.id + ":" + i;
      box.addEventListener("change", function () {
        state[box.dataset.k] = box.checked;
        salvarEstado();
        render();
      });
      boxes.push(box);
    });
  });

  /* ---- sumário lateral ---- */
  var toc = document.getElementById("toc");
  var links = {};
  var lastGroup = null;
  sections.forEach(function (sec) {
    var badge = sec.querySelector(".phase-n");
    var raw = badge ? badge.textContent.trim() : "";
    var short = /^FASE/i.test(raw) ? "F" + raw.replace(/^FASE\s*/i, "")
      : (/^ANEXO/i.test(raw) ? "A." + raw.slice(-1) : raw.slice(0, 5));
    if (sec.dataset.group && sec.dataset.group !== lastGroup) {
      var g = document.createElement("div");
      g.className = "toc-group";
      g.textContent = sec.dataset.group;
      toc.appendChild(g);
      lastGroup = sec.dataset.group;
    }
    var a = document.createElement("a");
    a.href = "#" + sec.id;
    a.innerHTML = "<b>" + short + "</b><span>" + (sec.dataset.title || "") + "</span>";
    toc.appendChild(a);
    links[sec.id] = a;
  });

  /* ---- os 4 estados ---- */
  function caixasDa(sec) { return Array.prototype.slice.call(sec.querySelectorAll('input[type="checkbox"]')); }
  function estados() {
    var r = {}, atual = null;
    fases.forEach(function (sec) {
      var n = +sec.dataset.fase, lista = caixasDa(sec);
      var feita = lista.length > 0 && lista.every(function (b) { return b.checked; });
      if (feita) { r[n] = "d-done"; }
      else if (atual === null) { atual = n; r[n] = "d-now"; }
      else if (n === atual + 1) { r[n] = "d-next"; }
      else { r[n] = "d-todo"; }
    });
    return r;
  }
  function relativo(n, fase) {            // mapa "onde você está no roteiro" ao terminar a fase `fase`
    if (n < fase) { return "d-done"; }
    if (n === fase) { return "d-now"; }
    if (n === fase + 1) { return "d-next"; }
    return "d-todo";
  }

  /* ---- mapa do roteiro (um SVG modelo, pintado conforme o estado) ---- */
  var tpl = document.getElementById("rot-tpl");
  function pintar(host, classeDe) {
    if (!tpl || !host) { return; }
    host.innerHTML = "";
    var svg = tpl.content.firstElementChild.cloneNode(true);
    Array.prototype.forEach.call(svg.querySelectorAll("g.bx[data-k]"), function (g) {
      var n = parseInt(g.getAttribute("data-k").slice(1), 10);
      var c = classeDe(n);
      if (c) { g.classList.add(c); g.setAttribute("aria-label", "Fase " + n + ": " + ESTADOS[c]); }
    });
    host.appendChild(svg);
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "arch-zoom";
    btn.textContent = "ampliar";
    btn.addEventListener("click", function () { abrirMapa(svg, host.getAttribute("data-titulo") || "Roteiro MLOps"); });
    host.appendChild(btn);
  }
  Array.prototype.forEach.call(document.querySelectorAll('.rot-host[data-modo="fase"]'), function (host) {
    var fase = parseInt(host.getAttribute("data-fase"), 10);
    pintar(host, function (n) { return relativo(n, fase); });
  });

  var dlg = null;
  function abrirMapa(svg, titulo) {
    if (!dlg) {
      dlg = document.createElement("dialog");
      dlg.className = "arch-dlg";
      dlg.innerHTML = '<div class="arch-dlg-bar"><span></span><button type="button">fechar</button></div><div class="arch-dlg-body"></div>';
      dlg.querySelector("button").addEventListener("click", function () { dlg.close(); });
      dlg.addEventListener("click", function (e) { if (e.target === dlg) { dlg.close(); } });
      document.body.appendChild(dlg);
    }
    dlg.querySelector(".arch-dlg-bar span").textContent = titulo;
    var corpo = dlg.querySelector(".arch-dlg-body");
    corpo.innerHTML = "";
    corpo.appendChild(svg.cloneNode(true));
    if (dlg.showModal) { dlg.showModal(); } else { dlg.setAttribute("open", ""); }
  }

  /* ---- render: contador, estados nas fases, no sumário, nas linhas e no mapa ao vivo ---- */
  var fill = document.getElementById("progFill");
  var num = document.getElementById("progNum");
  function selo(el, classe, texto) {
    el.classList.remove("d-done", "d-now", "d-next", "d-todo");
    el.classList.add(classe);
    el.textContent = "■ " + (texto || ESTADOS[classe]);
  }
  function render() {
    var done = boxes.filter(function (b) { return b.checked; }).length;
    fill.style.width = (boxes.length ? Math.round(done / boxes.length * 100) : 0) + "%";
    num.textContent = done + " / " + boxes.length;
    var est = estados();

    sections.forEach(function (sec) {
      var lista = caixasDa(sec);
      var ok = lista.filter(function (b) { return b.checked; }).length;
      var counter = sec.querySelector("[data-count-for]");
      if (counter && lista.length) {
        counter.textContent = ok + "/" + lista.length;
        counter.classList.toggle("full", ok === lista.length);
      }
      var n = +sec.dataset.fase, c = est[n], link = links[sec.id];
      if (link) {
        link.classList.remove("st-done", "st-now", "st-next", "st-todo");
        if (c) { link.classList.add("st-" + c.slice(2)); link.title = ESTADOS[c]; }
        link.classList.toggle("done", lista.length > 0 && ok === lista.length);
      }
      var badge = sec.querySelector(".estado-fase");
      if (badge && c) { selo(badge, c); }
      // cada linha do checklist: feita = "Já feito"; senão herda o estado da fase
      Array.prototype.forEach.call(sec.querySelectorAll("tr[data-item]"), function (tr) {
        var b = tr.querySelector('input[type="checkbox"]'), lbl = tr.querySelector(".estado-linha");
        if (lbl && c) { selo(lbl, b && b.checked ? "d-done" : c); }
      });
    });

    // painel de acompanhamento (tabela-resumo do mapa)
    Array.prototype.forEach.call(document.querySelectorAll("[data-estado-de]"), function (el) {
      var c = est[+el.getAttribute("data-estado-de")];
      if (c) { selo(el, c); }
    });
    var atual = Object.keys(est).filter(function (k) { return est[k] === "d-now"; })[0];
    var aqui = document.getElementById("voce-esta");
    if (aqui) {
      aqui.innerHTML = atual
        ? 'Você está na <a href="#f' + atual + '">Fase ' + atual + "</a>. " + done + " de " + boxes.length + " critérios validados no projeto <b></b>."
        : "Todas as fases validadas no projeto <b></b>.";
      aqui.querySelector("b").textContent = projeto;
    }
    Array.prototype.forEach.call(document.querySelectorAll('.rot-host[data-modo="progresso"]'), function (host) {
      pintar(host, function (n) { return est[n]; });
    });
  }

  /* ---- projeto atual ---- */
  if (campoProjeto) {
    campoProjeto.value = projeto;
    campoProjeto.addEventListener("change", function () {
      var novo = campoProjeto.value.trim();
      if (!novo) { campoProjeto.value = projeto; return; }
      projeto = novo;
      gravar(CHAVE_PROJETO, projeto);
      atualizarListaProjetos();
      carregarEstado();
      render();
    });
  }

  document.getElementById("reset").addEventListener("click", function () {
    if (!window.confirm("Desmarcar todos os itens do checklist do projeto \"" + projeto + "\"?")) { return; }
    state = {};
    salvarEstado();
    boxes.forEach(function (b) { b.checked = false; });
    render();
  });

  /* ---- copiar comandos (o prompt em .p fica de fora) ---- */
  Array.prototype.forEach.call(document.querySelectorAll(".copy"), function (btn) {
    btn.addEventListener("click", function () {
      var pre = btn.parentNode.querySelector("pre");
      if (!pre) { return; }
      var text = Array.prototype.map.call(pre.childNodes, function (n) {
        return (n.nodeType === 1 && n.classList.contains("p")) ? "" : n.textContent;
      }).join("");
      var ok = function () {
        btn.textContent = "copiado";
        btn.classList.add("ok");
        setTimeout(function () { btn.textContent = "copiar"; btn.classList.remove("ok"); }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(ok, function () { copiarFallback(text, ok); });
      } else { copiarFallback(text, ok); }
    });
  });
  function copiarFallback(text, ok) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); ok(); } catch (e) {}
    document.body.removeChild(ta);
  }

  /* ---- seção ativa no sumário ---- */
  if ("IntersectionObserver" in window) {
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          Object.keys(links).forEach(function (id) { links[id].classList.remove("on"); });
          if (links[en.target.id]) { links[en.target.id].classList.add("on"); }
        }
      });
    }, { rootMargin: "-80px 0px -70% 0px" });
    sections.forEach(function (s) { obs.observe(s); });
  }

  /* ---- tema: todo clique inverte; o sistema decide só o estado inicial ---- */
  var TEMA = "guia-mlops-tema";
  var btnTema = document.getElementById("tema");
  function sistemaEscuro() {
    try { return window.matchMedia("(prefers-color-scheme: dark)").matches; } catch (e) { return false; }
  }
  function aplicarTema(t, guardar) {
    document.documentElement.setAttribute("data-theme", t);
    if (btnTema) { btnTema.textContent = t === "dark" ? "Tema: escuro" : "Tema: claro"; }
    if (guardar) { gravar(TEMA, t); }
  }
  if (btnTema) {
    var salvo = ler(TEMA, "");
    aplicarTema(salvo === "dark" || salvo === "light" ? salvo : (sistemaEscuro() ? "dark" : "light"), false);
    btnTema.addEventListener("click", function () {
      aplicarTema(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark", true);
    });
  }

  /* ---- impressão: abre os blocos recolhíveis e devolve o estado depois ---- */
  var recolhidos = [];
  window.addEventListener("beforeprint", function () {
    recolhidos = Array.prototype.filter.call(document.querySelectorAll("details"), function (d) { return !d.open; });
    recolhidos.forEach(function (d) { d.open = true; });
  });
  window.addEventListener("afterprint", function () {
    recolhidos.forEach(function (d) { d.open = false; });
    recolhidos = [];
  });

  atualizarListaProjetos();
  carregarEstado();
  render();
})();
