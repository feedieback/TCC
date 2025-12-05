<?php
// dashboard_adafruit.php - Versão corrigida com carrossel automático 7s e renomeação de feeds

$ADAFRUIT_USERNAME = 'xxxxxxxxxx';
$ADAFRUIT_KEY = 'xxxxxxxxxx';

// map de nomes para exibição (conforme solicitado)
$FEED_NAMES = [
    "temperatura"  => "Temperatura Ambiente",
    "umidade-ar"   => "Umidade do Ar",
    "umidade-solo" => "Umidade do Solo",
    "indice-luz"   => "Luminosidade"
];

$DEFAULT_LIMIT = 200;
$CURL_TIMEOUT  = 10;

function adafruit_api_request($path, $method = 'GET', $data = null) {
    global $ADAFRUIT_USERNAME, $ADAFRUIT_KEY, $CURL_TIMEOUT;
    $url = "https://io.adafruit.com/api/v2/{$ADAFRUIT_USERNAME}{$path}";

    $ch = curl_init($url);
    $headers = [
        "X-AIO-Key: {$ADAFRUIT_KEY}",
        "Accept: application/json"
    ];
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
    curl_setopt($ch, CURLOPT_TIMEOUT, $CURL_TIMEOUT);

    if ($method === 'POST' && $data !== null) {
        curl_setopt($ch, CURLOPT_POST, true);
        curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($data));
        $headers[] = "Content-Type: application/json";
        curl_setopt($ch, CURLOPT_HTTPHEADER, $headers);
    }

    $resp = curl_exec($ch);
    $err  = curl_error($ch);
    $code = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($resp === false) {
        http_response_code(500);
        echo json_encode(['error' => "Curl error: {$err}"]);
        exit;
    }

    $decoded = json_decode($resp, true);
    return $decoded ?? ['raw' => $resp, 'status' => $code];
}

// Endpoints AJAX
if (php_sapi_name() !== 'cli' && isset($_GET['action'])) {
    header("Content-Type: application/json; charset=utf-8");
    $action = $_GET['action'];

    if ($action === "feeds") {
        $r = adafruit_api_request("/feeds");
        echo json_encode($r);
        exit;
    }

    if ($action === "data" && isset($_GET['feed_key'])) {
        $feed  = rawurlencode($_GET['feed_key']);
        $limit = intval($_GET['limit'] ?? $DEFAULT_LIMIT);
        if ($limit <= 0 || $limit > 1000) $limit = $DEFAULT_LIMIT;
        $r = adafruit_api_request("/feeds/{$feed}/data?limit={$limit}");
        echo json_encode($r);
        exit;
    }

    echo json_encode(['error' => 'Ação inválida']);
    exit;
}
?>
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Dashboard Integrada — Adafruit → Página PHP</title>

<!-- Chart.js -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.3.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3"></script>

<style>
:root{
  --bg:#0f1724; --card:#0b1220; --muted:#9aa7bf;
  --accent1:#6ee7b7; --accent2:#60a5fa;
}
body{ background: linear-gradient(180deg,#071024 0%, #071220 100%); color:#e6eef8; font-family:Inter,system-ui,Segoe UI,Roboto,Arial; margin:0; padding:18px;}
.top{ display:flex; gap:12px; align-items:center; margin-bottom:14px;}
.title-block{ flex:1;}
.controls{ display:flex; gap:8px; align-items:center;}
.card-grid{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin-top:12px;}
.card{ background:var(--card); padding:12px; border-radius:12px; box-shadow:0 6px 18px rgba(2,6,23,0.6); border:1px solid rgba(255,255,255,0.03);}
.card h3{ margin:0 0 6px; font-size:14px; color:var(--muted);}
.card .value{ font-size:20px; font-weight:700; margin-bottom:8px;}
.charts-area{ display:grid; grid-template-columns:2fr 1fr; gap:12px; margin-top:12px;}
.bigcard{ padding:14px; border-radius:14px;}
canvas{ width:100% !important; height:260px !important; display:block;}
.small-canvas{ height:150px !important;}
.feed-list{ max-height:220px; overflow:auto; margin-top:8px; border-top:1px dashed rgba(255,255,255,0.03); padding-top:8px;}
.feed-item{ padding:8px; cursor:pointer; border-radius:8px; display:flex; justify-content:space-between; align-items:center;}
.feed-item:hover{ background: rgba(255,255,255,0.02); }
.muted{ color:var(--muted); font-size:13px;}
.btn{ background:linear-gradient(90deg,var(--accent2),var(--accent1)); color:#012; border:none; padding:8px 12px; border-radius:10px; font-weight:700; cursor:pointer;}

/* CARROSSEL */
.carousel-container{ position:relative; width:100%; overflow:hidden; border-radius:14px; background:transparent; padding:8px;}
.carousel-track{ display:flex; transition:transform 0.6s ease-in-out; }
.carousel-item{ min-width:100%; box-sizing:border-box; padding:6px; }
.carousel-controls{ display:flex; gap:8px; justify-content:center; margin-top:8px;}
.carousel-dot{ width:10px; height:10px; border-radius:50%; background:rgba(255,255,255,0.12); cursor:pointer;}
.carousel-dot.active{ background:var(--accent1); box-shadow:0 2px 6px rgba(0,0,0,0.5); }
</style>
</head>
<body>
  <div class="top">
    <div class="title-block">
      <h1>Sistema de Monitoramento Agrícola</h1>
      <div class="muted">Usuário Adafruit: <strong><?php echo htmlspecialchars($ADAFRUIT_USERNAME);?></strong></div>
    </div>
    <div class="controls">
      <button class="btn" id="btnRefresh">Atualizar Agora</button>
    </div>
  </div>

  <div class="card-grid" id="cardsContainer"></div>

  <div class="charts-area">
    <div class="bigcard card" id="mainChartCard">
      <h3 id="mainChartTitle">Carrossel de Gráficos</h3>

      <div class="carousel-container">
        <div class="carousel-track" id="carouselTrack">
          <!-- slides serão inseridos via JS -->
        </div>
      </div>
      <div class="carousel-controls" id="carouselDots"></div>
    </div>

    <div class="card" style="height:100%;">
      <h3>Feeds Disponíveis</h3>
      <div class="feed-list" id="feedList">Carregando feeds...</div>
      <div style="height:12px;"></div>
      <h3>Configurações</h3>
      <div class="muted">Pontos por feed</div>
      <input type="range" id="limitRange" min="20" max="800" value="200" step="10" />
      <div class="muted" style="margin-top:6px;">Atualização automática a cada
        <select id="autoInterval">
          <option value="0">Desligado</option>
          <option value="5000">5s</option>
          <option value="7000" selected>7s</option>
          <option value="10000">10s</option>
          <option value="30000">30s</option>
        </select>
      </div>
    </div>
  </div>

  <div class="footer">Criado por Victor — Integração Adafruit → Página PHP</div>

<script>
// ---------- Config / globals ----------
let feeds = [];
let carousels = {};         // map feedKey -> {canvas, chart}
let carouselOrder = [];     // ordem dos feed keys
let currentIndex = 0;
let autoTimer = null;
let globalLimit = parseInt(document.getElementById('limitRange').value || 200);

// Helper AJAX
async function ajax(action, params = {}) {
  const url = new URL(location.href);
  url.searchParams.set('action', action);
  for (const k in params) url.searchParams.set(k, params[k]);
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error('Erro na requisição: ' + res.status);
  return res.json();
}

// escape
function escapeHtml(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ---------- Init ----------
document.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('btnRefresh').addEventListener('click', loadAll);
  document.getElementById('limitRange').addEventListener('input', () => document.getElementById('limitRange').title = document.getElementById('limitRange').value);
  document.getElementById('autoInterval').addEventListener('change', setupAuto);

  try {
    await loadFeeds();
    await loadAll();
    setupAuto(); // start auto if selected
  } catch(e) {
    console.error(e);
    alert('Erro: ' + e.message);
  }
});

// ---------- load feeds ----------
async function loadFeeds(){
  const list = document.getElementById('feedList');
  list.innerHTML = 'Carregando...';
  const data = await ajax('feeds');
  if (!Array.isArray(data)) { list.innerHTML = '<div class="muted">Erro ao obter feeds</div>'; console.error(data); return; }
  feeds = data;
  list.innerHTML = '';
  feeds.forEach(f => {
    const div = document.createElement('div');
    div.className = 'feed-item';
    const displayName = mapName(f.key, f.name);
    div.innerHTML = `<div><strong>${escapeHtml(displayName)}</strong><div class="muted">${escapeHtml(f.key)}</div></div><div class="muted">${f.updated_at? f.updated_at.split('T')[0] : ''}</div>`;
    div.onclick = () => { jumpToFeed(f.key); };
    list.appendChild(div);
  });
}

// mapeia o nome (usa o PHP-provided mapping via data attribute? we'll reconstruct same mapping in JS)
function mapName(key, originalName){
  const map = {
    "temperatura":"Temperatura Ambiente",
    "umidade-ar":"Umidade do Ar",
    "umidade-solo":"Umidade do Solo",
    "indice-luz":"Luminosidade"
  };
  return map[key] || originalName || key;
}

// ---------- Load all feed data, create cards + carousel slides ----------
async function loadAll(){
  const container = document.getElementById('cardsContainer');
  container.innerHTML = 'Carregando dados...';
  globalLimit = parseInt(document.getElementById('limitRange').value || 200);
  container.innerHTML = '';

  // reset carousel internals
  carousels = {};
  carouselOrder = [];
  const track = document.getElementById('carouselTrack');
  track.innerHTML = '';
  document.getElementById('carouselDots').innerHTML = '';

  for (const f of feeds) {
    try {
      const d = await ajax('data', { feed_key: f.key, limit: globalLimit });
      const points = Array.isArray(d) ? d.slice().reverse() : [];
      createOrUpdateCard(container, f, points);
      createCarouselSlide(f, points);
    } catch (e) {
      console.warn('Erro ao carregar feed', f.key, e);
    }
  }

  // if there's at least one slide, ensure first is visible
  if (carouselOrder.length) {
    showSlide(0);
    createDots();
  }
}

// ---------- small cards (sparklines) ----------
function createOrUpdateCard(container, feedMeta, points){
  const key = feedMeta.key;
  const last = points.length ? points[points.length - 1] : null;
  const lastValue = last ? last.value : '---';
  let card = document.getElementById('card_' + key);
  if (!card){
    card = document.createElement('div');
    card.className = 'card';
    card.id = 'card_' + key;
    const title = mapName(key, feedMeta.name);
    card.innerHTML = `<h3>${escapeHtml(title)}</h3>
      <div class="value" id="val_${key}">${escapeHtml(String(lastValue))}</div>
      <div class="muted">Feed: <strong>${escapeHtml(key)}</strong></div>
      <canvas id="spark_${key}" class="small-canvas"></canvas>`;
    container.appendChild(card);
  } else {
    document.getElementById('val_' + key).innerText = lastValue;
  }

  // draw sparkline
  try {
    const ctx = document.getElementById('spark_' + key).getContext('2d');
    const labels = points.map(p => p.created_at);
    const values = points.map(p => Number(p.value));
    if (ctx._chart) ctx._chart.destroy();
    ctx._chart = new Chart(ctx, {
      type:'line',
      data:{ labels, datasets:[{ data:values, fill:true, tension:0.3, pointRadius:0 }]},
      options:{ animation:false, plugins:{legend:{display:false}}, scales:{x:{display:false}, y:{display:false}}, elements:{ line:{borderWidth:1.6, borderColor:'rgba(255,255,255,0.6)', backgroundColor:'rgba(255,255,255,0.04)'} } }
    });
  } catch(e){ console.warn(e); }
}

// ---------- carousel slide creation ----------
function createCarouselSlide(feedMeta, points){
  const key = feedMeta.key;
  carouselOrder.push(key);
  const track = document.getElementById('carouselTrack');

  const item = document.createElement('div');
  item.className = 'carousel-item';
  // create a title + canvas
  const title = mapName(key, feedMeta.name);
  item.innerHTML = `<h4 class="carousel-title">${escapeHtml(title)}</h4><canvas id="main_canvas_${key}" style="width:100%;height:260px;"></canvas>`;
  track.appendChild(item);

  // store initial data object (we'll create charts lazily when slide shown to avoid many charts at once)
  carousels[key] = { feedMeta, points, canvasId: 'main_canvas_' + key, chart: null };
}

// ---------- create dots below carousel ----------
function createDots(){
  const dots = document.getElementById('carouselDots');
  dots.innerHTML = '';
  carouselOrder.forEach((k, idx) => {
    const d = document.createElement('div');
    d.className = 'carousel-dot' + (idx === 0 ? ' active' : '');
    d.onclick = () => { showSlide(idx); resetAuto(); };
    dots.appendChild(d);
  });
}

// ---------- show a slide by index ----------
function showSlide(idx){
  if (!carouselOrder.length) return;
  const track = document.getElementById('carouselTrack');
  if (idx < 0) idx = 0;
  if (idx >= carouselOrder.length) idx = carouselOrder.length - 1;
  currentIndex = idx;
  track.style.transform = `translateX(-${idx * 100}%)`;

  // update dots active
  const dots = document.getElementById('carouselDots').children;
  for (let i=0;i<dots.length;i++){ dots[i].classList.toggle('active', i===idx); }

  // render chart for this slide (lazy init)
  const key = carouselOrder[idx];
  const item = carousels[key];
  if (item && !item.chart){
    renderChartForKey(key, item.points);
  }
}

// ---------- render chart for a feed key into its canvas ----------
function renderChartForKey(feedKey, points){
  const obj = carousels[feedKey];
  if (!obj) return;
  const canvas = document.getElementById(obj.canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const labels = points.map(p => p.created_at);
  const values = points.map(p => Number(p.value));

  // create gradient
  const grad = ctx.createLinearGradient(0,0,0,260);
  grad.addColorStop(0, 'rgba(96,165,250,0.9)');
  grad.addColorStop(1, 'rgba(110,231,183,0.06)');

  const cfg = {
    type:'line',
    data:{ labels, datasets:[{ label: mapName(feedKey,''), data: values, borderWidth:2.2, tension:0.25, fill:true, backgroundColor: grad, borderColor:'rgba(96,165,250,0.95)', pointRadius:2 }]},
    options:{
      maintainAspectRatio:false, animation:{duration:300},
      plugins:{ legend:{display:false}, tooltip:{mode:'index', intersect:false} },
      scales:{
        x:{ type:'time', time:{ tooltipFormat:'dd/MM HH:mm' }, grid:{ color:'rgba(255,255,255,0.03)' } },
        y:{ ticks:{ color:'#cfe8ff' }, grid:{ color:'rgba(255,255,255,0.02)' } }
      }
    }
  };

  obj.chart = new Chart(ctx, cfg);
}

// ---------- navigation helpers ----------
function nextSlide(){ showSlide((currentIndex + 1) % carouselOrder.length); }
function prevSlide(){ showSlide((currentIndex - 1 + carouselOrder.length) % carouselOrder.length); }

// ---------- auto rotation ----------
function setupAuto(){
  if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
  const val = parseInt(document.getElementById('autoInterval').value || 0);
  if (val > 0){
    autoTimer = setInterval(() => { nextSlide(); }, val);
  }
}

function resetAuto(){
  setupAuto();
}

// skip to feed by key (called from feed list)
function jumpToFeed(key){
  const idx = carouselOrder.indexOf(key);
  if (idx >= 0) showSlide(idx);
  resetAuto();
}

// intervalo em milissegundos (aqui 5 segundos)
const AUTO_UPDATE_MS = 10000;

// função que atualiza tudo automaticamente
async function autoUpdateCharts() {
    try {
        // atualiza todos os cards + sparklines
        await loadAll();

        // atualiza o gráfico sendo exibido no carrossel
        if (currentFeedKey) {
            const limit = parseInt(document.getElementById('limitRange').value || 200);
            const d = await ajax('data', { feed_key: currentFeedKey, limit });
            const points = Array.isArray(d) ? d.slice().reverse() : [];
            renderMainChart(currentFeedKey, points);
        }

    } catch (e) {
        console.warn("Erro ao atualizar automaticamente:", e);
    }
}

// inicia a atualização automática
setInterval(autoUpdateCharts, AUTO_UPDATE_MS);


</script>
</body>
</html>
