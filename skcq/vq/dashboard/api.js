// VQ sweep dashboard — vanilla JS + Plotly. No frameworks.

const BLOCK_SIZES = [8, 10, 12, 16, 24, 32, 64, 128];
const K_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];
const FP_DTYPES = ['fp8_e5m2', 'fp8_e4m3', 'fp16', 'bf16', 'fp32'];

function fmtK(v) {
  if (v >= 1048576) return (v / 1048576) + 'M';
  if (v >= 1024) return (v / 1024) + 'K';
  return String(v);
}

function isKmeans(r) { return r.scheme && r.scheme.startsWith('kmeans_'); }

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  return r.json();
}
async function post(path) { return api(path, { method: 'POST' }); }

// ---------------------------------------------------------------------------
// Status bar + control buttons
// ---------------------------------------------------------------------------
function updateStatus(status) {
  const pct = status.total > 0 ? (status.completed / status.total * 100).toFixed(1) : 0;
  document.getElementById('status-text').textContent =
    `State: ${status.state} | Progress: ${status.completed}/${status.total} (${pct}%) | Failed: ${status.failed} | In queue: ${status.in_queue}`;
  const running = status.state === 'running' && !status.paused;
  const idle = status.state === 'idle' || status.state === 'stopped';
  const paused = status.state === 'paused';
  document.getElementById('btn-launch').disabled = !idle;
  document.getElementById('btn-pause').disabled = !running;
  document.getElementById('btn-resume').disabled = !paused;
  document.getElementById('btn-shutdown').disabled = idle;
}

async function pollStatus() {
  try { updateStatus(await api('/api/status')); } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Range panel
// ---------------------------------------------------------------------------
function makeCheckboxes(containerId, values, formatter) {
  const c = document.getElementById(containerId);
  c.innerHTML = '';
  for (const v of values) {
    const lbl = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = v;
    cb.checked = true;
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(' ' + (formatter ? formatter(v) : v)));
    c.appendChild(lbl);
  }
}

function makeResidualCheckboxes(container, values, formatter) {
  container.innerHTML = '';
  for (const v of values) {
    const lbl = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = v;
    cb.checked = true;
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(' ' + (formatter ? formatter(v) : v)));
    container.appendChild(lbl);
  }
}

function addResidualSlot() {
  const tmpl = document.getElementById('residual-template');
  const node = tmpl.content.cloneNode(true);
  const container = node.querySelector('.residual-block-sizes');
  const kContainer = node.querySelector('.residual-K');
  makeResidualCheckboxes(container, BLOCK_SIZES);
  makeResidualCheckboxes(kContainer, K_VALUES, fmtK);
  node.querySelector('.remove-residual').addEventListener('click', (e) => {
    e.target.closest('.residual-slot').remove();
  });
  document.getElementById('residuals').appendChild(node);
}

function getCheckedValues(container) {
  return Array.from(container.querySelectorAll('input[type=checkbox]:checked')).map(cb => Number(cb.value));
}

function buildRangeFromUI() {
  const metric = document.querySelector('input[name=primary-metric]:checked').value;
  const signSplit = document.querySelector('input[name=primary-sign-split]:checked').value === 'yes';
  const bsVals = getCheckedValues(document.getElementById('primary-block-sizes'));
  const kVals = getCheckedValues(document.getElementById('primary-K'));
  const scaleType = document.querySelector('input[name=scale-dtype-type]:checked').value;
  let scaleDtypes;
  if (scaleType === 'int') {
    const lo = Number(document.getElementById('bits-min').value);
    const hi = Number(document.getElementById('bits-max').value);
    scaleDtypes = [];
    for (let b = lo; b <= hi; b++) scaleDtypes.push('int' + b);
  } else {
    scaleDtypes = getCheckedValues(document.getElementById('fp-dtypes')).map(String);
    // fp-dtypes are strings already, but getCheckedValues returns Number — fix
    scaleDtypes = Array.from(document.querySelectorAll('#fp-dtypes input[type=checkbox]:checked')).map(cb => cb.value);
  }
  const range = {
    projection: ['gate', 'down'],  // fixed for now
    bpw_min: Number(document.getElementById('bpw-min').value),
    bpw_max: Number(document.getElementById('bpw-max').value),
    primary: {
      block_size: bsVals,
      K: kVals,
      metric: [metric],
      sign_split: [signSplit],
      scale_dtype: scaleDtypes,
    },
    residuals: [],
  };
  // Collect residual slots
  for (const slot of document.querySelectorAll('.residual-slot')) {
    const bs = getCheckedValues(slot.querySelector('.residual-block-sizes'));
    const k = getCheckedValues(slot.querySelector('.residual-K'));
    range.residuals.push({ block_size: bs, K: k });
  }
  return range;
}

async function applyRange() {
  const range = buildRangeFromUI();
  try {
    const resp = await fetch('/api/range', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(range),
    });
    const data = await resp.json();
    const banner = document.getElementById('apply-banner');
    banner.textContent = 'Changes will apply to next sweep';
    const est = document.getElementById('range-estimate');
    const workers = await api('/api/workers');
    const nWorkers = workers.length || 1;
    const hours = (data.est_configs * 30 / nWorkers / 3600).toFixed(1);
    est.textContent = `~${data.est_configs} configs | ~${hours}h est`;
  } catch (e) {
    document.getElementById('apply-banner').textContent = 'Error: ' + e;
  }
}

async function loadRange() {
  try {
    const r = await api('/api/range');
    // Update UI from range
    document.getElementById('bpw-min').value = r.bpw_min;
    document.getElementById('bpw-max').value = r.bpw_max;
    document.getElementById('bpw-text').textContent = `[${r.bpw_min}, ${r.bpw_max}]`;
    // Set metric
    if (r.primary.metric && r.primary.metric[0]) {
      const m = document.querySelector(`input[name=primary-metric][value="${r.primary.metric[0]}"]`);
      if (m) m.checked = true;
    }
    // Set sign split
    if (r.primary.sign_split) {
      const ss = document.querySelector(`input[name=primary-sign-split][value="${r.primary.sign_split[0] ? 'yes' : 'no'}"]`);
      if (ss) ss.checked = true;
    }
    // Set scale dtype
    if (r.primary.scale_dtype && r.primary.scale_dtype[0]) {
      const sd = r.primary.scale_dtype[0];
      if (sd.startsWith('int')) {
        document.querySelector('input[name=scale-dtype-type][value="int"]').checked = true;
      } else {
        document.querySelector('input[name=scale-dtype-type][value="fp"]').checked = true;
      }
    }
    toggleScaleDtype();
    // Residuals
    document.getElementById('residuals').innerHTML = '';
    for (const res of (r.residuals || [])) {
      addResidualSlot();
      const slots = document.querySelectorAll('.residual-slot');
      const slot = slots[slots.length - 1];
      // Set checked state for block sizes and K
      const bsBoxes = slot.querySelectorAll('.residual-block-sizes input[type=checkbox]');
      bsBoxes.forEach(cb => { cb.checked = res.block_size.includes(Number(cb.value)); });
      const kBoxes = slot.querySelectorAll('.residual-K input[type=checkbox]');
      kBoxes.forEach(cb => { cb.checked = res.K.includes(Number(cb.value)); });
    }
  } catch (e) { /* ignore on first load */ }
}

function toggleScaleDtype() {
  const type = document.querySelector('input[name=scale-dtype-type]:checked').value;
  document.getElementById('scale-int-row').classList.toggle('hidden', type !== 'int');
  document.getElementById('scale-fp-row').classList.toggle('hidden', type !== 'fp');
}

// ---------------------------------------------------------------------------
// Results charts
// ---------------------------------------------------------------------------
function paretoFrontier(rows) {
  const out = [];
  for (const r of rows) {
    const dom = rows.some(o =>
      o !== r && o.bits_per_weight <= r.bits_per_weight &&
      o.rel_fro_err <= r.rel_fro_err &&
      (o.bits_per_weight < r.bits_per_weight || o.rel_fro_err < r.rel_fro_err));
    if (!dom) out.push(r);
  }
  out.sort((a, b) => a.bits_per_weight - b.bits_per_weight);
  return out;
}

function renderPareto(results) {
  const projections = [...new Set(results.map(r => r.projection))];
  const traces = [];
  for (const proj of projections) {
    const projRows = results.filter(r => r.projection === proj);
    const km = projRows.filter(isKmeans);
    const ints = projRows.filter(r => !isKmeans(r));
    if (km.length > 0) {
      traces.push({
        x: km.map(r => r.bits_per_weight), y: km.map(r => r.rel_fro_err),
        mode: 'markers', name: `${proj} kmeans`,
        text: km.map(r => `${r.scheme}<br>bpw=${r.bits_per_weight.toFixed(3)}<br>err=${r.rel_fro_err.toExponential(3)}`),
        marker: {
          size: km.map(r => Math.max(5, Math.min(15, r.K / 4096 + 4))),
          color: km.map(r => r.block_size), colorscale: 'Viridis', showscale: true,
          colorbar: { title: 'block_size' }, opacity: 0.7,
        },
        hovertemplate: '%{text}<extra></extra>',
      });
    }
    if (ints.length > 0) {
      traces.push({
        x: ints.map(r => r.bits_per_weight), y: ints.map(r => r.rel_fro_err),
        mode: 'markers', name: `${proj} integer`,
        text: ints.map(r => r.scheme),
        marker: { symbol: 'x', size: 8, color: 'red', opacity: 0.6 },
        hovertemplate: '%{text}<br>bpw=%{x:.3f}<br>err=%{y:.6f}<extra></extra>',
      });
    }
    const pareto = paretoFrontier(projRows);
    traces.push({
      x: pareto.map(r => r.bits_per_weight), y: pareto.map(r => r.rel_fro_err),
      mode: 'lines', name: `${proj} Pareto`, line: { width: 2, dash: 'dash' }, hoverinfo: 'skip',
    });
  }
  Plotly.newPlot('pareto-chart', traces, {
    title: 'Pareto frontier: bpw vs error',
    xaxis: { title: 'bits per weight' }, yaxis: { title: 'rel Frobenius error' },
    hovermode: 'closest', height: 420,
  }, { responsive: true });
}

function renderHeatmap(results, projection) {
  const km = results.filter(r => isKmeans(r) && r.n_codebooks === 1 && r.projection === projection);
  const el = document.getElementById('heatmap-chart');
  if (km.length === 0) { el.innerHTML = '<p>No single-codebook results for ' + projection + '</p>'; return; }
  const bsVals = [...new Set(km.map(r => r.block_size))].sort((a, b) => a - b);
  const kVals = [...new Set(km.map(r => r.K))].sort((a, b) => a - b);
  const z = bsVals.map(bs => kVals.map(K => {
    const r = km.find(r => r.block_size === bs && r.K === K);
    return r ? r.rel_fro_err : null;
  }));
  const txt = bsVals.map(bs => kVals.map(K => {
    const r = km.find(r => r.block_size === bs && r.K === K);
    return r ? r.bits_per_weight.toFixed(2) : '';
  }));
  Plotly.newPlot(el, [{
    z, x: kVals.map(fmtK), y: bsVals.map(String), type: 'heatmap',
    colorscale: 'Viridis_r', colorbar: { title: 'error' },
    text: txt, texttemplate: '%{text}',
    hovertemplate: 'bs=%{y} K=%{x}<br>err=%{z:.6f}<br>bpw=%{text}<extra></extra>',
  }], {
    title: `${projection}: block_size × K → error`,
    xaxis: { title: 'K' }, yaxis: { title: 'block_size' }, height: 420,
  }, { responsive: true });
}

function renderBestTable(results) {
  const buckets = [[1.0,1.5],[1.5,2.0],[2.0,2.5],[2.5,3.0],[3.0,3.5],[3.5,4.0],[4.0,5.0],[5.0,6.0]];
  const projections = [...new Set(results.map(r => r.projection))].sort();
  let html = '<thead><tr><th>bucket</th>';
  for (const p of projections) html += `<th>${p} km</th><th>${p} int</th><th>winner</th>`;
  html += '</tr></thead><tbody>';
  for (const [lo, hi] of buckets) {
    html += `<tr><td>[${lo.toFixed(1)}-${hi.toFixed(1)})</td>`;
    for (const p of projections) {
      const rows = results.filter(r => r.projection === p && lo <= r.bits_per_weight && r.bits_per_weight < hi);
      const km = rows.filter(isKmeans);
      const ints = rows.filter(r => !isKmeans(r));
      const bk = km.length > 0 ? Math.min(...km.map(r => r.rel_fro_err)) : null;
      const bi = ints.length > 0 ? Math.min(...ints.map(r => r.rel_fro_err)) : null;
      html += `<td>${bk !== null ? bk.toFixed(6) : '-'}</td>`;
      html += `<td>${bi !== null ? bi.toFixed(6) : '-'}</td>`;
      let w = '-';
      if (bk !== null && bi !== null) w = bk < bi ? '<td class="winner-kmeans">km</td>' : '<td class="winner-int">int</td>';
      else html += '<td>-</td>';
      if (w !== '-') html += w; else html = html; // already added
    }
    html += '</tr>';
  }
  html += '</tbody>';
  document.getElementById('best-table').innerHTML = html;
}

async function pollResults() {
  try {
    const data = await api('/api/results');
    const results = data.results || [];
    if (results.length > 0) {
      renderPareto(results);
      const projSelect = document.getElementById('heatmap-projection');
      const projections = [...new Set(results.map(r => r.projection))];
      if (projSelect.options.length === 0) {
        for (const p of projections) {
          const opt = document.createElement('option');
          opt.value = p; opt.textContent = p;
          projSelect.appendChild(opt);
        }
      }
      renderHeatmap(results, projSelect.value || projections[0]);
      renderBestTable(results);
    }
  } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------------------
// GPU cards
// ---------------------------------------------------------------------------
function renderGPUCards(workers) {
  const container = document.getElementById('gpu-cards-container');
  container.innerHTML = '';
  for (const w of workers) {
    const card = document.createElement('div');
    card.className = 'gpu-card';
    const dev = w.devices && w.devices[0] ? w.devices[0] : { name: '?', total_vram_mb: 0 };
    const hb = w.last_heartbeat;
    const devStats = hb && hb.devices && hb.devices[0] ? hb.devices[0] : { used_mb: 0, util_pct: 0 };
    const vramPct = dev.total_vram_mb > 0 ? (devStats.used_mb / dev.total_vram_mb) : 0;
    const barLen = 20;
    const filled = Math.round(vramPct * barLen);
    const barStr = '█'.repeat(filled) + '░'.repeat(barLen - filled);
    const vramStr = `${barStr} ${devStats.used_mb} / ${dev.total_vram_mb} MB`;
    card.innerHTML = `
      <div class="gpu-card-header">
        <span class="status-dot ${w.connected ? 'connected' : ''}"></span>
        <strong>${w.name}</strong>
        <span style="color:#666">${dev.name}</span>
        <button class="toggle-btn">${w.enabled ? 'Disable' : 'Enable'}</button>
      </div>
      <div class="vram-bar">${vramStr}</div>
      <div>Utilization: ${devStats.util_pct.toFixed(1)}%</div>
      ${w.current_job ? `<div class="job-name">Job: ${w.current_job}</div>` : ''}
      <div class="gpu-chart" id="gpu-chart-${w.name}"></div>
    `;
    container.appendChild(card);
    // Toggle button
    card.querySelector('.toggle-btn').addEventListener('click', async () => {
      const action = w.enabled ? 'disable' : 'enable';
      await post(`/api/workers/${encodeURIComponent(w.name)}/${action}`);
      pollWorkers();
    });
    // GPU time-series chart
    const history = w.history || [];
    if (history.length > 0) {
      const times = history.map(h => new Date(h.t * 1000));
      const vramSeries = history.map(h => (h.devices && h.devices[0]) ? h.devices[0].used_mb : 0);
      const utilSeries = history.map(h => (h.devices && h.devices[0]) ? h.devices[0].util_pct : 0);
      Plotly.newPlot(`gpu-chart-${w.name}`, [
        { x: times, y: vramSeries, name: 'VRAM (MB)', yaxis: 'y' },
        { x: times, y: utilSeries, name: 'Util (%)', yaxis: 'y2' },
      ], {
        height: 200, margin: { t: 10, b: 30, l: 50, r: 50 },
        yaxis: { title: 'MB' }, yaxis2: { title: '%', overlaying: 'y', side: 'right' },
        legend: { orientation: 'h', y: -0.2 },
      }, { responsive: true });
    }
  }
}

async function pollWorkers() {
  try { renderGPUCards(await api('/api/workers')); } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function init() {
  // Build primary checkboxes
  makeCheckboxes('primary-block-sizes', BLOCK_SIZES);
  makeCheckboxes('primary-K', K_VALUES, fmtK);
  // Build FP dtype checkboxes
  const fpDiv = document.getElementById('fp-dtypes');
  fpDiv.innerHTML = '';
  for (const fp of FP_DTYPES) {
    const lbl = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.value = fp; cb.checked = true;
    lbl.appendChild(cb); lbl.appendChild(document.createTextNode(' ' + fp));
    fpDiv.appendChild(lbl);
  }
  // Scale dtype toggle
  document.querySelectorAll('input[name=scale-dtype-type]').forEach(r => r.addEventListener('change', toggleScaleDtype));
  // BPW slider text
  const bpwMin = document.getElementById('bpw-min');
  const bpwMax = document.getElementById('bpw-max');
  function updateBpwText() {
    const lo = Math.min(Number(bpwMin.value), Number(bpwMax.value));
    const hi = Math.max(Number(bpwMin.value), Number(bpwMax.value));
    document.getElementById('bpw-text').textContent = `[${lo.toFixed(1)}, ${hi.toFixed(1)}]`;
  }
  bpwMin.addEventListener('input', updateBpwText);
  bpwMax.addEventListener('input', updateBpwText);
  // Bits slider text
  const bitsMin = document.getElementById('bits-min');
  const bitsMax = document.getElementById('bits-max');
  function updateBitsText() {
    const lo = Math.min(Number(bitsMin.value), Number(bitsMax.value));
    const hi = Math.max(Number(bitsMin.value), Number(bitsMax.value));
    document.getElementById('bits-text').textContent = `[${lo}, ${hi}]`;
  }
  bitsMin.addEventListener('input', updateBitsText);
  bitsMax.addEventListener('input', updateBitsText);
  // Add residual button
  document.getElementById('add-residual').addEventListener('click', addResidualSlot);
  // Apply button
  document.getElementById('apply-range').addEventListener('click', applyRange);
  // Control buttons
  document.getElementById('btn-launch').addEventListener('click', () => post('/api/control/launch').then(pollStatus));
  document.getElementById('btn-pause').addEventListener('click', () => post('/api/control/pause').then(pollStatus));
  document.getElementById('btn-resume').addEventListener('click', () => post('/api/control/resume').then(pollStatus));
  document.getElementById('btn-requeue').addEventListener('click', () => post('/api/control/requeue-failed').then(pollStatus));
  document.getElementById('btn-shutdown').addEventListener('click', () => post('/api/control/shutdown').then(pollStatus));
  // Heatmap projection dropdown
  document.getElementById('heatmap-projection').addEventListener('change', () => pollResults());
  // Load range
  loadRange();
  // Start polling
  pollStatus(); setInterval(pollStatus, 2000);
  pollResults(); setInterval(pollResults, 5000);
  pollWorkers(); setInterval(pollWorkers, 5000);
}

document.addEventListener('DOMContentLoaded', init);
