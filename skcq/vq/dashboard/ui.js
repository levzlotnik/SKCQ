// VQ sweep dashboard — UI logic: rendering + DOM building.
// No fetch calls, no polling loops. Pure functions that take data and mutate DOM.
// index.html's inline script wires events + polling and calls these.

// ---------------------------------------------------------------------------
// Dual-thumb range slider over a discrete value array
// ---------------------------------------------------------------------------
function makeDualSlider(container, values, formatter, initialLoIdx, initialHiIdx) {
  const loIdx = initialLoIdx ?? 0;
  const hiIdx = initialHiIdx ?? values.length - 1;

  container.classList.add('dual-slider');
  container.innerHTML = `
    <div class="ds-track"></div>
    <div class="ds-fill"></div>
    <input type="range" class="ds-low" min="0" max="${values.length - 1}" step="1" value="${loIdx}">
    <input type="range" class="ds-high" min="0" max="${values.length - 1}" step="1" value="${hiIdx}">
    <span class="ds-label"></span>
  `;

  const lo = container.querySelector('.ds-low');
  const hi = container.querySelector('.ds-high');
  const fill = container.querySelector('.ds-fill');
  const label = container.querySelector('.ds-label');

  const callbacks = [];
  function emit() { callbacks.forEach(cb => cb(getValues())); }

  function update() {
    let a = Number(lo.value);
    let b = Number(hi.value);
    if (a > b) { [a, b] = [b, a]; }
    const pctA = (a / (values.length - 1)) * 100;
    const pctB = (b / (values.length - 1)) * 100;
    fill.style.left = pctA + '%';
    fill.style.width = (pctB - pctA) + '%';
    label.textContent = `[${formatter(values[a])}, ${formatter(values[b])}]`;
    emit();
  }

  lo.addEventListener('input', update);
  hi.addEventListener('input', update);
  update();

  function getValues() {
    let a = Number(lo.value);
    let b = Number(hi.value);
    if (a > b) { [a, b] = [b, a]; }
    return [values[a], values[b]];
  }

  function setValues(loVal, hiVal) {
    lo.value = values.indexOf(loVal);
    hi.value = values.indexOf(hiVal);
    update();
  }

  function getAllChecked() {
    let a = Number(lo.value);
    let b = Number(hi.value);
    if (a > b) { [a, b] = [b, a]; }
    return values.slice(a, b + 1);
  }

  return {
    getValues,
    getAllChecked,
    setValues,
    onChange(cb) { callbacks.push(cb); },
    container,
  };
}

// ---------------------------------------------------------------------------
// Checkbox group (for FP dtypes)
// ---------------------------------------------------------------------------
function makeCheckboxGroup(container, values, formatter) {
  container.classList.add('checkbox-group');
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
  return {
    getChecked() {
      return Array.from(container.querySelectorAll('input[type=checkbox]:checked'))
        .map(cb => cb.value);
    },
    setChecked(values) {
      container.querySelectorAll('input[type=checkbox]').forEach(cb => {
        cb.checked = values.includes(cb.value);
      });
    },
  };
}

// ---------------------------------------------------------------------------
// Range panel: build + read + load
// ---------------------------------------------------------------------------
const rangeState = {
  primaryBlockSizes: null,
  primaryK: null,
  scaleBits: null,
  fpDtypes: null,
  residualSlots: [],
};

function initRangePanel() {
  rangeState.primaryBlockSizes = makeDualSlider(
    document.getElementById('primary-block-sizes'), BLOCK_SIZES, String, 0, BLOCK_SIZES.length - 1);
  rangeState.primaryK = makeDualSlider(
    document.getElementById('primary-K'), K_VALUES, fmtK, 0, K_VALUES.length - 1);
  rangeState.scaleBits = makeDualSlider(
    document.getElementById('scale-bits'), [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], String, 0, 6);
  rangeState.fpDtypes = makeCheckboxGroup(
    document.getElementById('fp-dtypes'), FP_DTYPES);

  document.querySelectorAll('input[name=scale-dtype-type]').forEach(r =>
    r.addEventListener('change', toggleScaleDtype));
  document.getElementById('add-residual').addEventListener('click', addResidualSlot);
  document.getElementById('apply-range').addEventListener('click', applyRange);
}

function toggleScaleDtype() {
  const type = document.querySelector('input[name=scale-dtype-type]:checked').value;
  document.getElementById('scale-int-row').classList.toggle('hidden', type !== 'int');
  document.getElementById('scale-fp-row').classList.toggle('hidden', type !== 'fp');
}

function addResidualSlot() {
  const tmpl = document.getElementById('residual-template');
  const node = tmpl.content.cloneNode(true);
  const slot = node.querySelector('.residual-slot');
  document.getElementById('residuals').appendChild(node);

  const bsSlider = makeDualSlider(
    slot.querySelector('.residual-block-sizes'), BLOCK_SIZES, String, 0, BLOCK_SIZES.length - 1);
  const kSlider = makeDualSlider(
    slot.querySelector('.residual-K'), K_VALUES, fmtK, 0, K_VALUES.length - 1);

  const entry = { bsSlider, kSlider, element: slot };
  rangeState.residualSlots.push(entry);

  slot.querySelector('.remove-residual').addEventListener('click', () => {
    slot.remove();
    const idx = rangeState.residualSlots.indexOf(entry);
    if (idx >= 0) rangeState.residualSlots.splice(idx, 1);
  });
}

function buildRangeFromUI() {
  const metric = document.querySelector('input[name=primary-metric]:checked').value;
  const signSplit = document.querySelector('input[name=primary-sign-split]:checked').value === 'yes';
  const scaleType = document.querySelector('input[name=scale-dtype-type]:checked').value;
  let scaleDtypes;
  if (scaleType === 'int') {
    const [lo, hi] = rangeState.scaleBits.getValues();
    scaleDtypes = [];
    for (let b = lo; b <= hi; b++) scaleDtypes.push('int' + b);
  } else {
    scaleDtypes = rangeState.fpDtypes.getChecked();
  }

  const range = {
    projection: ['gate', 'down'],
    bpw_min: Number(document.getElementById('bpw-min').value),
    bpw_max: Number(document.getElementById('bpw-max').value),
    primary: {
      block_size: rangeState.primaryBlockSizes.getAllChecked(),
      K: rangeState.primaryK.getAllChecked(),
      metric: [metric],
      sign_split: [signSplit],
      scale_dtype: scaleDtypes,
    },
    residuals: [],
  };

  for (const slot of rangeState.residualSlots) {
    range.residuals.push({
      block_size: slot.bsSlider.getAllChecked(),
      K: slot.kSlider.getAllChecked(),
    });
  }
  return range;
}

async function applyRange() {
  const range = buildRangeFromUI();
  try {
    const data = await postRange(range);
    const banner = document.getElementById('apply-banner');
    banner.textContent = 'Changes will apply to next sweep';
    const workers = await fetchWorkers();
    const nWorkers = Math.max(workers.length, 1);
    const hours = (data.est_configs * 30 / nWorkers / 3600).toFixed(1);
    document.getElementById('range-estimate').textContent =
      `~${data.est_configs} configs | ~${hours}h est`;
  } catch (e) {
    document.getElementById('apply-banner').textContent = 'Error: ' + e;
  }
}

async function loadRangeIntoUI() {
  try {
    const r = await fetchRange();
    document.getElementById('bpw-min').value = r.bpw_min;
    document.getElementById('bpw-max').value = r.bpw_max;
    updateBpwText();

    if (r.primary.metric?.[0]) {
      const m = document.querySelector(`input[name=primary-metric][value="${r.primary.metric[0]}"]`);
      if (m) m.checked = true;
    }
    if (r.primary.sign_split) {
      const ss = document.querySelector(`input[name=primary-sign-split][value="${r.primary.sign_split[0] ? 'yes' : 'no'}"]`);
      if (ss) ss.checked = true;
    }
    if (r.primary.scale_dtype?.[0]) {
      const sd = r.primary.scale_dtype[0];
      if (sd.startsWith('int')) {
        document.querySelector('input[name=scale-dtype-type][value="int"]').checked = true;
      } else {
        document.querySelector('input[name=scale-dtype-type][value="fp"]').checked = true;
      }
    }
    toggleScaleDtype();

    // Set primary slider ranges
    if (r.primary.block_size?.length) {
      const lo = Math.min(...r.primary.block_size);
      const hi = Math.max(...r.primary.block_size);
      rangeState.primaryBlockSizes.setValues(lo, hi);
    }
    if (r.primary.K?.length) {
      const lo = Math.min(...r.primary.K);
      const hi = Math.max(...r.primary.K);
      rangeState.primaryK.setValues(lo, hi);
    }

    // Residuals
    document.getElementById('residuals').innerHTML = '';
    rangeState.residualSlots = [];
    for (const res of (r.residuals || [])) {
      addResidualSlot();
      const slot = rangeState.residualSlots[rangeState.residualSlots.length - 1];
      if (res.block_size?.length) {
        slot.bsSlider.setValues(Math.min(...res.block_size), Math.max(...res.block_size));
      }
      if (res.K?.length) {
        slot.kSlider.setValues(Math.min(...res.K), Math.max(...res.K));
      }
    }
  } catch (e) { /* ignore on first load */ }
}

function updateBpwText() {
  const lo = Math.min(Number(document.getElementById('bpw-min').value),
                       Number(document.getElementById('bpw-max').value));
  const hi = Math.max(Number(document.getElementById('bpw-min').value),
                       Number(document.getElementById('bpw-max').value));
  document.getElementById('bpw-text').textContent = `[${lo.toFixed(1)}, ${hi.toFixed(1)}]`;
}

// ---------------------------------------------------------------------------
// Status bar
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
      if (bk !== null && bi !== null) {
        html += bk < bi ? '<td class="winner-kmeans">km</td>' : '<td class="winner-int">int</td>';
      } else {
        html += '<td>-</td>';
      }
    }
    html += '</tr>';
  }
  html += '</tbody>';
  document.getElementById('best-table').innerHTML = html;
}

function renderResults(results) {
  if (results.length === 0) return;
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

// ---------------------------------------------------------------------------
// GPU cards
// ---------------------------------------------------------------------------
function renderGPUCards(workers) {
  const container = document.getElementById('gpu-cards-container');
  container.innerHTML = '';
  for (const w of workers) {
    const card = document.createElement('div');
    card.className = 'gpu-card';
    const dev = w.devices?.[0] || { name: '?', total_vram_mb: 0 };
    const hb = w.last_heartbeat;
    const devStats = hb?.devices?.[0] || { used_mb: 0, util_pct: 0 };
    const vramPct = dev.total_vram_mb > 0 ? (devStats.used_mb / dev.total_vram_mb) : 0;
    const barLen = 20;
    const filled = Math.round(vramPct * barLen);
    const barStr = '█'.repeat(filled) + '░'.repeat(barLen - filled);
    const vramStr = `${barStr} ${devStats.used_mb || 0} / ${dev.total_vram_mb} MB`;

    card.innerHTML = `
      <div class="gpu-card-header">
        <span class="status-dot ${w.connected ? 'connected' : ''}"></span>
        <strong>${w.name}</strong>
        <span style="color:#666">${dev.name || '?'}</span>
        <button class="toggle-btn">${w.enabled ? 'Disable' : 'Enable'}</button>
      </div>
      <div class="vram-bar">${vramStr}</div>
      <div>Utilization: ${(devStats.util_pct || 0).toFixed(1)}%</div>
      ${w.current_job ? `<div class="job-name">Job: ${w.current_job}</div>` : ''}
      <div class="gpu-chart" id="gpu-chart-${w.name}"></div>
    `;
    container.appendChild(card);

    card.querySelector('.toggle-btn').addEventListener('click', async () => {
      if (w.enabled) { await disableWorker(w.name); } else { await enableWorker(w.name); }
    });

    const history = w.history || [];
    if (history.length > 0) {
      const times = history.map(h => new Date(h.t * 1000));
      const vramSeries = history.map(h => h.devices?.[0]?.used_mb || 0);
      const utilSeries = history.map(h => h.devices?.[0]?.util_pct || 0);
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
