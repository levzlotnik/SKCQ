// VQ sweep dashboard — API client only.
// No DOM manipulation, no Plotly. Just fetch wrappers + shared constants.

const BLOCK_SIZES = [8, 10, 12, 16, 24, 32, 64, 128];
const K_VALUES = [
    16, 32, 64, 128, 256, 512, 1024, 2048,
    4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288, 1048576,
];
const FP_DTYPES = ["fp8_e5m2", "fp8_e4m3", "fp16", "bf16", "fp32"];

function fmtK(v) {
  if (v >= 1048576) return (v / 1048576) + 'M';
  if (v >= 1024) return (v / 1024) + 'K';
  return String(v);
}

function isKmeans(r) { return r.scheme && r.scheme.startsWith('kmeans_'); }

// ---------------------------------------------------------------------------
// API client
// ---------------------------------------------------------------------------
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  return r.json();
}

async function post(path) { return api(path, { method: 'POST' }); }

async function postJSON(path, data) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
}

async function fetchStatus() { return api('/api/status'); }
async function fetchResults() { return api('/api/results'); }
async function fetchRange() { return api('/api/range'); }
async function fetchWorkers() { return api('/api/workers'); }
async function postRange(range) { return postJSON('/api/range', range); }
async function enableWorker(name) { return post(`/api/workers/${encodeURIComponent(name)}/enable`); }
async function disableWorker(name) { return post(`/api/workers/${encodeURIComponent(name)}/disable`); }
async function launchSweep() { return post('/api/control/launch'); }
async function pauseSweep() { return post('/api/control/pause'); }
async function resumeSweep() { return post('/api/control/resume'); }
async function requeueFailed() { return post('/api/control/requeue-failed'); }
async function shutdownSweep() { return post('/api/control/shutdown'); }
