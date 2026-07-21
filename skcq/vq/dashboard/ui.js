// VQ sweep dashboard — UI logic (jQuery).
//
// Pure functions that take data and mutate DOM via $ selectors.
// index.html wires events + polling and calls these.
//
// No fetch calls live here (that's api.js). No event wiring lives here
// (that's index.html's inline script).

// ---------------------------------------------------------------------------
// Dual-thumb range slider over a discrete value array
// ---------------------------------------------------------------------------

/**
 * Build a dual-thumb slider inside the element selected by `sel`.
 *
 * Args:
 *   sel        — jQuery selector string or $ element
 *   values     — discrete array of values the slider indexes into
 *   formatter  — fn(value) -> string for the label
 *   loIdx/hiIdx — initial index range (default: full span)
 *
 * Returns an object with getValues, getAllChecked, setValues, onChange.
 */
function makeDualSlider(sel, values, formatter, loIdx, hiIdx) {
    const $el = $(sel);
    loIdx = loIdx ?? 0;
    hiIdx = hiIdx ?? values.length - 1;
    const maxIdx = values.length - 1;

    $el.addClass("dual-slider").append(
        $('<div class="ds-slider-area">').append(
            $('<div class="ds-track">'),
            $('<div class="ds-fill">'),
            $('<input type="range" class="ds-low">')
                .attr({ min: 0, max: maxIdx, step: 1, value: loIdx }),
            $('<input type="range" class="ds-high">')
                .attr({ min: 0, max: maxIdx, step: 1, value: hiIdx }),
        ),
        $('<span class="ds-label">'),
    );

    const $lo = $el.find(".ds-low");
    const $hi = $el.find(".ds-high");
    const $fill = $el.find(".ds-fill");
    const $label = $el.find(".ds-label");
    const callbacks = [];

    function sortedIndices() {
        let a = Number($lo.val());
        let b = Number($hi.val());
        if (a > b) {
            [a, b] = [b, a];
        }
        return [a, b];
    }

    function update() {
        const [a, b] = sortedIndices();
        const pctA = (a / maxIdx) * 100;
        const pctB = (b / maxIdx) * 100;
        $fill.css({ left: pctA + "%", width: (pctB - pctA) + "%" });
        $label.text(`[${formatter(values[a])}, ${formatter(values[b])}]`);
        callbacks.forEach(function (cb) {
            cb(getValues());
        });
    }

    $lo.add($hi).on("input", update);
    update();

    function getValues() {
        const [a, b] = sortedIndices();
        return [values[a], values[b]];
    }

    function setValues(loVal, hiVal) {
        $lo.val(values.indexOf(loVal));
        $hi.val(values.indexOf(hiVal));
        update();
    }

    function getAllChecked() {
        const [a, b] = sortedIndices();
        return values.slice(a, b + 1);
    }

    return {
        getValues: getValues,
        getAllChecked: getAllChecked,
        setValues: setValues,
        onChange: function (cb) {
            callbacks.push(cb);
        },
    };
}

// ---------------------------------------------------------------------------
// Checkbox group (for FP dtypes)
// ---------------------------------------------------------------------------

function makeCheckboxGroup(sel, values, formatter) {
    const $el = $(sel).addClass("checkbox-group").empty();

    values.forEach(function (v) {
        const label = formatter ? formatter(v) : v;
        $el.append(
            $("<label>").append(
                $('<input type="checkbox">').val(v).prop("checked", true),
                " " + label,
            ),
        );
    });

    return {
        getChecked: function () {
            return $el
                .find("input:checked")
                .map(function (_, cb) {
                    return $(cb).val();
                })
                .get();
        },
        setChecked: function (vals) {
            $el.find("input").each(function (_, cb) {
                $(cb).prop("checked", vals.includes($(cb).val()));
            });
        },
    };
}

// ---------------------------------------------------------------------------
// Range panel state
// ---------------------------------------------------------------------------

const BPW_VALUES = [];
for (let b = 1.0; b <= 6.0 + 1e-9; b += 0.1) {
    BPW_VALUES.push(Math.round(b * 10) / 10);
}

const SCALE_BITS_VALUES = [];
for (let b = 2; b <= 16; b++) {
    SCALE_BITS_VALUES.push(b);
}

const rs = {};

function initRangePanel() {
    rs.bpw = makeDualSlider("#bpw-range", BPW_VALUES, function (v) {
        return v.toFixed(1);
    });
    rs.pbs = makeDualSlider("#primary-block-sizes", BLOCK_SIZES, String);
    rs.pK = makeDualSlider("#primary-K", K_VALUES, fmtK);
    rs.sbits = makeDualSlider("#scale-bits", SCALE_BITS_VALUES, String, 0, 6);
    rs.fp = makeCheckboxGroup("#fp-dtypes", FP_DTYPES);
    rs.resSlots = [];

    $('input[name="scale-dtype-type"]').on("change", toggleScaleDtype);
    $("#add-residual").on("click", addResidualSlot);
    $("#apply-range").on("click", applyRange);
}

function toggleScaleDtype() {
    const type = $('input[name="scale-dtype-type"]:checked').val();
    $("#scale-int-row").toggleClass("hidden", type !== "int");
    $("#scale-fp-row").toggleClass("hidden", type !== "fp");
}

function addResidualSlot() {
    const $slot = $($("#residual-template").html());
    $("#residuals").append($slot);

    const bs = makeDualSlider(
        $slot.find(".residual-block-sizes"),
        BLOCK_SIZES,
        String,
    );
    const k = makeDualSlider(
        $slot.find(".residual-K"),
        K_VALUES,
        fmtK,
    );

    $slot.find(".remove-residual").on("click", function () {
        $slot.remove();
        const idx = rs.resSlots.findIndex(function (s) {
            return s.$el.is($slot);
        });
        if (idx >= 0) {
            rs.resSlots.splice(idx, 1);
        }
    });

    rs.resSlots.push({ bs: bs, k: k, $el: $slot });
}

function buildRangeFromUI() {
    const [bpwMin, bpwMax] = rs.bpw.getValues();
    const metric = $('input[name="primary-metric"]:checked').val();
    const signSplit = $('input[name="primary-sign-split"]:checked').val() === "yes";
    const scaleType = $('input[name="scale-dtype-type"]:checked').val();

    let scaleDtypes;
    if (scaleType === "int") {
        const [lo, hi] = rs.sbits.getValues();
        scaleDtypes = [];
        for (let b = lo; b <= hi; b++) {
            scaleDtypes.push("int" + b);
        }
    } else {
        scaleDtypes = rs.fp.getChecked();
    }

    const residuals = rs.resSlots.map(function (s) {
        return {
            block_size: s.bs.getAllChecked(),
            K: s.k.getAllChecked(),
        };
    });

    return {
        projection: ["gate", "down"],
        bpw_min: bpwMin,
        bpw_max: bpwMax,
        primary: {
            block_size: rs.pbs.getAllChecked(),
            K: rs.pK.getAllChecked(),
            metric: [metric],
            sign_split: [signSplit],
            scale_dtype: scaleDtypes,
        },
        residuals: residuals,
    };
}

async function applyRange() {
    try {
        const data = await postRange(buildRangeFromUI());
        $("#apply-banner").text(
            data.applied === "now"
                ? `Applied: ${data.total} configs, ${data.in_queue} in queue`
                : "Changes will apply to next sweep",
        );
        const workers = await fetchWorkers();
        const nWorkers = Math.max(workers.length, 1);
        const hours = ((data.est_configs * 30) / nWorkers / 3600).toFixed(1);
        $("#range-estimate").text(
            `~${data.est_configs} configs | ~${hours}h est`,
        );
    } catch (e) {
        $("#apply-banner").text("Error: " + e);
    }
}

async function loadRangeIntoUI() {
    try {
        const r = await fetchRange();

        if (r.primary.metric && r.primary.metric[0]) {
            const val = r.primary.metric[0];
            $(`input[name="primary-metric"][value="${val}"]`).prop("checked", true);
        }

        if (r.primary.sign_split) {
            const val = r.primary.sign_split[0] ? "yes" : "no";
            $(`input[name="primary-sign-split"][value="${val}"]`).prop("checked", true);
        }

        if (r.primary.scale_dtype && r.primary.scale_dtype[0]) {
            const sd = r.primary.scale_dtype[0];
            const type = sd.startsWith("int") ? "int" : "fp";
            $(`input[name="scale-dtype-type"][value="${type}"]`).prop("checked", true);
        }

        toggleScaleDtype();
        rs.bpw.setValues(r.bpw_min, r.bpw_max);

        if (r.primary.block_size && r.primary.block_size.length) {
            const lo = Math.min(...r.primary.block_size);
            const hi = Math.max(...r.primary.block_size);
            rs.pbs.setValues(lo, hi);
        }

        if (r.primary.K && r.primary.K.length) {
            const lo = Math.min(...r.primary.K);
            const hi = Math.max(...r.primary.K);
            rs.pK.setValues(lo, hi);
        }

        $("#residuals").empty();
        rs.resSlots = [];

        for (const res of (r.residuals || [])) {
            addResidualSlot();
            const slot = rs.resSlots[rs.resSlots.length - 1];
            if (res.block_size && res.block_size.length) {
                slot.bs.setValues(
                    Math.min(...res.block_size),
                    Math.max(...res.block_size),
                );
            }
            if (res.K && res.K.length) {
                slot.k.setValues(
                    Math.min(...res.K),
                    Math.max(...res.K),
                );
            }
        }
    } catch (e) {
        // ignore on first load — range not yet set
    }
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function updateStatus(s) {
    const pct = s.total > 0 ? ((s.completed / s.total) * 100).toFixed(1) : 0;
    $("#status-text").text(
        `State: ${s.state} | Progress: ${s.completed}/${s.total} (${pct}%) | ` +
        `Failed: ${s.failed} | In queue: ${s.in_queue}`,
    );

    const running = s.state === "running" && !s.paused;
    const idle = s.state === "idle" || s.state === "stopped";
    const paused = s.state === "paused";

    $("#btn-launch").prop("disabled", !idle);
    $("#btn-pause").prop("disabled", !running);
    $("#btn-resume").prop("disabled", !paused);
    $("#btn-shutdown").prop("disabled", idle);
}

// ---------------------------------------------------------------------------
// Results charts
// ---------------------------------------------------------------------------

function paretoFrontier(rows) {
    const out = [];
    for (const r of rows) {
        const dominated = rows.some(function (o) {
            return (
                o !== r &&
                o.bits_per_weight <= r.bits_per_weight &&
                o.rel_fro_err <= r.rel_fro_err &&
                (o.bits_per_weight < r.bits_per_weight ||
                    o.rel_fro_err < r.rel_fro_err)
            );
        });
        if (!dominated) {
            out.push(r);
        }
    }
    out.sort(function (a, b) {
        return a.bits_per_weight - b.bits_per_weight;
    });
    return out;
}

function renderPareto(results) {
    const projections = [...new Set(results.map(function (r) {
        return r.projection;
    }))];
    const traces = [];

    for (const proj of projections) {
        const projRows = results.filter(function (r) {
            return r.projection === proj;
        });
        const km = projRows.filter(isKmeans);
        const ints = projRows.filter(function (r) {
            return !isKmeans(r);
        });

        if (km.length > 0) {
            traces.push({
                x: km.map(function (r) { return r.bits_per_weight; }),
                y: km.map(function (r) { return r.rel_fro_err; }),
                mode: "markers",
                name: `${proj} kmeans`,
                text: km.map(function (r) {
                    return `${r.scheme}<br>bpw=${r.bits_per_weight.toFixed(3)}` +
                        `<br>err=${r.rel_fro_err.toExponential(3)}`;
                }),
                marker: {
                    size: km.map(function (r) {
                        return Math.max(5, Math.min(15, r.K / 4096 + 4));
                    }),
                    color: km.map(function (r) { return r.block_size; }),
                    colorscale: "Viridis",
                    showscale: true,
                    colorbar: { title: "block_size" },
                    opacity: 0.7,
                },
                hovertemplate: "%{text}<extra></extra>",
            });
        }

        if (ints.length > 0) {
            traces.push({
                x: ints.map(function (r) { return r.bits_per_weight; }),
                y: ints.map(function (r) { return r.rel_fro_err; }),
                mode: "markers",
                name: `${proj} integer`,
                text: ints.map(function (r) { return r.scheme; }),
                marker: { symbol: "x", size: 8, color: "red", opacity: 0.6 },
                hovertemplate: "%{text}<br>bpw=%{x:.3f}<br>err=%{y:.6f}<extra></extra>",
            });
        }

        const pf = paretoFrontier(projRows);
        traces.push({
            x: pf.map(function (r) { return r.bits_per_weight; }),
            y: pf.map(function (r) { return r.rel_fro_err; }),
            mode: "lines",
            name: `${proj} Pareto`,
            line: { width: 2, dash: "dash" },
            hoverinfo: "skip",
        });
    }

    Plotly.newPlot("pareto-chart", traces, {
        title: "Pareto frontier: bpw vs error",
        xaxis: { title: "bits per weight" },
        yaxis: { title: "rel Frobenius error" },
        hovermode: "closest",
        height: 420,
    }, { responsive: true });
}

function renderHeatmap(results, projection) {
    const km = results.filter(function (r) {
        return isKmeans(r) && r.n_codebooks === 1 && r.projection === projection;
    });

    if (km.length === 0) {
        $("#heatmap-chart").html(
            `<p>No single-codebook results for ${projection}</p>`,
        );
        return;
    }

    const bsVals = [...new Set(km.map(function (r) {
        return r.block_size;
    }))].sort(function (a, b) { return a - b; });

    const kVals = [...new Set(km.map(function (r) {
        return r.K;
    }))].sort(function (a, b) { return a - b; });

    const z = bsVals.map(function (bs) {
        return kVals.map(function (K) {
            const r = km.find(function (r) {
                return r.block_size === bs && r.K === K;
            });
            return r ? r.rel_fro_err : null;
        });
    });

    const txt = bsVals.map(function (bs) {
        return kVals.map(function (K) {
            const r = km.find(function (r) {
                return r.block_size === bs && r.K === K;
            });
            return r ? r.bits_per_weight.toFixed(2) : "";
        });
    });

    Plotly.newPlot("heatmap-chart", [{
        z: z,
        x: kVals.map(function (K) { return Math.log2(K); }),
        y: bsVals.map(String),
        type: "heatmap",
        colorscale: "Viridis_r",
        colorbar: { title: "error" },
        text: txt,
        texttemplate: "%{text}",
        hovertemplate: "bs=%{y} K=%{x}<br>err=%{z:.6f}<br>bpw=%{text}<extra></extra>",
        customdata: kVals.map(fmtK),
    }], {
        title: `${projection}: block_size × K → error`,
        xaxis: {
            title: "K",
            tickvals: kVals.map(function (K) { return Math.log2(K); }),
            ticktext: kVals.map(fmtK),
        },
        yaxis: { title: "block_size" },
        height: 420,
    }, { responsive: true });
}

function renderBestTable(results) {
    const buckets = [];
    for (let b = 1.0; b < 6.0; b += 0.25) {
        buckets.push([b, b + 0.25]);
    }
    const projections = [...new Set(results.map(function (r) {
        return r.projection;
    }))].sort();

    let html = "<thead><tr><th>bucket</th>";
    for (const p of projections) {
        html += `<th>${p} km err</th><th>${p} km config</th>`;
        html += `<th>${p} int err</th><th>${p} int config</th>`;
        html += `<th>winner</th>`;
    }
    html += "</tr></thead><tbody>";

    for (const [lo, hi] of buckets) {
        html += `<tr><td>[${lo.toFixed(2)}-${hi.toFixed(2)})</td>`;
        for (const p of projections) {
            const rows = results.filter(function (r) {
                return r.projection === p &&
                    lo <= r.bits_per_weight &&
                    r.bits_per_weight < hi;
            });
            const km = rows.filter(isKmeans);
            const ints = rows.filter(function (r) { return !isKmeans(r); });

            let bkCfg = null, biCfg = null;
            if (km.length > 0) {
                bkCfg = km.reduce(function (a, b) {
                    return a.rel_fro_err < b.rel_fro_err ? a : b;
                });
            }
            if (ints.length > 0) {
                biCfg = ints.reduce(function (a, b) {
                    return a.rel_fro_err < b.rel_fro_err ? a : b;
                });
            }

            const bk = bkCfg ? bkCfg.rel_fro_err : null;
            const bi = biCfg ? biCfg.rel_fro_err : null;

            html += `<td>${bk !== null ? bk.toFixed(6) : "-"}</td>`;
            html += `<td>${bkCfg
                ? `${bkCfg.scheme}<br>bpw=${bkCfg.bits_per_weight.toFixed(3)}`
                : "-"}</td>`;
            html += `<td>${bi !== null ? bi.toFixed(6) : "-"}</td>`;
            html += `<td>${biCfg
                ? `${biCfg.scheme}<br>bpw=${biCfg.bits_per_weight.toFixed(3)}`
                : "-"}</td>`;

            if (bk !== null && bi !== null) {
                html += bk < bi
                    ? '<td class="winner-kmeans">km</td>'
                    : '<td class="winner-int">int</td>';
            } else if (bk !== null) {
                html += '<td class="winner-kmeans">km</td>';
            } else if (bi !== null) {
                html += '<td class="winner-int">int</td>';
            } else {
                html += "<td>-</td>";
            }
        }
        html += "</tr>";
    }
    html += "</tbody>";

    $("#best-table").html(html);
}

function renderResults(results) {
    if (results.length === 0) {
        return;
    }

    renderPareto(results);

    const $sel = $("#heatmap-projection");
    const projections = [...new Set(results.map(function (r) {
        return r.projection;
    }))];

    if ($sel.children().length === 0) {
        projections.forEach(function (p) {
            $sel.append($("<option>").val(p).text(p));
        });
    }

    const proj = $sel.val() || projections[0];
    renderHeatmap(results, proj);
    renderBestTable(results);
}

// ---------------------------------------------------------------------------
// GPU cards
// ---------------------------------------------------------------------------

function renderGPUCards(workers) {
    const $container = $("#gpu-cards-container").empty();

    for (const w of workers) {
        const dev = (w.devices && w.devices.length > 0) ? w.devices[0] : null;
        const hb = w.last_heartbeat;
        const ds = (hb && hb.devices && hb.devices.length > 0)
            ? hb.devices[0]
            : null;

        const devName = dev ? dev.name : "—";
        const totalVram = dev ? dev.total_vram_mb : 0;
        const usedVram = ds ? (ds.used_mb || 0) : 0;
        const utilPct = ds ? (ds.util_pct || 0) : 0;

        const vramPct = totalVram > 0 ? usedVram / totalVram : 0;
        const filled = Math.round(vramPct * 20);
        const bar = "\u2588".repeat(filled) + "\u2591".repeat(20 - filled);

        const vramStr = dev
            ? `${bar} ${usedVram} / ${totalVram} MB`
            : (w.enabled ? "Connecting..." : "Disabled");

        // 3 states: gray (disabled), orange (enabled but not connected), green (connected)
        let dotClass = "";
        const toggleLabel = w.enabled ? "Disable" : "Enable";
        if (w.connected) {
            dotClass = "connected";      // green
        } else if (w.enabled) {
            dotClass = "connecting";     // orange
        }  // else gray (no class)

        const $card = $('<div class="gpu-card">');

        const $header = $('<div class="gpu-card-header">').append(
            $(`<span class="status-dot ${dotClass}">`),
            $("<strong>").text(w.name),
            $('<span style="color:#666">').text(devName),
        );

        const $toggleBtn = $(`<button class="toggle-btn">${toggleLabel}</button>`);
        $toggleBtn.on("click", function () {
            if (w.enabled) {
                disableWorker(w.name);
            } else {
                enableWorker(w.name);
            }
        });
        $header.append($toggleBtn);

        $card.append($header);
        $card.append($('<div class="vram-bar">').text(vramStr));
        $card.append($("<div>").text(`Utilization: ${utilPct.toFixed(1)}%`));

        if (w.current_job) {
            $card.append(
                $('<div class="job-name">').text(`Job: ${w.current_job}`),
            );
        }

        $card.append($(`<div class="gpu-chart" id="gpu-chart-${w.name}">`));
        $container.append($card);

        renderGPUChart(w, dev);
    }
}

function renderGPUChart(w, dev) {
    const history = w.history || [];
    if (history.length === 0 || !dev) {
        return;
    }

    const chartId = `gpu-chart-${w.name}`;
    const times = history.map(function (h) {
        return new Date(h.t * 1000);
    });

    const vramSeries = history.map(function (h) {
        return (h.devices && h.devices[0]) ? (h.devices[0].used_mb || 0) : 0;
    });

    const utilSeries = history.map(function (h) {
        return (h.devices && h.devices[0]) ? (h.devices[0].util_pct || 0) : 0;
    });

    Plotly.newPlot(chartId, [
        { x: times, y: vramSeries, name: "VRAM (MB)", yaxis: "y" },
        { x: times, y: utilSeries, name: "Util (%)", yaxis: "y2" },
    ], {
        height: 200,
        margin: { t: 10, b: 30, l: 50, r: 50 },
        yaxis: { title: "MB" },
        yaxis2: { title: "%", overlaying: "y", side: "right" },
        legend: { orientation: "h", y: -0.2 },
    }, { responsive: true });
}
