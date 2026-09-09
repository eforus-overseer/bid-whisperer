/* Interactive charts for the bid-whisperer report, rendered with Plotly.
   One shared theme keeps every figure in the editorial palette. Data comes from
   assets/site_data.json (aggregates only, computed by scripts/build_site_data.py). */

const INK = "#12110f";
const MUTED = "#52514e";
const FAINT = "#8a8880";
const PAPER = "#fcfcfb";
const GRID = "rgba(18,17,15,0.08)";
const ACCENT = "#2a78d6";
const ORANGE = "#eb6834";
const GREEN = "#1baf7a";
const GOLD = "#eda100";
const CAT = [ACCENT, ORANGE, GREEN, GOLD, "#e87ba4", "#4a3aa7"];

const FONT = "Geist, system-ui, sans-serif";
const MONO = "'Geist Mono', ui-monospace, monospace";

const BASE_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: FONT, size: 13, color: MUTED },
  margin: { l: 60, r: 24, t: 16, b: 44 },
  hoverlabel: { font: { family: MONO, size: 12 }, bgcolor: INK, bordercolor: INK },
  xaxis: { gridcolor: GRID, zeroline: false, linecolor: GRID, tickfont: { size: 12 },
    title: { font: { size: 12, color: FAINT } } },
  yaxis: { gridcolor: GRID, zeroline: false, linecolor: GRID, tickfont: { size: 12 },
    title: { font: { size: 12, color: FAINT } } },
  legend: { font: { size: 12 }, orientation: "h", y: 1.12, x: 0 },
  colorway: CAT,
};

const CONFIG = { displayModeBar: false, responsive: true, scrollZoom: false };

function layout(over) {
  return deepMerge(structuredClone(BASE_LAYOUT), over || {});
}
function deepMerge(a, b) {
  for (const k in b) {
    if (b[k] && typeof b[k] === "object" && !Array.isArray(b[k])) {
      a[k] = deepMerge(a[k] || {}, b[k]);
    } else a[k] = b[k];
  }
  return a;
}
function pct(x, d = 1) { return (x * 100).toFixed(d) + "%"; }

const draw = {};

draw.funnel = (d, el) => {
  const f = d.funnel;
  Plotly.newPlot(el, [{
    type: "bar", orientation: "h",
    y: f.stages.slice().reverse(),
    x: f.counts.slice().reverse(),
    marker: { color: [ACCENT, GREEN, GOLD, ORANGE].slice(0, f.stages.length).reverse() },
    text: f.counts.slice().reverse().map((c, i) =>
      "  " + c.toLocaleString() + " (" + pct(f.pct_req.slice().reverse()[i]) + ")"),
    textposition: "outside", textfont: { family: MONO, size: 12, color: MUTED },
    hovertemplate: "%{y}: %{x:,} impressions<extra></extra>",
    cliponaxis: false,
  }], layout({ margin: { l: 158, r: 90, t: 8, b: 30 },
    xaxis: { range: [0, Math.max(...f.counts) * 1.22], showgrid: false, showticklabels: false },
    yaxis: { showgrid: false, tickfont: { size: 13 } } }), CONFIG);
};

draw.auction = (d, el) => {
  const a = d.auction;
  const types = a.map(x => x.type.replace("_", " "));
  Plotly.newPlot(el, [{
    type: "bar", x: types, y: a.map(x => x.win_rate), marker: { color: ACCENT },
    hovertemplate: "%{x}<br>win rate %{y:.1%}<extra></extra>",
    text: a.map(x => pct(x.win_rate, 0)), textposition: "outside",
    textfont: { family: MONO, color: MUTED },
  }], layout({
    margin: { l: 56, r: 24, t: 8, b: 60 }, showlegend: false,
    yaxis: { title: { text: "win rate" }, tickformat: ".0%", range: [0, 1.05] },
    xaxis: { tickfont: { family: MONO } },
    annotations: a.map((x, i) => ({
      x: types[i], y: x.win_rate, yshift: 34, showarrow: false,
      text: "median bid " + x.median_bid + " CPM", font: { size: 11, color: FAINT } })),
  }), CONFIG);
};

draw.price = (d, el) => {
  const traces = [["bid", ACCENT, "our bid"], ["feedback_bid", ORANGE, "clearing price"]].map(([k, c, nm]) => {
    const p = d.price[k];
    const centers = p.edges.slice(0, -1).map((e, i) => (e + p.edges[i + 1]) / 2);
    return { type: "bar", name: nm, x: centers, y: p.counts, marker: { color: c, opacity: 0.72 },
      hovertemplate: "10^%{x:.2f} CPM<br>%{y:,} auctions<extra>" + nm + "</extra>" };
  });
  Plotly.newPlot(el, traces, layout({
    barmode: "overlay",
    xaxis: { title: { text: "bid / clearing price (log10 CPM)" } },
    yaxis: { title: { text: "auctions" } } }), CONFIG);
};

draw.iv = (d, el) => {
  const colors = { strong: GREEN, medium: ACCENT, weak: GOLD, useless: FAINT };
  const rows = d.iv.slice().reverse();
  Plotly.newPlot(el, [{
    type: "bar", orientation: "h",
    y: rows.map(x => x.feature), x: rows.map(x => x.iv),
    marker: { color: rows.map(x => colors[x.strength] || MUTED) },
    text: rows.map(x => "  " + x.iv.toFixed(3)), textposition: "outside",
    textfont: { family: MONO, size: 11, color: MUTED },
    hovertemplate: "%{y}<br>IV %{x:.4f} (%{customdata})<extra></extra>",
    customdata: rows.map(x => x.strength), cliponaxis: false,
  }], layout({ margin: { l: 140, r: 50, t: 8, b: 40 },
    xaxis: { title: { text: "information value" }, range: [0, Math.max(...rows.map(x => x.iv)) * 1.25] },
    shapes: [0.02, 0.1].map(v => ({ type: "line", x0: v, x1: v, y0: -0.5, y1: rows.length - 0.5,
      line: { color: FAINT, width: 1, dash: "dot" } })) }), CONFIG);
};

draw.ctrTrends = (d, el) => {
  const mk = (arr, name, color) => ({
    type: "scatter", mode: "lines+markers", name,
    x: arr.map(p => p.bin), y: arr.map(p => p.ctr),
    line: { color, width: 2.5 }, marker: { size: 8, color },
    customdata: arr.map(p => p.n),
    hovertemplate: name + " %{x}<br>CTR %{y:.1%} (n=%{customdata:,})<extra></extra>",
  });
  Plotly.newPlot(el, [
    mk(d.ctr_session, "session depth", ACCENT),
    mk(d.ctr_view, "viewability", GREEN),
  ], layout({
    xaxis: { title: { text: "feature bucket (ordered)" }, type: "category" },
    yaxis: { title: { text: "CTR" }, tickformat: ".0%" },
    shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: d.meta.overall_ctr,
      y1: d.meta.overall_ctr, line: { color: FAINT, width: 1, dash: "dash" } }],
  }), CONFIG);
};

draw.bidCurves = (d, el) => {
  const t = d.bid_curves.targets.map(x => x * 100);
  const traces = d.bid_curves.pairs.map((p, i) => ({
    type: "scatter", mode: "lines+markers",
    name: p.label + " (n=" + p.n + ")", x: t, y: p.bids,
    line: { color: CAT[i], width: 2.5 }, marker: { size: 6, color: CAT[i] },
    hovertemplate: p.label + "<br>win %{x:.0f}% needs %{y:.2f} CPM<extra></extra>",
  }));
  Plotly.newPlot(el, traces, layout({
    xaxis: { title: { text: "target win rate (%)" } },
    yaxis: { title: { text: "recommended bid (CPM)" } } }), CONFIG);
};

draw.calibration = (d, el) => {
  const c = d.calibration;
  Plotly.newPlot(el, [
    { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], line: { color: FAINT, dash: "dash", width: 1 },
      hoverinfo: "skip", showlegend: false },
    { type: "scatter", mode: "markers+text", x: c.map(p => p.target), y: c.map(p => p.median_realised),
      marker: { size: 15, color: ACCENT, line: { color: PAPER, width: 2 } },
      text: c.map(p => pct(p.median_realised, 0)), textposition: "top center",
      textfont: { family: MONO, size: 11, color: MUTED },
      customdata: c.map(p => p.n_pairs),
      hovertemplate: "target %{x:.0%}<br>realised %{y:.1%} (%{customdata} pairs)<extra></extra>",
      showlegend: false },
  ], layout({
    xaxis: { title: { text: "target win rate" }, tickformat: ".0%", range: [0, 1] },
    yaxis: { title: { text: "median realised win rate" }, tickformat: ".0%", range: [0, 1] },
    margin: { l: 60, r: 24, t: 8, b: 44 } }), CONFIG);
};

draw.hourly = (d, el) => {
  const h = d.hourly.map(x => x.hour);
  const mk = (key, name, color, axis, fmt) => ({
    type: "scatter", mode: "lines+markers", name, x: h, y: d.hourly.map(x => x[key]),
    line: { color, width: 2.4 }, marker: { size: 5, color }, yaxis: axis,
    hovertemplate: name + " at %{x}:00 UTC = " + fmt + "<extra></extra>",
  });
  Plotly.newPlot(el, [
    mk("requests", "requests", FAINT, "y", "%{y:,}"),
    mk("win_rate", "win rate", ACCENT, "y2", "%{y:.0%}"),
    mk("ctr", "CTR", ORANGE, "y2", "%{y:.1%}"),
  ], layout({
    margin: { l: 64, r: 60, t: 30, b: 44 },
    xaxis: { title: { text: "hour of day (UTC)" }, dtick: 3 },
    yaxis: { title: { text: "requests" }, side: "left" },
    yaxis2: { title: { text: "rate" }, overlaying: "y", side: "right", tickformat: ".0%", showgrid: false },
    shapes: [{ type: "rect", x0: 7.5, x1: 10.5, y0: 0, y1: 1, yref: "paper", xref: "x",
      fillcolor: "rgba(18,17,15,0.05)", line: { width: 0 } }],
  }), CONFIG);
};

draw.lorenz = (d, el) => {
  const names = { domain: "domains", url: "pages", ad_slot: "ad slots" };
  const traces = [
    { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], line: { color: FAINT, dash: "dash", width: 1 },
      name: "equal shares", hoverinfo: "skip" },
  ];
  Object.keys(d.lorenz).forEach((k, i) => {
    const L = d.lorenz[k];
    traces.push({ type: "scatter", mode: "lines", x: L.x, y: L.y,
      name: names[k] + " (Gini " + L.gini.toFixed(2) + ")", line: { color: CAT[i], width: 2.5 },
      hovertemplate: names[k] + "<br>smallest %{x:.0%} carry %{y:.1%}<extra></extra>" });
  });
  Plotly.newPlot(el, traces, layout({
    xaxis: { title: { text: "share of publishers, ranked small to large" }, tickformat: ".0%", range: [0, 1] },
    yaxis: { title: { text: "share of impressions" }, tickformat: ".0%", range: [0, 1] } }), CONFIG);
};

draw.os = (d, el) => {
  const rows = d.os.slice().sort((a, b) => a.ctr - b.ctr);
  Plotly.newPlot(el, [{
    type: "bar", orientation: "h", y: rows.map(x => x.name), x: rows.map(x => x.ctr),
    marker: { color: rows.map(x => x.served < 300 ? FAINT : GREEN) },
    text: rows.map(x => "  " + pct(x.ctr)), textposition: "outside",
    textfont: { family: MONO, size: 11, color: MUTED },
    customdata: rows.map(x => x.served),
    hovertemplate: "%{y}<br>CTR %{x:.1%} on %{customdata:,} served<extra></extra>", cliponaxis: false,
  }], layout({ margin: { l: 80, r: 50, t: 8, b: 40 },
    xaxis: { title: { text: "CTR on served impressions" }, tickformat: ".0%",
      range: [0, Math.max(...rows.map(x => x.ctr)) * 1.25] },
    shapes: [{ type: "line", x0: d.meta.overall_ctr, x1: d.meta.overall_ctr, y0: -0.5, y1: rows.length - 0.5,
      line: { color: INK, width: 1, dash: "dash" } }] }), CONFIG);
};

draw.cookie = (d, el) => {
  const c = d.cookie_age;
  Plotly.newPlot(el, [{
    type: "scatter", mode: "lines+markers", x: c.map(x => x.bin), y: c.map(x => x.ctr),
    line: { color: ORANGE, width: 2.5 }, marker: { size: 8, color: ORANGE },
    customdata: c.map(x => x.served),
    hovertemplate: "%{x}<br>CTR %{y:.1%} (%{customdata:,} served)<extra></extra>",
  }], layout({ margin: { l: 60, r: 20, t: 8, b: 70 },
    xaxis: { title: { text: "cookie age" }, type: "category", tickangle: -30 },
    yaxis: { title: { text: "CTR" }, tickformat: ".0%" },
    shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: d.meta.overall_ctr, y1: d.meta.overall_ctr,
      line: { color: FAINT, width: 1, dash: "dash" } }] }), CONFIG);
};

draw.archetypes = (d, el) => {
  const a = d.archetypes;
  const traces = Object.keys(a.names).map(c => {
    const pts = a.points.filter(p => p.c == c);
    return { type: "scatter", mode: "markers", name: a.names[c] + " (" + pts.length + ")",
      x: pts.map(p => p.mobile), y: pts.map(p => p.bid),
      marker: { size: pts.map(p => Math.sqrt(p.n) * 1.4 + 5), color: CAT[c], opacity: 0.66,
        line: { color: PAPER, width: 1 } },
      customdata: pts.map(p => [p.n, p.ctr]),
      hovertemplate: "mobile %{x:.0%}<br>median bid %{y:.2f} CPM<br>%{customdata[0]:,} imps, CTR %{customdata[1]:.1%}<extra></extra>" };
  });
  Plotly.newPlot(el, traces, layout({
    margin: { l: 64, r: 20, t: 30, b: 50 },
    xaxis: { title: { text: "share of a domain's impressions on phones or tablets" }, tickformat: ".0%",
      range: [-0.06, 1.06] },
    yaxis: { title: { text: "median bid (CPM, log)" }, type: "log" },
    legend: { orientation: "h", y: 1.14, x: 0, font: { size: 11 } } }), CONFIG);
};

draw.states = (d, el) => {
  Plotly.newPlot(el, [{
    type: "choropleth", locationmode: "USA-states",
    locations: d.states.map(s => s.code), z: d.states.map(s => s.ctr),
    zmin: 0.06, zmax: 0.125,
    colorscale: [[0, "#eef3fb"], [0.5, "#7fb0e8"], [1, "#1b4f96"]],
    marker: { line: { color: PAPER, width: 1 } },
    colorbar: { title: { text: "CTR", font: { size: 11, color: FAINT } }, tickformat: ".0%",
      thickness: 12, len: 0.7, x: 0.98 },
    customdata: d.states.map(s => [s.served, s.impressions]),
    hovertemplate: "%{location}<br>CTR %{z:.1%}<br>%{customdata[0]:,} served of %{customdata[1]:,}<extra></extra>",
  }], layout({
    margin: { l: 0, r: 0, t: 0, b: 0 },
    geo: { scope: "usa", bgcolor: "rgba(0,0,0,0)", lakecolor: "rgba(0,0,0,0)",
      landcolor: "#f1f0eb", subunitcolor: PAPER },
  }), CONFIG);
};

function showFallback() {
  document.querySelectorAll(".chart").forEach(c => {
    const p = document.createElement("p");
    p.className = "chart-fallback";
    p.textContent = "Interactive charts load from assets/site_data.json. Serve this folder over http " +
      "(for example: python3 -m http.server, run inside docs/) rather than opening the file directly.";
    c.replaceChildren(p);
  });
}

async function boot() {
  let data;
  try {
    const res = await fetch("assets/site_data.json");
    if (!res.ok) throw new Error(res.status);
    data = await res.json();
  } catch (e) {
    showFallback();
    return;
  }
  const map = {
    "chart-funnel": draw.funnel, "chart-auction": draw.auction, "chart-price": draw.price,
    "chart-iv": draw.iv, "chart-ctr-trends": draw.ctrTrends, "chart-bid-curves": draw.bidCurves,
    "chart-calibration": draw.calibration, "chart-hourly": draw.hourly, "chart-lorenz": draw.lorenz,
    "chart-os": draw.os, "chart-cookie": draw.cookie, "chart-archetypes": draw.archetypes,
    "chart-states": draw.states,
  };
  const io = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting && !e.target.dataset.drawn) {
        e.target.dataset.drawn = "1";
        try { map[e.target.id](data, e.target); } catch (err) { console.error(e.target.id, err); }
        io.unobserve(e.target);
      }
    });
  }, { rootMargin: "200px" });
  Object.keys(map).forEach(id => {
    const el = document.getElementById(id);
    if (el) io.observe(el);
  });
}
document.addEventListener("DOMContentLoaded", boot);
