"use strict";

// Data written by src/make_web_map.py (relative to index.html at the site root)
const SUMMARY_URL = "map/clusters.json";
const TILES_URL = "map/clusters.pmtiles";
const POINTS_MAX_ZOOM = 12; // centroids are drawn below this zoom...
const POLYGONS_MIN_ZOOM = 11; // ...and polygons from this zoom, cross-fading in between
const MAX_SELECTED = 3; // a map can only carry 3 categorical colours that stay distinguishable

// Colours: categorical slots 1-3, sequential blue ramp and blue/red diverging ramp,
// each stepped separately for light and dark surfaces (validated for colour-vision deficiency).
const THEMES = {
  light: {
    slots: ["#2a78d6", "#eb6834", "#1baf7a"],
    neutral: "#898781",
    ink: "#0b0b0b",
    surface: "#fcfcfb",
    sequential: ["#b7d3f6", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    diverging: ["#911e22", "#e34948", "#f2958e", "#f0efec", "#86b6ef", "#3987e5", "#184f95"],
    basemap: "positron",
  },
  dark: {
    slots: ["#3987e5", "#d95926", "#199e70"],
    neutral: "#898781",
    ink: "#f0efec",
    surface: "#1a1a19",
    sequential: ["#104281", "#1c5cab", "#3987e5", "#86b6ef", "#b7d3f6"],
    diverging: ["#f2958e", "#e34948", "#a72528", "#383835", "#1c5cab", "#3987e5", "#86b6ef"],
    basemap: "dark",
  },
};

// Keyless basemaps: OpenFreeMap vector styles, with Esri imagery as the satellite option
const BASEMAP_STYLES = {
  positron: "https://tiles.openfreemap.org/styles/positron",
  dark: "https://tiles.openfreemap.org/styles/dark",
};
const SATELLITE_SOURCE = {
  type: "raster",
  tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
  tileSize: 256,
  maxzoom: 19,
  attribution: "Imagery © Esri, Maxar, Earthstar Geographics and the GIS User Community",
};

const DATA_ATTRIBUTION =
  'Elevation: <a href="https://www.linz.govt.nz/">LINZ</a> · Land cover: LCDB v6.0, Manaaki Whenua · ' +
  'Geology © <a href="https://doi.org/10.21420/5XTJ-5718">GNS Science</a> 2023 (CC BY 4.0)';

// Labels for the two ends of diverging scales
const POLES = {
  dz: ["Erosion", "Deposition"],
  dndvi: ["Less green", "Greener"],
};

const FEATURE_LAYERS = ["c-points", "c-points-hit", "c-polygons-fill", "c-polygons-line"];

const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const theme = () => THEMES[darkQuery.matches ? "dark" : "light"];

const params = new URLSearchParams(location.search);
const state = {
  colourBy: params.get("colour") || "cluster",
  selected: [], // [{ id, slot }] in selection order; a cluster keeps its colour while selected
  showOthers: params.get("others") !== "0",
  change: ["erosion", "deposition"].includes(params.get("change")) ? params.get("change") : "all",
  basemap: params.get("basemap") === "satellite" ? "satellite" : "map",
  sort: { key: "id", dir: 1 },
  profileView: "chart",
};

let summary;
let fieldByKey = {};
let clusterById = {};
let map;
let hoverId = null;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "style") Object.assign(node.style, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function svgIcon(path) {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "1.6");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.setAttribute("aria-hidden", "true");
  const p = document.createElementNS(ns, "path");
  p.setAttribute("d", path);
  svg.append(p);
  return svg;
}
const ZOOM_ICON = "M6.5 2.5h-4v4M9.5 2.5h4v4M6.5 13.5h-4v-4M9.5 13.5h4v-4";

const formatters = {};
function number(value, decimals) {
  formatters[decimals] ??= new Intl.NumberFormat(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
  return formatters[decimals].format(value).replace("-", "−");
}

function formatArea(m2) {
  return m2 >= 10000 ? `${number(m2 / 1e4, 1)} ha` : `${number(m2, 0)} m²`;
}

function formatValue(field, value, { signed = false } = {}) {
  if (value === null || value === undefined || Number.isNaN(value)) return "–";
  if (field.key === "area") return formatArea(value);
  let text = number(value, field.decimals);
  if (signed && value > 0) text = `+${text}`;
  if (!field.unit) return text;
  return field.unit === "°" || field.unit === "%" ? `${text}${field.unit}` : `${text} ${field.unit}`;
}

function ordinal(n) {
  const v = Math.round(n);
  const suffixes = ["th", "st", "nd", "rd"];
  const mod = v % 100;
  return `${v}${suffixes[(mod - 20) % 10] || suffixes[mod] || suffixes[0]}`;
}

function clusterName(id) {
  return `Cluster ${id}`;
}

function slotOf(id) {
  const s = state.selected.find((item) => item.id === id);
  return s ? s.slot : -1;
}

function selectedIds() {
  return state.selected.map((s) => s.id);
}

function landCoverName(code) {
  if (code === null || code === undefined) return null;
  return summary.lcdb_classes[String(code)] || `LCDB class ${code}`;
}

// ---------------------------------------------------------------------------
// Colour scales
// ---------------------------------------------------------------------------
function stopsFor(field) {
  const t = theme();
  if (field.scale === "diverging") {
    const extent = Math.max(Math.abs(field.p02 ?? 0), Math.abs(field.p98 ?? 0)) || 1;
    const half = (t.diverging.length - 1) / 2;
    return t.diverging.map((colour, i) => [-extent + (i * extent) / half, colour]);
  }
  let lo = field.p02;
  let hi = field.p98;
  if (!(hi > lo)) {
    lo = field.min;
    hi = field.max > lo ? field.max : lo + 1;
  }
  const n = t.sequential.length;
  return t.sequential.map((colour, i) => [lo + (i * (hi - lo)) / (n - 1), colour]);
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function interpolateColour(stops, value) {
  if (value <= stops[0][0]) return stops[0][1];
  for (let i = 1; i < stops.length; i++) {
    const [v1, c1] = stops[i];
    if (value <= v1) {
      const [v0, c0] = stops[i - 1];
      const f = (value - v0) / (v1 - v0);
      const a = hexToRgb(c0);
      const b = hexToRgb(c1);
      return `rgb(${a.map((x, k) => Math.round(x + (b[k] - x) * f)).join(",")})`;
    }
  }
  return stops[stops.length - 1][1];
}

// The colour a feature is drawn with (for keys in tooltips and popups)
function featureColour(props) {
  const t = theme();
  if (state.colourBy === "cluster") {
    const slot = slotOf(props.cluster);
    return slot >= 0 ? t.slots[slot] : t.neutral;
  }
  const field = fieldByKey[state.colourBy];
  const value = props[field.key];
  return typeof value === "number" ? interpolateColour(stopsFor(field), value) : t.neutral;
}

function colourExpression() {
  const t = theme();
  if (state.colourBy === "cluster") {
    if (!state.selected.length) return t.neutral;
    return ["match", ["get", "cluster"], ...state.selected.flatMap((s) => [s.id, t.slots[s.slot]]), t.neutral];
  }
  const field = fieldByKey[state.colourBy];
  return [
    "case",
    ["==", ["typeof", ["get", field.key]], "number"],
    ["interpolate", ["linear"], ["get", field.key], ...stopsFor(field).flat()],
    t.neutral,
  ];
}

// Colouring by cluster: selected clusters in focus, the rest as grey context.
// Colouring by a variable: every area at full strength (unticking "show clusters
// that aren't selected" filters to the selection instead).
function opacityExpression() {
  if (state.colourBy !== "cluster") return 0.85;
  if (!state.selected.length) return 0.6;
  return ["case", ["in", ["get", "cluster"], ["literal", selectedIds()]], 0.85, 0.35];
}

function sortKeyExpression() {
  if (!state.selected.length) return 0;
  return ["case", ["in", ["get", "cluster"], ["literal", selectedIds()]], 1, 0];
}

function filterExpression() {
  const parts = ["all"];
  if (state.change === "erosion") parts.push(["<", ["get", "dz"], 0]);
  if (state.change === "deposition") parts.push([">", ["get", "dz"], 0]);
  if (state.selected.length && !state.showOthers) {
    parts.push(["in", ["get", "cluster"], ["literal", selectedIds()]]);
  }
  return parts.length > 1 ? parts : null;
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------
let tilesUrl;
let basemapLayers = {}; // basemap layer id -> its own visibility
const basemapStyles = {};

function loadBasemapStyle(name) {
  basemapStyles[name] ??= fetch(BASEMAP_STYLES[name]).then((r) => {
    if (!r.ok) throw new Error(`basemap style: HTTP ${r.status}`);
    return r.json();
  });
  return basemapStyles[name];
}

function dataLayers() {
  const pointLayer = { source: "clusters", "source-layer": "points", maxzoom: POINTS_MAX_ZOOM };
  const polygonLayer = { source: "clusters", "source-layer": "polygons", minzoom: POLYGONS_MIN_ZOOM };
  return [
    {
      id: "c-points",
      type: "circle",
      ...pointLayer,
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 1.6, 8, 2.6, 11, 4.5],
        "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 6, 0, 9, 0.8],
      },
    },
    // Invisible, larger targets so small dots are easy to hover and tap
    { id: "c-points-hit", type: "circle", ...pointLayer, paint: { "circle-radius": 11, "circle-opacity": 0 } },
    { id: "c-polygons-fill", type: "fill", ...polygonLayer, paint: {} },
    {
      id: "c-polygons-line",
      type: "line",
      ...polygonLayer,
      minzoom: 12,
      paint: { "line-width": ["interpolate", ["linear"], ["zoom"], 12, 0.4, 16, 1.4] },
    },
    { id: "c-hover-polygon", type: "line", ...polygonLayer, filter: ["==", ["get", "id"], -1], paint: { "line-width": 2.5 } },
    {
      id: "c-hover-point",
      type: "circle",
      ...pointLayer,
      filter: ["==", ["get", "id"], -1],
      paint: { "circle-radius": 7, "circle-opacity": 0, "circle-stroke-width": 2 },
    },
  ];
}

// The basemap style with the satellite layer underneath and the data layers
// inserted below the basemap's labels, so place names stay readable.
function buildStyle(base) {
  const style = structuredClone(base);
  basemapLayers = Object.fromEntries(style.layers.map((l) => [l.id, l.layout?.visibility || "visible"]));
  style.sources.satellite = SATELLITE_SOURCE;
  style.sources.clusters = { type: "vector", url: `pmtiles://${tilesUrl}`, attribution: DATA_ATTRIBUTION };
  const satellite = state.basemap === "satellite";
  if (satellite) {
    for (const layer of style.layers) layer.layout = { ...layer.layout, visibility: "none" };
  }
  const firstLabel = style.layers.findIndex((l) => l.type === "symbol");
  style.layers.splice(firstLabel < 0 ? style.layers.length : firstLabel, 0, ...dataLayers());
  style.layers.unshift({
    id: "satellite", type: "raster", source: "satellite", layout: { visibility: satellite ? "visible" : "none" },
  });
  return style;
}

async function setBasemapTheme() {
  const style = buildStyle(await loadBasemapStyle(theme().basemap));
  map.setStyle(style);
  map.once("styledata", applyMapStyle);
}

function applyMapStyle() {
  if (!map || !map.getLayer("c-points")) return;
  const t = theme();
  const colour = colourExpression();
  const opacity = opacityExpression();
  const filter = filterExpression();
  const sortKey = sortKeyExpression();

  const satellite = state.basemap === "satellite";
  for (const [id, visibility] of Object.entries(basemapLayers)) {
    map.setLayoutProperty(id, "visibility", satellite ? "none" : visibility);
  }
  map.setLayoutProperty("satellite", "visibility", satellite ? "visible" : "none");

  map.setPaintProperty("c-points", "circle-color", colour);
  map.setPaintProperty("c-points", "circle-opacity", [
    "interpolate", ["linear"], ["zoom"], POLYGONS_MIN_ZOOM, opacity, POINTS_MAX_ZOOM, 0,
  ]);
  map.setPaintProperty("c-points", "circle-stroke-color", t.surface);
  map.setPaintProperty("c-points", "circle-stroke-opacity", [
    "interpolate", ["linear"], ["zoom"], POLYGONS_MIN_ZOOM, 0.8, POINTS_MAX_ZOOM, 0,
  ]);
  map.setPaintProperty("c-polygons-fill", "fill-color", colour);
  map.setPaintProperty("c-polygons-fill", "fill-opacity", [
    "interpolate", ["linear"], ["zoom"], POLYGONS_MIN_ZOOM, 0, POINTS_MAX_ZOOM, opacity,
  ]);
  map.setPaintProperty("c-polygons-line", "line-color", colour);
  map.setPaintProperty("c-polygons-line", "line-opacity", opacity);
  map.setPaintProperty("c-hover-polygon", "line-color", satellite ? "#ffffff" : t.ink);
  map.setPaintProperty("c-hover-point", "circle-stroke-color", satellite ? "#ffffff" : t.ink);

  map.setLayoutProperty("c-points", "circle-sort-key", sortKey);
  map.setLayoutProperty("c-polygons-fill", "fill-sort-key", sortKey);
  for (const id of FEATURE_LAYERS) map.setFilter(id, filter);
  map.setFilter("c-hover-polygon", ["==", ["get", "id"], hoverId ?? -1]);
  map.setFilter("c-hover-point", ["==", ["get", "id"], hoverId ?? -1]);
}

function fitPadding() {
  const panel = document.getElementById("panel");
  if (window.innerWidth > 720) {
    return { top: 40, bottom: 40, right: 40, left: panel.offsetWidth + 52 };
  }
  return { top: 30, left: 24, right: 24, bottom: panel.offsetHeight + 24 };
}

function zoomToCluster(id) {
  const b = clusterById[id].bounds;
  map.fitBounds([[b[0], b[1]], [b[2], b[3]]], {
    padding: fitPadding(),
    maxZoom: 14,
    duration: reducedMotion.matches ? 0 : 900,
  });
}

function setHover(id) {
  if (id === hoverId) return;
  hoverId = id;
  const filter = ["==", ["get", "id"], id ?? -1];
  map.setFilter("c-hover-polygon", filter);
  map.setFilter("c-hover-point", filter);
}

// ---------------------------------------------------------------------------
// Tooltip & popup
// ---------------------------------------------------------------------------
const tooltip = document.getElementById("tooltip");

function placeTooltip(x, y) {
  const app = document.getElementById("app").getBoundingClientRect();
  const { offsetWidth: w, offsetHeight: h } = tooltip;
  let left = x + 14;
  let top = y + 14;
  if (left + w > app.width - 8) left = x - w - 14;
  if (top + h > app.height - 8) top = y - h - 14;
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}

function showTooltip(content, x, y) {
  tooltip.replaceChildren(...content);
  tooltip.hidden = false;
  placeTooltip(x, y);
}

function hideTooltip() {
  tooltip.hidden = true;
}

function valueRow(field, value, opts) {
  return el("div", { class: "tip-row" }, el("strong", {}, formatValue(field, value, opts)), el("span", {}, field.label));
}

function featureTooltip(props) {
  const rows = [
    el("div", { class: "tip-title" },
      el("span", { class: "line-key", style: { background: featureColour(props) } }),
      clusterName(props.cluster)),
  ];
  const keys = ["dz", "area"];
  if (state.colourBy !== "cluster" && !keys.includes(state.colourBy)) keys.unshift(state.colourBy);
  for (const key of keys) {
    if (fieldByKey[key]) rows.push(valueRow(fieldByKey[key], props[key], { signed: key === "dz" }));
  }
  return rows;
}

function popupContent(props) {
  const id = props.cluster;
  const slot = slotOf(id);
  const t = theme();
  const kind = props.dz < 0 ? "Erosion" : "Deposition";
  const rows = summary.fields
    .filter((f) => f.key !== "area")
    .map((f) => el("div", { class: "popup-row" },
      el("span", {}, f.label),
      el("strong", {}, formatValue(f, props[f.key], { signed: f.key.startsWith("dz") }))));
  const landCover = landCoverName(props.lcdb);
  if (landCover) rows.push(el("div", { class: "popup-row" }, el("span", {}, "Land cover 2018"), el("strong", {}, landCover)));
  const landCover2023 = landCoverName(props.lcdb23);
  if (landCover2023) rows.push(el("div", { class: "popup-row" }, el("span", {}, "Land cover 2023"), el("strong", {}, landCover2023)));
  if (props.rock) rows.push(el("div", { class: "popup-row" }, el("span", {}, "Rock type"), el("strong", {}, props.rock)));
  if (props.unit) rows.push(el("div", { class: "popup-row" }, el("span", {}, "Geological unit"), el("strong", {}, props.unit)));
  if (props.in_river === 1) rows.push(el("div", { class: "popup-row" }, el("span", {}, "Location"), el("strong", {}, "In a river channel")));

  return el("div", {},
    el("div", { class: "popup-title" },
      el("span", { class: slot >= 0 ? "swatch" : "swatch empty", style: slot >= 0 ? { background: t.slots[slot] } : {} }),
      clusterName(id)),
    el("p", { class: "popup-sub" }, `${kind} · ${formatArea(props.area)}`),
    el("div", { class: "popup-rows" }, rows),
    el("div", { class: "popup-actions" },
      el("button", { type: "button", class: "link-button", onclick: () => toggleCluster(id) },
        slot >= 0 ? "Remove highlight" : `Highlight ${clusterName(id).toLowerCase()}`),
      el("button", { type: "button", class: "link-button", onclick: () => zoomToCluster(id) }, "Zoom to cluster")));
}

let popup;
function openPopup(lngLat, props) {
  popup?.remove();
  popup = new maplibregl.Popup({ maxWidth: "300px", focusAfterOpen: false })
    .setLngLat(lngLat)
    .setDOMContent(popupContent(props))
    .addTo(map);
}

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------
function toggleCluster(id) {
  const index = state.selected.findIndex((s) => s.id === id);
  if (index >= 0) {
    state.selected.splice(index, 1);
  } else {
    if (state.selected.length >= MAX_SELECTED) state.selected.shift(); // replace the oldest
    const used = new Set(state.selected.map((s) => s.slot));
    state.selected.push({ id, slot: [0, 1, 2].find((s) => !used.has(s)) });
  }
  popup?.remove();
  // Re-rendering the table replaces its buttons; keep keyboard focus on the same cluster
  const focused = document.activeElement?.dataset?.cluster;
  update();
  if (focused !== undefined) document.querySelector(`.cluster-toggle[data-cluster="${focused}"]`)?.focus();
}

function renderLegend() {
  const root = document.getElementById("legend");
  root.replaceChildren();
  if (state.colourBy === "cluster") {
    const t = theme();
    root.append(
      el("h2", {}, "Clusters"),
      el("p", { class: "hint" },
        `Select up to ${MAX_SELECTED} clusters below to compare them in colour; the rest are shown in grey.`),
      el("div", { class: "chips" },
        state.selected.map((s) => el("span", { class: "chip" },
          el("span", { class: "swatch", style: { background: t.slots[s.slot] } }),
          clusterName(s.id),
          el("button", {
            type: "button",
            class: "chip-remove",
            "aria-label": `Remove ${clusterName(s.id).toLowerCase()}`,
            title: "Remove",
            onclick: () => toggleCluster(s.id),
          }, "×"))),
        state.showOthers || !state.selected.length
          ? el("span", { class: "chip muted" },
            el("span", { class: "swatch", style: { background: t.neutral } }),
            state.selected.length ? "Other clusters" : "All clusters")
          : null)
    );
  } else {
    const field = fieldByKey[state.colourBy];
    const stops = stopsFor(field);
    const gradient = stops.map(([, colour], i) => `${colour} ${((i / (stops.length - 1)) * 100).toFixed(1)}%`);
    const lo = stops[0][0];
    const hi = stops[stops.length - 1][0];
    const mid = field.scale === "diverging" ? 0 : field.p50;
    const poles = field.scale === "diverging" ? POLES[field.key] || ["Negative", "Positive"] : ["Lower", "Higher"];
    root.append(
      el("h2", {}, field.label),
      el("p", { class: "hint" }, field.description),
      el("div", { class: "scale" },
        el("div", { class: "scale-poles" }, el("span", {}, poles[0]), el("span", {}, poles[1])),
        el("div", { class: "scale-bar", style: { background: `linear-gradient(to right, ${gradient.join(", ")})` } }),
        el("div", { class: "scale-ticks" },
          el("span", {}, `≤ ${formatValue(field, lo)}`),
          el("span", {}, field.scale === "diverging" ? formatValue(field, 0) : `median ${formatValue(field, mid)}`),
          el("span", {}, `≥ ${formatValue(field, hi)}`))),
      el("p", { class: "no-data" },
        el("span", { class: "swatch", style: { background: theme().neutral } }), "No value")
    );
  }
}

const CLUSTER_COLUMNS = [
  { key: "id", label: "Cluster", value: (c) => c.id },
  { key: "count", label: "Areas", value: (c) => c.count, format: (v) => number(v, 0) },
  { key: "area_ha", label: "ha", value: (c) => c.area_ha, format: (v) => number(v, 0) },
  { key: "erosion_share", label: "Erosion", value: (c) => c.erosion_share, format: (v) => `${number(v * 100, 0)}%` },
  { key: "dz", label: "Median Δz", value: (c) => c.median.dz, format: (v) => formatValue(fieldByKey.dz, v, { signed: true }) },
];

function renderClusters() {
  const root = document.getElementById("clusters");
  const t = theme();
  const { key, dir } = state.sort;
  const column = CLUSTER_COLUMNS.find((c) => c.key === key);
  const rows = [...summary.clusters].sort((a, b) => (column.value(a) - column.value(b)) * dir);

  const head = el("tr", {},
    CLUSTER_COLUMNS.map((c) => el("th", {
      scope: "col",
      "aria-sort": c.key === key ? (dir > 0 ? "ascending" : "descending") : null,
    }, el("button", {
      type: "button",
      onclick: () => {
        state.sort = { key: c.key, dir: c.key === key ? -dir : c.key === "id" ? 1 : -1 };
        renderClusters();
      },
    }, c.label, c.key === key ? (dir > 0 ? " ↑" : " ↓") : ""))),
    el("th", { scope: "col" }, el("span", { class: "visually-hidden" }, "Zoom")));

  const body = rows.map((c) => {
    const slot = slotOf(c.id);
    return el("tr", { class: slot >= 0 ? "selected" : null },
      el("td", {}, el("button", {
        type: "button",
        class: "cluster-toggle",
        "data-cluster": c.id,
        "aria-pressed": slot >= 0 ? "true" : "false",
        title: slot >= 0 ? "Remove highlight" : "Highlight this cluster",
        onclick: () => toggleCluster(c.id),
      },
      el("span", { class: slot >= 0 ? "swatch" : "swatch empty", style: slot >= 0 ? { background: t.slots[slot] } : {} }),
      clusterName(c.id))),
      CLUSTER_COLUMNS.slice(1).map((col) => el("td", {}, col.format(col.value(c)))),
      el("td", {}, el("button", {
        type: "button",
        class: "zoom-button",
        title: `Zoom to ${clusterName(c.id).toLowerCase()}`,
        "aria-label": `Zoom to ${clusterName(c.id).toLowerCase()}`,
        onclick: () => zoomToCluster(c.id),
      }, svgIcon(ZOOM_ICON))));
  });

  root.replaceChildren(
    el("div", { class: "legend-head" },
      el("label", { class: "checkbox" },
        el("input", {
          type: "checkbox",
          checked: state.showOthers,
          disabled: !state.selected.length,
          onchange: (e) => { state.showOthers = e.target.checked; update(); },
        }),
        "Show clusters that aren't selected"),
      state.selected.length
        ? el("button", { type: "button", class: "link-button", onclick: () => { state.selected = []; update(); } }, "Clear")
        : null),
    el("table", { class: "cluster-table" }, el("thead", {}, head), el("tbody", {}, body))
  );
}

function profileFields() {
  return summary.fields.filter((f) => f.scale);
}

function profileTooltip(field) {
  const t = theme();
  return [
    el("div", { class: "tip-title" }, field.label),
    ...state.selected.map((s) => {
      const c = clusterById[s.id];
      return el("div", { class: "tip-row" },
        el("strong", {}, `${formatValue(field, c.median[field.key], { signed: field.key === "dz" })}`),
        el("span", {},
          el("span", { class: "line-key", style: { background: t.slots[s.slot], marginRight: "6px", verticalAlign: "middle" } }),
          `${clusterName(s.id)} · ${ordinal(c.percentile[field.key])} percentile`));
    }),
  ];
}

function renderProfile() {
  const root = document.getElementById("profile");
  root.hidden = !state.selected.length;
  if (!state.selected.length) return;
  const t = theme();
  const isChart = state.profileView === "chart";
  const series = state.selected.map((s) => ({ ...s, cluster: clusterById[s.id] }));

  const legend = el("div", { class: "scale-poles", style: { justifyContent: "flex-start", gap: "14px", flexWrap: "wrap" } },
    series.map((s) => el("span", { style: { display: "inline-flex", alignItems: "center", gap: "6px", color: "var(--text-secondary)" } },
      el("span", { class: "swatch", style: { background: t.slots[s.slot] } }), clusterName(s.id))));

  let body;
  if (isChart) {
    body = el("div", { class: "profile-chart" },
      el("span", {}),
      el("div", { class: "profile-axis" }, el("span", {}, "Lower"), el("span", {}, "Typical"), el("span", {}, "Higher")),
      profileFields().flatMap((field) => {
        const row = el("div", {
          class: "profile-bars",
          tabindex: "0",
          "aria-label": `${field.label}: ${series.map((s) => `${clusterName(s.id)} ${ordinal(s.cluster.percentile[field.key])} percentile`).join(", ")}`,
        },
        el("span", { class: "gridline", style: { left: "25%" } }),
        el("span", { class: "gridline", style: { left: "75%" } }),
        series.map((s) => {
          const pct = s.cluster.percentile[field.key];
          const track = el("div", { class: "profile-track" });
          if (pct !== null && pct !== undefined) {
            const up = pct >= 50;
            track.append(el("span", {
              class: `profile-bar ${up ? "up" : "down"}`,
              style: { left: `${Math.min(pct, 50)}%`, width: `${Math.abs(pct - 50)}%`, background: t.slots[s.slot] },
            }));
          }
          return track;
        }));
        const show = () => {
          const r = row.getBoundingClientRect();
          showTooltip(profileTooltip(field), r.right, r.top);
        };
        row.addEventListener("pointerenter", show);
        row.addEventListener("focus", show);
        row.addEventListener("pointerleave", hideTooltip);
        row.addEventListener("blur", hideTooltip);
        return [el("span", { class: "profile-label", title: field.description }, field.label), row];
      }));
  } else {
    const cell = (s, field) => el("td", {},
      formatValue(field, s.cluster.median[field.key], { signed: field.key === "dz" }),
      el("span", { class: "pct" }, `${ordinal(s.cluster.percentile[field.key])} pct`));
    body = el("table", { class: "profile-table" },
      el("thead", {}, el("tr", {}, el("th", { scope: "col" }, "Median"),
        series.map((s) => el("th", { scope: "col", title: clusterName(s.id) },
          el("span", { class: "series" },
            el("span", { class: "swatch", style: { background: t.slots[s.slot] } }),
            el("span", { class: "visually-hidden" }, "Cluster "), String(s.id)))))),
      el("tbody", {},
        el("tr", {}, el("td", {}, "Areas"), series.map((s) => el("td", {}, number(s.cluster.count, 0)))),
        el("tr", {}, el("td", {}, "Total area"), series.map((s) => el("td", {}, `${number(s.cluster.area_ha, 0)} ha`))),
        el("tr", {}, el("td", {}, "Erosion share"), series.map((s) => el("td", {}, `${number(s.cluster.erosion_share * 100, 0)}%`))),
        profileFields().map((field) => el("tr", {}, el("td", {}, field.label), series.map((s) => cell(s, field)))),
        summary.clusters.some((c) => c.rock_types)
          ? el("tr", {}, el("td", {}, "Main rock types"),
            series.map((s) => el("td", { class: "land-cover" },
              (s.cluster.rock_types || []).slice(0, 2).map(([name, share]) =>
                el("div", {}, `${name} (${number(share * 100, 0)}%)`)))))
          : null,
        el("tr", {}, el("td", {}, "Main land cover"),
          series.map((s) => el("td", { class: "land-cover" },
            (s.cluster.land_cover || []).length
              ? s.cluster.land_cover.slice(0, 2).map(([code, share]) =>
                el("div", {}, `${landCoverName(code)} (${number(share * 100, 0)}%)`))
              : "–")))));
  }

  root.replaceChildren(
    el("div", { class: "legend-head" },
      el("h2", {}, "Cluster profiles"),
      el("button", {
        type: "button",
        class: "link-button",
        onclick: () => { state.profileView = isChart ? "table" : "chart"; renderProfile(); },
      }, isChart ? "Show as table" : "Show as chart")),
    el("p", { class: "hint" },
      isChart
        ? "Where each cluster's median sits among all areas: bars to the right mean higher than typical."
        : "Median of each cluster, and the percentile it sits at among all areas."),
    legend,
    body
  );
}

function syncControls() {
  document.getElementById("colour-by").value = state.colourBy;
  for (const group of document.querySelectorAll(".segmented")) {
    const value = state[group.dataset.setting];
    for (const button of group.querySelectorAll("button")) {
      button.setAttribute("aria-checked", button.dataset.value === value ? "true" : "false");
    }
  }
}

function syncUrl() {
  const p = new URLSearchParams();
  if (state.colourBy !== "cluster") p.set("colour", state.colourBy);
  p.set("clusters", selectedIds().join(","));
  if (!state.showOthers) p.set("others", "0");
  if (state.change !== "all") p.set("change", state.change);
  if (state.basemap !== "map") p.set("basemap", state.basemap);
  history.replaceState(history.state, "", `${location.pathname}?${p.toString()}${location.hash}`);
}

function update() {
  applyMapStyle();
  renderLegend();
  renderClusters();
  renderProfile();
  syncControls();
  syncUrl();
}

function setupControls() {
  const select = document.getElementById("colour-by");
  select.append(
    el("option", { value: "cluster" }, "Cluster"),
    ...summary.fields.filter((f) => f.scale).map((f) => el("option", { value: f.key }, f.label))
  );
  select.addEventListener("change", () => { state.colourBy = select.value; update(); });

  for (const group of document.querySelectorAll(".segmented")) {
    group.addEventListener("click", (e) => {
      const button = e.target.closest("button");
      if (!button) return;
      state[group.dataset.setting] = button.dataset.value;
      update();
    });
  }

  const panel = document.getElementById("panel");
  const toggle = document.getElementById("panel-toggle");
  toggle.addEventListener("click", () => {
    const collapsed = panel.classList.toggle("collapsed");
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.title = collapsed ? "Show panel" : "Hide panel";
  });
  // Keep the map's bottom controls above the panel when it is a bottom sheet
  new ResizeObserver(() => {
    document.documentElement.style.setProperty("--sheet-height", `${panel.offsetHeight}px`);
  }).observe(panel);
}

function initialSelection() {
  if (params.has("clusters")) {
    const ids = params.get("clusters").split(",").filter((v) => v.trim() !== "").map(Number)
      .filter((id) => id in clusterById);
    return [...new Set(ids)].slice(0, MAX_SELECTED);
  }
  // By default highlight the three largest clusters
  return [...summary.clusters].sort((a, b) => b.count - a.count).slice(0, MAX_SELECTED).map((c) => c.id);
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
async function main() {
  const loading = document.getElementById("loading");
  let baseStyle;
  try {
    const response = await fetch(SUMMARY_URL, { cache: "no-cache" });
    if (!response.ok) throw new Error(`${SUMMARY_URL}: HTTP ${response.status}`);
    [summary, baseStyle] = await Promise.all([response.json(), loadBasemapStyle(theme().basemap)]);
  } catch (err) {
    loading.textContent = `Could not load the map data (${err.message}).`;
    throw err;
  }
  fieldByKey = Object.fromEntries(summary.fields.map((f) => [f.key, f]));
  clusterById = Object.fromEntries(summary.clusters.map((c) => [c.id, c]));
  if (state.colourBy !== "cluster" && !fieldByKey[state.colourBy]?.scale) state.colourBy = "cluster";
  state.selected = initialSelection().map((id, slot) => ({ id, slot }));

  const updated = new Date(summary.generated);
  document.getElementById("meta").textContent =
    `${number(summary.count, 0)} areas · ${summary.clusters.length} clusters · updated ` +
    updated.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });

  setupControls();

  const protocol = new pmtiles.Protocol();
  maplibregl.addProtocol("pmtiles", protocol.tile);
  // The version query makes browsers fetch new tiles after each data update
  tilesUrl = new URL(`${TILES_URL}?v=${encodeURIComponent(summary.generated)}`, location.href).href;
  const b = summary.bounds;

  map = new maplibregl.Map({
    container: "map",
    style: buildStyle(baseStyle),
    bounds: [[b[0], b[1]], [b[2], b[3]]],
    fitBoundsOptions: { padding: fitPadding() },
    hash: true,
    maxZoom: 18,
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  map.addControl(new maplibregl.FullscreenControl(), "top-right");
  map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");

  map.on("load", () => {
    loading.hidden = true;
    update();
  });
  map.on("error", (e) => console.error(e.error || e));
  // Some basemap styles reference icons missing from their sprite; use a blank image
  map.on("styleimagemissing", (e) => {
    if (!map.hasImage(e.id)) map.addImage(e.id, { width: 1, height: 1, data: new Uint8Array(4) });
  });

  for (const layer of ["c-polygons-fill", "c-points-hit"]) {
    map.on("mousemove", layer, (e) => {
      const feature = e.features?.[0];
      if (!feature) return;
      map.getCanvas().style.cursor = "pointer";
      setHover(feature.properties.id);
      showTooltip(featureTooltip(feature.properties), e.point.x, e.point.y);
    });
    map.on("mouseleave", layer, () => {
      map.getCanvas().style.cursor = "";
      setHover(null);
      hideTooltip();
    });
    map.on("click", layer, (e) => {
      const feature = e.features?.[0];
      if (!feature) return;
      hideTooltip();
      openPopup(e.lngLat, feature.properties);
    });
  }

  darkQuery.addEventListener("change", () => {
    setBasemapTheme();
    update();
  });
}

main();
