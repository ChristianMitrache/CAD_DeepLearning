"""Generalized static HTML gallery builder.

Takes a list of record dicts (each must have an `id` key) and writes a
paginated, filterable, sortable HTML page plus a sibling .json of the
same data. No server needed.

Per-dataset wrappers only have to build the records list and decide which
columns to expose for sort/filter. Badge values render directly from the
record keys listed in `badge_keys`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HTML_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: ui-sans-serif, system-ui; margin: 0; padding: 1rem 2rem; background:#fafafa; }
  h1 { margin-top: 0; }
  .controls { position: sticky; top: 0; background: #fafafa; padding: 0.8rem 0; border-bottom: 1px solid #ddd; z-index: 10; }
  .controls label { margin-right: 1rem; font-size: 0.9rem; }
  .controls input, .controls select { padding: 3px 6px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-top: 1rem; }
  .tile { background: white; border: 1px solid #ddd; border-radius: 6px; padding: 6px; font-size: 12px; }
  .tile img { width: 100%; height: auto; border-radius: 4px; background:#f0f0f0; }
  .tile .id { font-family: monospace; font-size: 10px; word-break: break-all; color: #555; }
  .pager { margin: 1rem 0; text-align: center; }
  .pager button { padding: 4px 10px; margin: 0 3px; }
  .badge { display:inline-block; padding:1px 5px; border-radius:3px; background:#eef; margin-right:3px; font-size:10px; }
  .stats { color: #666; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="stats" id="stats"></div>
<div class="controls" id="controls">
__FILTER_CONTROLS__
  <label>sort
    <select id="sortKey">
__SORT_OPTIONS__
    </select>
  </label>
  <label>
    <select id="sortDir">
      <option value="desc">desc</option>
      <option value="asc">asc</option>
    </select>
  </label>
  <label>per page
    <select id="perPage">
      <option>100</option><option selected>500</option><option>1000</option>
    </select>
  </label>
</div>
<div class="pager" id="pagerTop"></div>
<div class="grid" id="grid"></div>
<div class="pager" id="pagerBottom"></div>

<script>
const DATA = __DATA__;
const THUMBS = "__THUMB_PREFIX__";
const BADGE_KEYS = __BADGE_KEYS__;
const FILTER_KEYS = __FILTER_KEYS__;
let page = 0;

function filtered() {
  const key = document.getElementById('sortKey').value;
  const dir = document.getElementById('sortDir').value;
  let rows = DATA.slice();
  for (const k of FILTER_KEYS) {
    const loEl = document.getElementById('min_' + k);
    const hiEl = document.getElementById('max_' + k);
    const lo = loEl && loEl.value !== '' ? +loEl.value : -Infinity;
    const hi = hiEl && hiEl.value !== '' ? +hiEl.value : Infinity;
    rows = rows.filter(d => {
      const v = d[k];
      if (v === undefined || v === null) return true;
      return v >= lo && v <= hi;
    });
  }
  rows.sort((a,b) => {
    const va = a[key], vb = b[key];
    if (va < vb) return dir === 'asc' ? -1 : 1;
    if (va > vb) return dir === 'asc' ? 1 : -1;
    return 0;
  });
  return rows;
}

function badgeHtml(d) {
  const parts = [];
  for (const k of BADGE_KEYS) {
    const v = d[k];
    if (v === undefined || v === null || v === '') continue;
    const text = typeof v === 'number' ? v.toLocaleString() : v;
    parts.push(`<span class="badge">${k} ${text}</span>`);
  }
  return parts.join(' ');
}

function render() {
  const rows = filtered();
  const per = +document.getElementById('perPage').value;
  const nPages = Math.max(1, Math.ceil(rows.length / per));
  if (page >= nPages) page = nPages - 1;
  const slice = rows.slice(page * per, (page + 1) * per);
  document.getElementById('stats').textContent =
    `${rows.length.toLocaleString()} entries match. Showing page ${page+1}/${nPages} (${slice.length}).`;
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  for (const d of slice) {
    const div = document.createElement('div');
    div.className = 'tile';
    div.innerHTML = `
      <img loading="lazy" src="${THUMBS}${d.id}.png" onerror="this.style.opacity=0.2">
      <div class="id">${d.id}</div>
      <div>${badgeHtml(d)}</div>`;
    grid.appendChild(div);
  }
  for (const el of [document.getElementById('pagerTop'), document.getElementById('pagerBottom')]) {
    el.innerHTML = `
      <button ${page===0?'disabled':''} onclick="page=0;render()">first</button>
      <button ${page===0?'disabled':''} onclick="page--;render()">prev</button>
      page ${page+1}/${nPages}
      <button ${page>=nPages-1?'disabled':''} onclick="page++;render()">next</button>
      <button ${page>=nPages-1?'disabled':''} onclick="page=${nPages-1};render()">last</button>`;
  }
}

for (const el of document.querySelectorAll('.controls input, .controls select')) {
  el.addEventListener('input', () => { page=0; render(); });
  el.addEventListener('change', () => { page=0; render(); });
}
render();
</script>
</body>
</html>
"""


def build_gallery(
    records: list[dict],
    out_html: Path,
    thumb_dir: Path,
    columns: list[dict] | None = None,
    title: str = "Gallery",
    badge_keys: list[str] | None = None,
) -> None:
    """Write a paginated/filterable HTML gallery of records.

    Args:
        records: list of dicts; each must have an `id` key (used for thumb lookup).
        out_html: output .html path. A sibling .json is written with the same data.
        thumb_dir: directory holding `{id}.png` thumbnails. Rendered path in the HTML
            is made relative to `out_html` so the output is portable.
        columns: optional list of `{key, label, sortable, filterable}` dicts. `sortable`
            keys appear in the sort dropdown; `filterable` keys get min/max inputs.
            Defaults to one entry per numeric key in the first record (sortable=True,
            filterable=True) plus `id` (sortable only).
        title: HTML title + h1.
        badge_keys: keys to render as badges on each tile. Defaults to every non-`id`
            key in the first record.
    """
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        out_html.write_text(HTML_TEMPLATE
                            .replace("__TITLE__", title)
                            .replace("__DATA__", "[]")
                            .replace("__THUMB_PREFIX__", "")
                            .replace("__BADGE_KEYS__", "[]")
                            .replace("__FILTER_KEYS__", "[]")
                            .replace("__FILTER_CONTROLS__", "")
                            .replace("__SORT_OPTIONS__", '<option value="id">id</option>'))
        return

    first = records[0]
    if "id" not in first:
        raise ValueError("records must contain an 'id' key")

    if columns is None:
        columns = [{"key": "id", "label": "id", "sortable": True, "filterable": False}]
        for k, v in first.items():
            if k == "id":
                continue
            is_num = isinstance(v, (int, float)) and not isinstance(v, bool)
            columns.append({"key": k, "label": k, "sortable": True, "filterable": is_num})

    if badge_keys is None:
        badge_keys = [k for k in first if k != "id"]

    sort_keys = [c for c in columns if c.get("sortable", True)]
    filter_keys = [c for c in columns if c.get("filterable", False)]

    sort_options = "\n".join(
        f'      <option value="{c["key"]}">{c.get("label", c["key"])}</option>' for c in sort_keys
    )
    filter_controls = "\n".join(
        f'  <label>min {c.get("label", c["key"])} '
        f'<input type="number" id="min_{c["key"]}" style="width:80px"></label>\n'
        f'  <label>max {c.get("label", c["key"])} '
        f'<input type="number" id="max_{c["key"]}" style="width:80px"></label>'
        for c in filter_keys
    )

    rel_thumbs = os.path.relpath(Path(thumb_dir), out_html.parent).replace(os.sep, "/") + "/"

    html = (HTML_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__DATA__", json.dumps(records))
            .replace("__THUMB_PREFIX__", rel_thumbs)
            .replace("__BADGE_KEYS__", json.dumps(badge_keys))
            .replace("__FILTER_KEYS__", json.dumps([c["key"] for c in filter_keys]))
            .replace("__FILTER_CONTROLS__", filter_controls)
            .replace("__SORT_OPTIONS__", sort_options))

    out_html.write_text(html)
    data_path = out_html.with_suffix(".json")
    data_path.write_text(json.dumps(records))
