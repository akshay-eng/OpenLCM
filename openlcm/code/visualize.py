"""OpenLCM LST Visualizer.

Two output modes:
  1. Terminal — rich Tree with file → symbol hierarchy + stats
  2. HTML     — deterministic template with D3.js force-directed graph;
                data injected as JSON, no AI-generated markup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


# ── Data extraction ───────────────────────────────────────────────────────────

def build_graph_data(graph, repo_id: str, max_symbols: int = 2000, max_edges: int = 5000) -> dict:
    """Query LSTGraph and return a JSON-serializable dict for visualization."""
    with graph._lock:
        conn = graph._conn

        file_rows = conn.execute(
            "SELECT file_id, file_path, language FROM lcm_lst_files WHERE repo_id=? ORDER BY file_path",
            (repo_id,),
        ).fetchall()

        # Classes and functions first, then methods; skip raw imports/variables by default
        sym_rows = conn.execute(
            """SELECT s.symbol_id, s.file_id, s.kind, s.name, s.qualified_name,
                      s.signature, s.docstring, s.line_start
               FROM lcm_lst_symbols s
               JOIN lcm_lst_files f ON s.file_id = f.file_id
               WHERE f.repo_id = ?
               ORDER BY
                 CASE s.kind WHEN 'class' THEN 0 WHEN 'function' THEN 1
                             WHEN 'method' THEN 2 ELSE 3 END,
                 s.symbol_id
               LIMIT ?""",
            (repo_id, max_symbols),
        ).fetchall()

        edge_rows = conn.execute(
            """SELECT e.edge_type, e.from_file_id, e.to_file_id,
                      e.from_symbol_id, e.to_symbol_id, e.to_name
               FROM lcm_lst_edges e
               JOIN lcm_lst_files f ON e.from_file_id = f.file_id
               WHERE f.repo_id = ?""",
            (repo_id,),
        ).fetchall()

        stats = graph.get_stats(repo_id, include_repo_meta=True)

    file_map: dict[int, tuple[str, str]] = {fid: (fp, lang) for fid, fp, lang in file_rows}

    nodes: list[dict] = []
    file_id_to_nid: dict[int, str] = {}
    sym_id_to_nid: dict[int, str] = {}
    name_to_nid: dict[str, str] = {}  # simple + qualified name → nid for to_name resolution

    for fid, fpath, lang in file_rows:
        nid = f"f{fid}"
        file_id_to_nid[fid] = nid
        nodes.append({
            "id": nid,
            "label": fpath.replace("\\", "/").split("/")[-1],
            "type": "file",
            "file": fpath,
            "lang": lang,
            "qualified_name": fpath,
            "signature": "",
            "docstring": "",
        })

    for sym_id, file_id, kind, name, qname, sig, doc, line in sym_rows:
        nid = f"s{sym_id}"
        sym_id_to_nid[sym_id] = nid
        fpath = file_map.get(file_id, ("", ""))[0]
        nodes.append({
            "id": nid,
            "label": name,
            "type": kind,
            "file": fpath,
            "line": line,
            "qualified_name": qname or name,
            "signature": (sig or "")[:200],
            "docstring": (doc or "").split("\n")[0][:150],
        })
        name_to_nid[name] = nid
        if qname:
            name_to_nid[qname] = nid

    edges: list[dict] = []
    seen: set[tuple] = set()
    for etype, from_fid, to_fid, from_sid, to_sid, to_name in edge_rows:
        src = sym_id_to_nid.get(from_sid) if from_sid else None
        src = src or file_id_to_nid.get(from_fid)

        tgt = sym_id_to_nid.get(to_sid) if to_sid else None
        tgt = tgt or file_id_to_nid.get(to_fid)
        # Most call/import edges only store to_name — resolve via name index
        if not tgt and to_name:
            tgt = name_to_nid.get(to_name)
            if not tgt and "." in to_name:
                tgt = name_to_nid.get(to_name.rsplit(".", 1)[-1])

        if not src or not tgt or src == tgt:
            continue
        key = (src, tgt, etype)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"source": src, "target": tgt, "type": etype})
        if len(edges) >= max_edges:
            break

    # Language breakdown from stats
    lang_counts: dict[str, int] = {}
    for _, fpath, lang in file_rows:
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    return {
        "repo_id": repo_id,
        "stats": {
            "files": stats.get("files", 0),
            "symbols": stats.get("symbols", 0),
            "edges": stats.get("edges", 0),
            "languages": lang_counts,
            "git": stats.get("git", {}),
        },
        "nodes": nodes,
        "edges": edges,
        "truncated": len(sym_rows) >= max_symbols or len(edges) >= max_edges,
    }


# ── HTML render ───────────────────────────────────────────────────────────────

def render_html(data: dict, output_path: str) -> str:
    """Write a self-contained HTML visualization to output_path."""
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    repo_id = data.get("repo_id", "")
    git = data.get("stats", {}).get("git", {})
    repo_label = git.get("origin_url") or repo_id

    html = _HTML_TEMPLATE.replace("__DATA_JSON__", data_json).replace(
        "__REPO_LABEL__", _esc_attr(repo_label)
    )
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def _esc_attr(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


# ── Terminal render ───────────────────────────────────────────────────────────

def render_terminal(graph, repo_id: str) -> None:
    """Print a rich tree view of the LST to stdout."""
    try:
        from rich.console import Console
        from rich.tree import Tree
        from rich.table import Table
        from rich.text import Text
        from rich.panel import Panel
        from rich import box
    except ImportError:
        _fallback_terminal(graph, repo_id)
        return

    console = Console()
    stats = graph.get_stats(repo_id, include_repo_meta=True)
    files = graph.list_files(repo_id, limit=500)

    # ── Stats header ─────────────────────────────────────────────────────
    grid = Table.grid(padding=(0, 4))
    grid.add_row(
        Text(f"{stats['files']:,}", style="bold white"),
        Text(f"{stats['symbols']:,}", style="bold white"),
        Text(f"{stats['edges']:,}", style="bold white"),
    )
    grid.add_row(
        Text("files", style="dim"),
        Text("symbols", style="dim"),
        Text("edges", style="dim"),
    )

    git = stats.get("git", {})
    git_line = ""
    if git.get("branch"):
        git_line = f"  [dim]branch:[/] [cyan]{git['branch']}[/]"
        if git.get("commit_hash"):
            git_line += f"  [dim]commit:[/] [yellow]{git['commit_hash'][:12]}[/]"

    console.print()
    console.print(Panel(
        grid,
        title=f"[bold blue]OpenLCM Code Graph[/]  [dim]{repo_id}[/]",
        subtitle=git_line or None,
        expand=False,
    ))
    console.print()

    # ── File tree ─────────────────────────────────────────────────────────
    KIND_ICON = {
        "class": "[bold magenta]C[/]",
        "function": "[bold green]f[/]",
        "method": "[bold cyan]m[/]",
        "import": "[dim yellow]i[/]",
        "variable": "[dim white]v[/]",
    }
    KIND_STYLE = {
        "class": "magenta",
        "function": "green",
        "method": "cyan",
        "import": "yellow",
        "variable": "white",
    }

    tree = Tree(f"[bold]{repo_id}[/]")

    # Group files by directory
    dir_branches: dict[str, Tree] = {}

    def _dir_branch(path: str) -> Tree:
        parts = path.replace("\\", "/").split("/")
        if len(parts) == 1:
            return tree
        dir_key = "/".join(parts[:-1])
        if dir_key not in dir_branches:
            # Create nested dirs
            parent_key = "/".join(parts[:-2]) if len(parts) > 2 else ""
            parent = dir_branches.get(parent_key, tree)
            dir_branches[dir_key] = parent.add(f"[bold blue]{parts[-2]}/[/]")
        return dir_branches[dir_key]

    for finfo in files[:100]:
        fpath = finfo["file_path"]
        lang = finfo.get("language", "")
        parent = _dir_branch(fpath)
        fname = fpath.replace("\\", "/").split("/")[-1]
        lang_tag = f"[dim] ({lang})[/]" if lang else ""
        fbranch = parent.add(f"[white]{fname}[/]{lang_tag}")

        # Add symbols for this file
        syms = graph.get_file_symbols(fpath, repo_id)
        shown = 0
        for sym in syms:
            kind = sym.get("kind", "")
            if kind in ("import", "variable"):
                continue
            icon = KIND_ICON.get(kind, "·")
            style = KIND_STYLE.get(kind, "white")
            name = sym.get("name", "")
            line = sym.get("line_start", "")
            sig = sym.get("signature", "")
            sig_str = f"[dim] {sig[:50]}[/]" if sig else ""
            line_str = f"[dim]:{line}[/]" if line else ""
            fbranch.add(f"{icon} [{style}]{name}[/]{line_str}{sig_str}")
            shown += 1
            if shown >= 12:
                remaining = len([s for s in syms if s.get("kind") not in ("import", "variable")]) - shown
                if remaining > 0:
                    fbranch.add(f"[dim]… {remaining} more[/]")
                break

    if len(files) > 100:
        tree.add(f"[dim]… {len(files) - 100} more files (use --output to see all in HTML)[/]")

    console.print(tree)
    console.print()


def _fallback_terminal(graph, repo_id: str) -> None:
    stats = graph.get_stats(repo_id)
    print(f"\nOpenLCM Code Graph — {repo_id}")
    print(f"  Files:   {stats['files']:,}")
    print(f"  Symbols: {stats['symbols']:,}")
    print(f"  Edges:   {stats['edges']:,}")
    files = graph.list_files(repo_id, limit=50)
    for f in files:
        print(f"  {f['file_path']}")
        syms = graph.get_file_symbols(f["file_path"], repo_id)
        for s in syms[:8]:
            print(f"    [{s['kind']}] {s['name']}")
    print()


# ── HTML Template ─────────────────────────────────────────────────────────────
# Data is injected as __DATA_JSON__ and __REPO_LABEL__ placeholders.
# No AI-generated markup — this is the canonical template.


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenLCM Code Graph</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0d1117;color:#e6edf3;display:flex;height:100vh;overflow:hidden}

/* ── Sidebar ── */
#sidebar{width:270px;min-width:270px;background:#161b22;border-right:1px solid #21262d;display:flex;flex-direction:column;overflow:hidden}
#sb-header{padding:14px 16px;border-bottom:1px solid #21262d}
#sb-header h1{font-size:13px;font-weight:700;color:#58a6ff;letter-spacing:.3px}
#sb-header .repo{font-size:11px;color:#8b949e;margin-top:3px;word-break:break-all;line-height:1.4}
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:10px 12px;border-bottom:1px solid #21262d}
.stat-box{background:#0d1117;border-radius:5px;padding:7px 10px}
.stat-val{font-size:17px;font-weight:700;color:#e6edf3;font-variant-numeric:tabular-nums}
.stat-lbl{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-top:1px}
.section{padding:10px 14px;border-bottom:1px solid #21262d}
.section-title{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:7px}

/* ── Legend ── */
.leg-item{display:flex;align-items:center;gap:7px;margin-bottom:5px;cursor:pointer;user-select:none;padding:2px 0}
.leg-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;transition:opacity .15s}
.leg-item[data-shape="rect"] .leg-dot{border-radius:2px}
.leg-item span{font-size:12px;color:#c9d1d9;transition:color .15s}
.leg-item.off span{color:#484f58}
.leg-item.off .leg-dot{opacity:.2}
.edge-leg{display:flex;align-items:center;gap:7px;margin-bottom:5px;cursor:pointer;user-select:none}
.edge-line{width:24px;height:2px;flex-shrink:0;border-radius:1px}
.edge-item.off span{color:#484f58}
.edge-item.off .edge-line{opacity:.15}

/* ── Search + list ── */
#search{margin:8px 12px 0;padding:7px 10px;background:#0d1117;border:1px solid #30363d;border-radius:5px;color:#e6edf3;font-size:12px;width:calc(100% - 24px);outline:none}
#search:focus{border-color:#58a6ff}
#search::placeholder{color:#484f58}
#node-list{flex:1;overflow-y:auto;padding:4px 8px 8px}
#node-list::-webkit-scrollbar{width:4px}
#node-list::-webkit-scrollbar-thumb{background:#21262d;border-radius:2px}
.nl-item{display:flex;align-items:center;gap:7px;padding:5px 6px;border-radius:4px;cursor:pointer}
.nl-item:hover{background:#1c2128}
.nl-item.active{background:#1f6feb22;outline:1px solid #1f6feb55}
.nl-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.nl-dot.rect{border-radius:2px}
.nl-name{font-size:12px;color:#c9d1d9;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}
.nl-kind{font-size:10px;color:#484f58;flex-shrink:0}
.nl-empty{font-size:12px;color:#484f58;padding:12px 6px}

/* ── Canvas ── */
#main{flex:1;position:relative;overflow:hidden}
#viz{width:100%;height:100%;display:block;cursor:default}

/* ── Tooltip ── */
#tip{position:fixed;background:#161b22;border:1px solid #30363d;border-radius:7px;padding:11px 13px;font-size:12px;pointer-events:none;opacity:0;transition:opacity .1s;max-width:280px;z-index:200;box-shadow:0 4px 20px rgba(0,0,0,.5)}
.tip-name{font-weight:600;color:#58a6ff;font-size:13px;margin-bottom:3px;word-break:break-all}
.tip-kind{color:#8b949e;font-size:11px;margin-bottom:5px}
.tip-sig{font-family:monospace;color:#c9d1d9;font-size:10px;margin-bottom:4px;word-break:break-all;background:#0d1117;padding:4px 6px;border-radius:3px}
.tip-doc{color:#8b949e;font-size:11px;font-style:italic;margin-bottom:4px}
.tip-file{color:#3fb950;font-size:10px}

/* ── Detail panel ── */
#detail{position:absolute;top:14px;right:14px;width:280px;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;display:none;z-index:100;box-shadow:0 4px 24px rgba(0,0,0,.6)}
.det-close{float:right;cursor:pointer;color:#8b949e;font-size:18px;line-height:1;margin-top:-2px}
.det-close:hover{color:#e6edf3}
.det-name{font-weight:700;color:#58a6ff;font-size:13px;margin-bottom:3px;padding-right:18px;word-break:break-all}
.det-kind{font-size:11px;color:#8b949e;margin-bottom:8px}
.det-sig{font-family:monospace;font-size:10px;color:#c9d1d9;background:#0d1117;padding:7px 9px;border-radius:4px;margin-bottom:7px;word-break:break-all;white-space:pre-wrap}
.det-doc{font-size:11px;color:#8b949e;font-style:italic;margin-bottom:7px;line-height:1.5}
.det-file{font-size:11px;color:#3fb950;margin-bottom:4px}
.det-sec{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin:10px 0 5px}
.det-ref{font-size:12px;color:#c9d1d9;padding:2px 0;cursor:pointer}
.det-ref:hover{color:#58a6ff}

/* ── Controls ── */
#ctrls{position:absolute;bottom:14px;right:14px;display:flex;gap:5px}
.cb{background:#21262d;border:1px solid #30363d;color:#c9d1d9;border-radius:5px;padding:6px 11px;cursor:pointer;font-size:11px}
.cb:hover{background:#30363d}
#hint{position:absolute;bottom:14px;left:284px;font-size:11px;color:#3a3f47;pointer-events:none}

/* ── Loading ── */
#loading{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;color:#484f58;font-size:13px;pointer-events:none}
#loading-bar{width:160px;height:3px;background:#21262d;border-radius:2px;margin:10px auto 0;overflow:hidden}
#loading-fill{height:100%;background:#58a6ff;border-radius:2px;width:0%;transition:width .3s}
</style>
</head>
<body>
<div id="sidebar">
  <div id="sb-header">
    <h1>OpenLCM Code Graph</h1>
    <div class="repo">__REPO_LABEL__</div>
  </div>
  <div class="stat-grid">
    <div class="stat-box"><div class="stat-val" id="s-files">—</div><div class="stat-lbl">Files</div></div>
    <div class="stat-box"><div class="stat-val" id="s-syms">—</div><div class="stat-lbl">Symbols</div></div>
    <div class="stat-box"><div class="stat-val" id="s-edges">—</div><div class="stat-lbl">Edges</div></div>
    <div class="stat-box"><div class="stat-val" id="s-langs">—</div><div class="stat-lbl">Languages</div></div>
  </div>
  <div class="section">
    <div class="section-title">Node Types</div>
    <div class="leg-item" data-type="file" data-shape="rect"><div class="leg-dot" style="background:#4A90D9;border-radius:2px"></div><span>File</span></div>
    <div class="leg-item" data-type="class"><div class="leg-dot" style="background:#9B59B6"></div><span>Class</span></div>
    <div class="leg-item" data-type="function"><div class="leg-dot" style="background:#27AE60"></div><span>Function</span></div>
    <div class="leg-item" data-type="method"><div class="leg-dot" style="background:#16A085"></div><span>Method</span></div>
    <div class="leg-item" data-type="import"><div class="leg-dot" style="background:#E67E22"></div><span>Import</span></div>
    <div class="leg-item" data-type="variable"><div class="leg-dot" style="background:#7F8C8D"></div><span>Variable</span></div>
  </div>
  <div class="section">
    <div class="section-title">Edge Types</div>
    <div class="edge-leg edge-item" data-etype="calls"><div class="edge-line" style="background:#888"></div><span style="font-size:12px;color:#c9d1d9">calls</span></div>
    <div class="edge-leg edge-item" data-etype="imports"><div class="edge-line" style="background:#4A90D9"></div><span style="font-size:12px;color:#c9d1d9">imports</span></div>
    <div class="edge-leg edge-item" data-etype="inherits"><div class="edge-line" style="background:#9B59B6;height:3px"></div><span style="font-size:12px;color:#c9d1d9">inherits</span></div>
  </div>
  <input id="search" placeholder="Search symbols…" type="text" autocomplete="off" spellcheck="false">
  <div id="node-list"></div>
</div>

<div id="main">
  <canvas id="viz"></canvas>
  <div id="loading">Simulating layout…<div id="loading-bar"><div id="loading-fill"></div></div></div>
  <div id="tip"></div>
  <div id="detail">
    <span class="det-close" id="det-close">×</span>
    <div class="det-name" id="det-name"></div>
    <div class="det-kind" id="det-kind"></div>
    <div class="det-sig" id="det-sig" style="display:none"></div>
    <div class="det-doc" id="det-doc" style="display:none"></div>
    <div class="det-file" id="det-file"></div>
    <div class="det-sec" id="det-callers-h" style="display:none">Called by</div>
    <div id="det-callers"></div>
    <div class="det-sec" id="det-callees-h" style="display:none">Calls</div>
    <div id="det-callees"></div>
    <div class="det-sec" id="det-inherits-h" style="display:none">Inherits</div>
    <div id="det-inherits"></div>
  </div>
  <div id="ctrls">
    <button class="cb" id="btn-fit">Fit</button>
    <button class="cb" id="btn-reset">Reset</button>
  </div>
  <div id="hint">scroll to zoom · drag to pan · click node for details</div>
</div>

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
"use strict";
const DATA = __DATA_JSON__;

// ── Config ────────────────────────────────────────────────────────────────────
const TYPE_COLOR={file:'#4A90D9',class:'#9B59B6',function:'#27AE60',method:'#16A085',import:'#E67E22',variable:'#7F8C8D'};
const TYPE_R={file:11,class:9,function:8,method:7,import:5,variable:5};
const EDGE_COLOR={calls:'rgba(120,120,120,0.45)',imports:'rgba(74,144,217,0.35)',inherits:'rgba(155,89,182,0.65)'};
const EDGE_W={calls:0.9,imports:0.9,inherits:1.8};

// ── Stats ─────────────────────────────────────────────────────────────────────
const S=DATA.stats;
document.getElementById('s-files').textContent=fmt(S.files);
document.getElementById('s-syms').textContent=fmt(S.symbols);
document.getElementById('s-edges').textContent=fmt(S.edges);
document.getElementById('s-langs').textContent=Object.keys(S.languages||{}).length;
function fmt(n){return n>=1000?(n/1000).toFixed(1)+'k':String(n)}

// ── Node map ──────────────────────────────────────────────────────────────────
const nodeMap={};
DATA.nodes.forEach(n=>nodeMap[n.id]=n);

// ── Canvas setup ──────────────────────────────────────────────────────────────
const canvas=document.getElementById('viz');
const ctx=canvas.getContext('2d');
let dpr=window.devicePixelRatio||1;

function resize(){
  dpr=window.devicePixelRatio||1;
  const W=canvas.offsetWidth,H=canvas.offsetHeight;
  canvas.width=W*dpr; canvas.height=H*dpr;
  scheduleRender();
}
window.addEventListener('resize',resize);
// Initial size
dpr=window.devicePixelRatio||1;
canvas.width=canvas.offsetWidth*dpr;
canvas.height=canvas.offsetHeight*dpr;

// ── State ─────────────────────────────────────────────────────────────────────
let transform=d3.zoomIdentity;
let hovNode=null,activeNode=null,connSet=null;
let isDragging=false,dragNode=null;
let raf=null;

// Hidden sets for type filtering
const hiddenTypes=new Set(),hiddenEdges=new Set();

function scheduleRender(){
  if(raf) return;
  raf=requestAnimationFrame(()=>{raf=null;render();});
}

// ── Simulation ────────────────────────────────────────────────────────────────
const sim=d3.forceSimulation(DATA.nodes)
  .force('link',d3.forceLink(DATA.edges).id(d=>d.id)
    .distance(d=>d.type==='inherits'?80:d.type==='imports'?150:120).strength(0.4))
  .force('charge',d3.forceManyBody().strength(d=>d.type==='file'?-350:d.type==='class'?-200:-120))
  .force('center',d3.forceCenter(0,0))
  .force('collide',d3.forceCollide().radius(d=>(TYPE_R[d.type]||7)+5))
  .alphaDecay(0.022)
  .on('tick',()=>{
    const p=Math.round((1-sim.alpha())*100);
    document.getElementById('loading-fill').style.width=p+'%';
    scheduleRender();
  });

sim.on('end',()=>{
  document.getElementById('loading').style.display='none';
  scheduleRender();
  setTimeout(fitView,200);
});

// ── Zoom ──────────────────────────────────────────────────────────────────────
const zoom=d3.zoom()
  .scaleExtent([0.03,12])
  .filter(ev=>!isDragging||(ev.type!=='mousedown'&&ev.type!=='touchstart'))
  .on('zoom',e=>{transform=e.transform;scheduleRender();});
d3.select(canvas).call(zoom);

// ── Hit test ─────────────────────────────────────────────────────────────────
function toGraph(cx,cy){return[(cx-transform.x)/transform.k,(cy-transform.y)/transform.k]}

function hitTest(gx,gy){
  // Iterate in reverse so topmost (last drawn) wins
  for(let i=DATA.nodes.length-1;i>=0;i--){
    const n=DATA.nodes[i];
    if(!n.x||hiddenTypes.has(n.type)) continue;
    const r=(TYPE_R[n.type]||7)+3; // +3px hit slop
    if(Math.abs(gx-n.x)<=r&&Math.abs(gy-n.y)<=r) return n;
  }
  return null;
}

// ── Pointer events (drag + hover, in capture so they beat D3 zoom) ────────────
canvas.addEventListener('mousedown',e=>{
  const [gx,gy]=toGraph(e.offsetX,e.offsetY);
  const hit=hitTest(gx,gy);
  if(hit){
    e.stopPropagation(); // prevent D3 zoom pan
    isDragging=true; dragNode=hit;
    sim.alphaTarget(0.3).restart();
    hit.fx=hit.x; hit.fy=hit.y;
    canvas.style.cursor='grabbing';
  }
},true);

canvas.addEventListener('mousemove',e=>{
  if(isDragging&&dragNode){
    const[gx,gy]=toGraph(e.offsetX,e.offsetY);
    dragNode.fx=gx; dragNode.fy=gy;
    return;
  }
  const[gx,gy]=toGraph(e.offsetX,e.offsetY);
  const hit=hitTest(gx,gy);
  if(hit!==hovNode){hovNode=hit;canvas.style.cursor=hit?'pointer':'default';scheduleRender();}
  if(hit) showTip(hit,e.clientX,e.clientY);
  else hideTip();
},true);

canvas.addEventListener('mouseup',e=>{
  if(isDragging&&dragNode){
    isDragging=false;
    sim.alphaTarget(0);
    dragNode.fx=null; dragNode.fy=null; dragNode=null;
    canvas.style.cursor='default';
  }
},true);

canvas.addEventListener('mouseleave',()=>{hovNode=null;hideTip();scheduleRender();},true);

canvas.addEventListener('click',e=>{
  if(isDragging) return;
  const[gx,gy]=toGraph(e.offsetX,e.offsetY);
  const hit=hitTest(gx,gy);
  if(hit) selectNode(hit);
  else clearSelect();
},true);

// ── Render ────────────────────────────────────────────────────────────────────
function render(){
  const W=canvas.offsetWidth,H=canvas.offsetHeight;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.save();
  ctx.scale(dpr,dpr);
  ctx.translate(transform.x,transform.y);
  ctx.scale(transform.k,transform.k);

  const k=transform.k;
  const onlyDots=k<0.18;
  const showLabels=k>0.55;

  // ── Edges ──────────────────────────────────────────────────────────────
  if(onlyDots){
    // Ultra-minimal: single grey pass
    ctx.globalAlpha=0.12;
    ctx.strokeStyle='#888';
    ctx.lineWidth=0.6/k;
    ctx.setLineDash([]);
    ctx.beginPath();
    for(const e of DATA.edges){
      if(!e.source.x) continue;
      ctx.moveTo(e.source.x,e.source.y);
      ctx.lineTo(e.target.x,e.target.y);
    }
    ctx.stroke();
    ctx.globalAlpha=1;
  } else {
    // Two-pass per edge type: dim non-connected, bright connected
    for(const [type,color] of Object.entries(EDGE_COLOR)){
      if(hiddenEdges.has(type)) continue;
      ctx.strokeStyle=color;
      ctx.lineWidth=(EDGE_W[type]||1)/k;
      ctx.setLineDash(type==='imports'?[5/k,3/k]:[]);

      if(connSet){
        // Dim pass
        ctx.globalAlpha=0.05;
        ctx.beginPath();
        for(const e of DATA.edges){
          if(e.type!==type||!e.source.x) continue;
          const s=e.source.id||e.source,t=e.target.id||e.target;
          if(!connSet.has(s)&&!connSet.has(t)){ctx.moveTo(e.source.x,e.source.y);ctx.lineTo(e.target.x,e.target.y);}
        }
        ctx.stroke();
        // Bright pass
        ctx.globalAlpha=0.9;
        ctx.beginPath();
        for(const e of DATA.edges){
          if(e.type!==type||!e.source.x) continue;
          const s=e.source.id||e.source,t=e.target.id||e.target;
          if(connSet.has(s)||connSet.has(t)){ctx.moveTo(e.source.x,e.source.y);ctx.lineTo(e.target.x,e.target.y);}
        }
        ctx.stroke();
      } else {
        ctx.globalAlpha=0.5;
        ctx.beginPath();
        for(const e of DATA.edges){
          if(e.type!==type||!e.source.x) continue;
          ctx.moveTo(e.source.x,e.source.y);
          ctx.lineTo(e.target.x,e.target.y);
        }
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);
    ctx.globalAlpha=1;
  }

  // ── Nodes ──────────────────────────────────────────────────────────────
  for(const n of DATA.nodes){
    if(!n.x||hiddenTypes.has(n.type)) continue;
    const isActive=activeNode&&n.id===activeNode.id;
    const isHov=hovNode&&n.id===hovNode.id;
    const isDim=connSet&&!connSet.has(n.id)&&!isActive;

    ctx.globalAlpha=isDim?0.12:1;

    const r=onlyDots?Math.max(1.5,2/k):(TYPE_R[n.type]||7);

    // Glow ring for active/hovered
    if((isActive||isHov)&&!onlyDots){
      ctx.beginPath();
      if(n.type==='file') ctx.rect(n.x-r-3/k,n.y-r-3/k,(r+3/k)*2,(r+3/k)*2);
      else ctx.arc(n.x,n.y,r+3/k,0,Math.PI*2);
      ctx.fillStyle=isActive?'rgba(88,166,255,0.22)':'rgba(255,255,255,0.08)';
      ctx.fill();
    }

    // Node shape
    ctx.beginPath();
    if(n.type==='file'&&!onlyDots) ctx.rect(n.x-r,n.y-r,r*2,r*2);
    else ctx.arc(n.x,n.y,r,0,Math.PI*2);
    ctx.fillStyle=TYPE_COLOR[n.type]||'#555';
    ctx.fill();

    if((isActive||isHov)&&!onlyDots){
      ctx.strokeStyle=isActive?'#58a6ff':'rgba(255,255,255,0.6)';
      ctx.lineWidth=1.5/k;
      ctx.stroke();
    }

    // Labels — only when zoomed in enough and not dimmed
    if(showLabels&&!isDim&&!onlyDots){
      ctx.globalAlpha=isActive?1:0.65;
      ctx.fillStyle=isActive?'#e6edf3':'#8b949e';
      ctx.font=`${9/k}px system-ui,sans-serif`;
      ctx.textAlign='center';
      const lbl=n.label.length>22?n.label.slice(0,20)+'…':n.label;
      ctx.fillText(lbl,n.x,n.y+r+10/k);
    }
  }

  ctx.globalAlpha=1;
  ctx.restore();
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
const tipEl=document.getElementById('tip');
function showTip(n,cx,cy){
  let h=`<div class="tip-name">${esc(n.label)}</div>`+
        `<div class="tip-kind">${n.type}${n.lang?' · '+n.lang:''}</div>`;
  if(n.signature) h+=`<div class="tip-sig">${esc(n.signature.slice(0,120))}</div>`;
  if(n.docstring) h+=`<div class="tip-doc">${esc(n.docstring)}</div>`;
  if(n.file)      h+=`<div class="tip-file">${esc(n.file)}${n.line?':'+n.line:''}</div>`;
  tipEl.innerHTML=h;
  tipEl.style.opacity='1';
  tipEl.style.left=Math.min(cx+15,window.innerWidth-296)+'px';
  tipEl.style.top=Math.max(4,cy-10)+'px';
}
function hideTip(){tipEl.style.opacity='0';}

// ── Select node ───────────────────────────────────────────────────────────────
function selectNode(n){
  activeNode=n;
  // Build connected set
  connSet=new Set([n.id]);
  DATA.edges.forEach(e=>{
    const s=e.source.id||e.source,t=e.target.id||e.target;
    if(s===n.id) connSet.add(t);
    if(t===n.id) connSet.add(s);
  });
  showDetail(n);
  setActiveNl(n.id);
  scheduleRender();
}
function clearSelect(){
  activeNode=null;connSet=null;
  document.getElementById('detail').style.display='none';
  setActiveNl(null);
  scheduleRender();
}
document.getElementById('det-close').onclick=clearSelect;
document.getElementById('btn-reset').onclick=clearSelect;

function showDetail(n){
  const det=document.getElementById('detail');
  document.getElementById('det-name').textContent=n.label;
  document.getElementById('det-kind').textContent=n.type+(n.lang?' · '+n.lang:'')+(n.line?' · line '+n.line:'');
  const sigEl=document.getElementById('det-sig');
  if(n.signature){sigEl.textContent=n.signature;sigEl.style.display='block';}else sigEl.style.display='none';
  const docEl=document.getElementById('det-doc');
  if(n.docstring){docEl.textContent=n.docstring;docEl.style.display='block';}else docEl.style.display='none';
  document.getElementById('det-file').textContent=n.file?(n.file+(n.line?':'+n.line:'')):'';

  function refs(ids,el,hdr){
    const valid=ids.filter(Boolean).slice(0,10);
    document.getElementById(hdr).style.display=valid.length?'block':'none';
    document.getElementById(el).innerHTML=valid.map(m=>`<div class="det-ref" data-id="${m.id}">${esc(m.label)}</div>`).join('');
  }
  const callers=DATA.edges.filter(e=>(e.target.id||e.target)===n.id&&e.type==='calls').map(e=>nodeMap[e.source.id||e.source]);
  const callees=DATA.edges.filter(e=>(e.source.id||e.source)===n.id&&e.type==='calls').map(e=>nodeMap[e.target.id||e.target]);
  const inherits=DATA.edges.filter(e=>(e.source.id||e.source)===n.id&&e.type==='inherits').map(e=>nodeMap[e.target.id||e.target]);
  refs(callers,'det-callers','det-callers-h');
  refs(callees,'det-callees','det-callees-h');
  refs(inherits,'det-inherits','det-inherits-h');
  det.querySelectorAll('.det-ref').forEach(el=>el.addEventListener('click',()=>selectNode(nodeMap[el.dataset.id])));
  det.style.display='block';
}

function setActiveNl(id){
  document.querySelectorAll('.nl-item').forEach(el=>el.classList.toggle('active',el.dataset.id===id));
}

// ── Fit view ──────────────────────────────────────────────────────────────────
function fitView(){
  const nodes=DATA.nodes.filter(n=>n.x!==undefined&&!hiddenTypes.has(n.type));
  if(!nodes.length) return;
  let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
  for(const n of nodes){
    const r=TYPE_R[n.type]||7;
    x0=Math.min(x0,n.x-r); x1=Math.max(x1,n.x+r);
    y0=Math.min(y0,n.y-r); y1=Math.max(y1,n.y+r);
  }
  const W=canvas.offsetWidth,H=canvas.offsetHeight;
  const sc=0.88/Math.max((x1-x0)/W,(y1-y0)/H,0.001);
  const tx=W/2-sc*((x0+x1)/2),ty=H/2-sc*((y0+y1)/2);
  d3.select(canvas).transition().duration(650)
    .call(zoom.transform,d3.zoomIdentity.translate(tx,ty).scale(sc));
}
document.getElementById('btn-fit').onclick=fitView;

// ── Node list sidebar ─────────────────────────────────────────────────────────
const nlEl=document.getElementById('node-list');
function renderList(q=''){
  nlEl.innerHTML='';
  const ql=q.toLowerCase();
  const items=DATA.nodes.filter(n=>!ql||n.label.toLowerCase().includes(ql)||(n.qualified_name||'').toLowerCase().includes(ql)).slice(0,300);
  if(!items.length){nlEl.innerHTML='<div class="nl-empty">No results</div>';return;}
  const frag=document.createDocumentFragment();
  items.forEach(n=>{
    const d=document.createElement('div');
    d.className='nl-item'; d.dataset.id=n.id;
    const isFile=n.type==='file';
    d.innerHTML=`<div class="nl-dot${isFile?' rect':''}" style="background:${TYPE_COLOR[n.type]||'#555'}"></div>`+
      `<div class="nl-name" title="${esc(n.qualified_name||n.label)}">${esc(n.label)}</div>`+
      `<div class="nl-kind">${n.type}</div>`;
    d.addEventListener('click',()=>selectNode(n));
    frag.appendChild(d);
  });
  nlEl.appendChild(frag);
}
renderList();
document.getElementById('search').addEventListener('input',e=>renderList(e.target.value));

// ── Legend toggles ────────────────────────────────────────────────────────────
document.querySelectorAll('.leg-item').forEach(el=>{
  el.addEventListener('click',()=>{
    const t=el.dataset.type;
    if(hiddenTypes.has(t)){hiddenTypes.delete(t);el.classList.remove('off');}
    else{hiddenTypes.add(t);el.classList.add('off');}
    scheduleRender();
  });
});
document.querySelectorAll('.edge-item').forEach(el=>{
  el.addEventListener('click',()=>{
    const t=el.dataset.etype;
    if(hiddenEdges.has(t)){hiddenEdges.delete(t);el.classList.remove('off');}
    else{hiddenEdges.add(t);el.classList.add('off');}
    scheduleRender();
  });
});

// ── Util ──────────────────────────────────────────────────────────────────────
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
</script>
</body>
</html>"""
