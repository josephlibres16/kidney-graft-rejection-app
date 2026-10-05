"""
Kidney Graft Rejection Predictor — web application.

Serves the Optuna-tuned XGBoost model trained on GSE98320 (Notebook 2).
The model is loaded from assets/ and is never retrained or modified here.

Run locally:  python app.py      Deployed:  Hugging Face Spaces (Gradio SDK)
"""

import gc
import gzip
import html
import inspect
import io
import json
import os
import tempfile

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.figure import Figure
import gradio as gr

# =====================================================================
# 1. Assets
# =====================================================================
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

_REQUIRED = ["deploy_meta.json", "probesets.txt.gz", "final_tuned_xgboost.json",
             "top20_dictionary.csv", "demo_specimens.npz"]
_missing = [f for f in _REQUIRED if not os.path.exists(os.path.join(ASSETS, f))]
if _missing:
    raise SystemExit(
        "Missing model assets in assets/: " + ", ".join(_missing) +
        "\nRun the export notebook (Export_Deployment_Assets.ipynb) in Colab, unzip its output "
        "into the assets/ folder of this repository, then start the app again.")

with open(os.path.join(ASSETS, "deploy_meta.json")) as fh:
    META = json.load(fh)

with gzip.open(os.path.join(ASSETS, "probesets.txt.gz"), "rt") as fh:
    PROBES = [ln.strip() for ln in fh if ln.strip()]
PROBE_INDEX = {p: i for i, p in enumerate(PROBES)}
N_PROBES = len(PROBES)

MODEL = xgb.XGBClassifier()
MODEL.load_model(os.path.join(ASSETS, "final_tuned_xgboost.json"))

DICT20 = pd.read_csv(os.path.join(ASSETS, "top20_dictionary.csv"))

_demo = np.load(os.path.join(ASSETS, "demo_specimens.npz"), allow_pickle=True)
DEMO_X = _demo["X"].astype(np.float32)
DEMO_IDS = [str(s) for s in _demo["gsm"]]
DEMO_LABEL = _demo["label"].astype(int)
DEMO_RECORDED = _demo["recorded_prob"].astype(float)

THRESHOLD = float(META.get("threshold", 0.5))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "200"))   # cap per upload; lower it on small instances

assert META["n_probesets"] == N_PROBES, "deploy_meta.json and probesets.txt.gz disagree"

THESIS_TITLE = META.get("thesis_title", "Predicting Kidney Graft Rejection from Biopsy Transcriptomic Data")
AUTHORS = META.get("authors", [])
INSTITUTION = META.get("institution", "School of Information Technology · Mapúa University")

esc = lambda x: html.escape(str(x))
REJ_COLOR, NON_COLOR, INK = "#B42318", "#1D5FA8", "#111827"


# =====================================================================
# 2. Reading an uploaded expression file
# =====================================================================
CHUNK_ROWS = 4000          # rows parsed at a time, so a large file never loads whole


def _open_text(path):
    return (gzip.open(path, "rt", encoding="utf-8", errors="ignore") if path.lower().endswith(".gz")
            else open(path, "rt", encoding="utf-8", errors="ignore"))


def _clean(v):
    return str(v).strip().strip('"')


def _sniff(path):
    """Read only the first lines: is this a GEO series matrix, and which separator does it use?"""
    head = []
    with _open_text(path) as fh:
        for _ in range(400):
            ln = fh.readline()
            if not ln:
                break
            head.append(ln)
    if not head:
        raise ValueError("The file is empty.")
    is_series = any(ln.startswith("!series_matrix_table_begin") for ln in head)
    body = "".join(head)
    if is_series:
        body = body.split("!series_matrix_table_begin", 1)[-1]
    return is_series, ("\t" if body.count("\t") >= body.count(",") else ",")


def _table_handle(path, is_series):
    """Open the file positioned on the header row of the expression table."""
    fh = _open_text(path)
    if is_series:
        for ln in fh:
            if ln.startswith("!series_matrix_table_begin"):
                break
        else:
            fh.close()
            raise ValueError("The series matrix contains no expression table.")
    return fh


def build_matrix(path):
    """
    Returns (X aligned to the model's probesets, specimen IDs, notes, coverage).

    The file is parsed in chunks and written straight into a float32 array, so memory stays
    flat no matter how large the upload is. Probesets the file does not contain are left
    missing; XGBoost handles missing values natively.
    """
    is_series, sep = _sniff(path)
    notes = ["Read as a GEO series matrix." if is_series else "Read as a table."]

    # ---- look at the header and a few rows to decide the layout ----
    with _table_handle(path, is_series) as fh:
        header = fh.readline()
        if not header:
            raise ValueError("The expression table has no header row.")
        columns = [_clean(c) for c in header.rstrip("\n").split(sep)]
        probe_cols = sum(c in PROBE_INDEX for c in columns[1:])
        probe_rows = 0
        for _ in range(200):
            ln = fh.readline()
            if not ln or ln.startswith("!"):
                break
            if _clean(ln.split(sep, 1)[0]) in PROBE_INDEX:
                probe_rows += 1

    if probe_rows == 0 and probe_cols == 0:
        raise ValueError(
            "No PrimeView (GPL15207) probeset IDs were recognised in this file. "
            "The app expects the GSE98320 probeset IDs, for example 11719943_at.")

    matched = set()

    if probe_cols > probe_rows:
        # ---- specimens as rows: few rows, read them directly ----
        # Parsed by hand rather than with pandas: a table this wide (tens of thousands of
        # columns) costs hundreds of megabytes in a DataFrame but almost nothing row by row.
        notes.append("Detected specimens as rows.")
        targets = []
        for k, c in enumerate(columns):
            j = PROBE_INDEX.get(c)
            if j is not None and j not in matched:
                targets.append((k, j))
                matched.add(j)
        ids, rows = [], []
        with _table_handle(path, is_series) as fh:
            fh.readline()                                   # skip the header row
            for ln in fh:
                if ln.startswith("!") or not ln.strip():
                    continue
                parts = ln.rstrip("\n").split(sep)
                row = np.full(N_PROBES, np.nan, dtype=np.float32)
                for k, j in targets:
                    if k < len(parts):
                        try:
                            row[j] = float(_clean(parts[k]))
                        except ValueError:
                            pass
                ids.append(_clean(parts[0]))
                rows.append(row)
                if len(ids) >= MAX_SAMPLES:
                    break
        if not rows:
            raise ValueError("The file has a header but no specimen rows.")
        X = np.vstack(rows)
        del rows
        gc.collect()
    else:
        # ---- probesets as rows (the GEO layout): stream the table ----
        ids = columns[1:]
        if len(ids) > MAX_SAMPLES:
            notes.append(f"File holds {len(ids)} specimens; only the first {MAX_SAMPLES} were scored.")
            ids = ids[:MAX_SAMPLES]
        X = np.full((len(ids), N_PROBES), np.nan, dtype=np.float32)
        with _table_handle(path, is_series) as fh:
            fh.readline()                                   # skip the header row
            reader = pd.read_csv(fh, sep=sep, header=None, index_col=0,
                                 usecols=range(len(ids) + 1), chunksize=CHUNK_ROWS)
            for chunk in reader:
                rows, keep_mask = [], []
                for i in chunk.index:
                    j = PROBE_INDEX.get(_clean(i))
                    new = j is not None and j not in matched
                    keep_mask.append(new)
                    if new:
                        rows.append(j)
                        matched.add(j)
                if not rows:
                    continue
                vals = chunk[keep_mask].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
                X[:, rows] = vals.T
        del reader
        gc.collect()

    coverage = len(matched) / N_PROBES
    if coverage < 0.5:
        raise ValueError(f"Only {coverage:.1%} of the model's probesets were found in this file "
                         f"({len(matched):,} of {N_PROBES:,}). Scoring was not attempted.")
    notes.append(f"Matched {len(matched):,} of {N_PROBES:,} probesets ({coverage:.1%}).")
    if coverage < 0.95:
        notes.append("Coverage is below 95%; treat these scores with caution.")
    return X, [str(i) for i in ids], notes, coverage


def predict(X):
    probs = MODEL.predict_proba(X)[:, 1].astype(float)
    return probs


# =====================================================================
# 3. Marker panel (top-20 probesets, no SHAP)
# =====================================================================
def panel_table(x_row):
    d = DICT20.copy()
    vals = [float(x_row[PROBE_INDEX[p]]) if p in PROBE_INDEX else np.nan for p in d["probeset"]]
    d["value"] = vals
    d["closer_to"] = np.where(
        np.isnan(d["value"]), "not in file",
        np.where((d["value"] - d["mean_log2_rejection"]).abs() <= (d["value"] - d["mean_log2_non_rejection"]).abs(),
                 "rejection", "non-rejection"))
    return d


def panel_summary(d):
    usable = d[d["closer_to"] != "not in file"]
    n_rej = int((usable["closer_to"] == "rejection").sum())
    return n_rej, len(usable)


def panel_figure(d, specimen):
    """Each top-20 probeset: this biopsy's expression against the two training class averages."""
    d = d.iloc[::-1].reset_index(drop=True)                   # rank 1 at the top
    y = np.arange(len(d))
    fig = Figure(figsize=(10.5, 0.55 * len(d) + 2.0), dpi=120)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)

    for i in y:
        lo, hi = sorted([d.loc[i, "mean_log2_non_rejection"], d.loc[i, "mean_log2_rejection"]])
        ax.plot([lo, hi], [i, i], color="#D7DBE2", lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(d["mean_log2_non_rejection"], y, s=58, color=NON_COLOR, zorder=3, label="Training average · non-rejection")
    ax.scatter(d["mean_log2_rejection"], y, s=58, color=REJ_COLOR, zorder=3, label="Training average · rejection")
    ok = d["value"].notna().to_numpy()
    ax.scatter(d.loc[ok, "value"], y[ok], s=135, marker="D", facecolor="white",
               edgecolor=INK, linewidth=1.8, zorder=4, label="This biopsy")

    ax.set_yticks(y)
    def short(name):
        name = str(name)
        return name if len(name) <= 18 else name.split(" /// ")[0] + " et al."
    ax.set_yticklabels([f"{short(r.gene_symbol)}  ·  {r.probeset}" for r in d.itertuples()], fontsize=10.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10.5, colors="#4B5563")
    ax.grid(axis="x", color="#EEF0F4", zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.set_xlabel("log2 expression", fontsize=11.5, color="#374151", labelpad=8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=10.5)
    fig.text(0.5, 0.012, f"{specimen} · top 20 probesets by gain in the final model · "
                         f"class averages come from the training partition only",
             ha="center", va="bottom", fontsize=9.5, color="#6B7280")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    return fig


# =====================================================================
# 4. HTML building blocks
# =====================================================================
CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
footer { display: none !important; }
.kg-header { background: linear-gradient(135deg, #1E2A5A 0%, #2F3F86 100%); color: #fff; border-radius: 14px; padding: 26px 30px; }
.kg-header .kg-kicker { font-size: 13px; letter-spacing: .12em; text-transform: uppercase; opacity: .8; }
.kg-header h1 { color: #fff !important; font-size: 34px; line-height: 1.15; margin: 6px 0 8px; font-weight: 750; }
.kg-header .kg-title { font-size: 16.5px; line-height: 1.45; opacity: .95; max-width: 980px; }
.kg-header .kg-authors { margin-top: 12px; font-size: 15px; opacity: .9; }
.kg-badges { margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px; }
.kg-badge { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); padding: 4px 11px; border-radius: 999px; font-size: 13.5px; color: #fff; }
.kg-notice { background: #FFF8E6; border: 1px solid #F3D48B; color: #5C4200; border-radius: 10px; padding: 11px 16px; font-size: 14.5px; line-height: 1.5; }
.kg-step { display: flex; align-items: center; gap: 12px; margin: 6px 0 2px; }
.kg-step .kg-num { width: 32px; height: 32px; border-radius: 50%; background: #3F4FA8; color: #fff; display: grid; place-items: center; font-weight: 700; font-size: 16px; flex: none; }
.kg-step .kg-h { font-size: 21px; font-weight: 700; color: #111827; }
.kg-step .kg-d { font-size: 15px; color: #4B5563; }
.kg-card { background: #fff; border: 1px solid #E5E7EB; border-radius: 14px; padding: 22px 26px; color: #111827; }
.kg-empty { text-align: center; color: #6B7280; padding: 38px 20px; border-style: dashed; }
.kg-empty .kg-empty-t { font-size: 20px; font-weight: 650; color: #374151; margin-bottom: 6px; }
.kg-verdict { border-left-width: 8px; }
.kg-verdict.rej { border-left-color: #B42318; } .kg-verdict.non { border-left-color: #1D5FA8; }
.kg-vtop { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; flex-wrap: wrap; }
.kg-eyebrow { font-size: 13px; letter-spacing: .1em; text-transform: uppercase; color: #6B7280; font-weight: 600; }
.kg-vlabel { font-size: 36px; font-weight: 800; line-height: 1.1; margin-top: 4px; }
.rej .kg-vlabel { color: #B42318; } .non .kg-vlabel { color: #1D5FA8; }
.kg-vsub { font-size: 15.5px; color: #4B5563; margin-top: 6px; }
.kg-score { text-align: right; }
.kg-score-num { font-size: 46px; font-weight: 800; line-height: 1; font-variant-numeric: tabular-nums; color: #111827; }
.kg-score-cap { font-size: 13.5px; color: #6B7280; margin-top: 4px; }
.kg-meter { position: relative; height: 16px; background: #EEF0F4; border-radius: 999px; margin: 30px 0 8px; }
.kg-meter-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; }
.rej .kg-meter-fill { background: #B42318; } .non .kg-meter-fill { background: #1D5FA8; }
.kg-meter-thr { position: absolute; top: -9px; bottom: -9px; width: 3px; background: #111827; border-radius: 2px; }
.kg-meter-thr span { position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%); white-space: nowrap; font-size: 12.5px; font-weight: 700; color: #111827; margin-bottom: 2px; }
.kg-scale { display: flex; justify-content: space-between; font-size: 13px; color: #6B7280; }
.kg-facts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 20px; }
@media (max-width: 720px) { .kg-facts { grid-template-columns: 1fr; } .kg-score { text-align: left; } }
.kg-fact { background: #F9FAFB; border-radius: 10px; padding: 12px 14px; }
.kg-fact .n { font-size: 22px; font-weight: 750; color: #111827; font-variant-numeric: tabular-nums; }
.kg-fact .l { font-size: 13.5px; color: #4B5563; margin-top: 2px; }
.kg-warn { margin-top: 16px; background: #FFF8E6; border: 1px solid #F3D48B; border-radius: 10px; padding: 10px 14px; font-size: 14.5px; color: #5C4200; }
.kg-foot { margin-top: 12px; font-size: 13px; color: #6B7280; }
.kg-tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
@media (max-width: 720px) { .kg-tiles { grid-template-columns: repeat(2, 1fr); } }
.kg-tile { background: #fff; border: 1px solid #E5E7EB; border-radius: 12px; padding: 14px 16px; }
.kg-tile .n { font-size: 34px; font-weight: 800; line-height: 1.1; color: #111827; font-variant-numeric: tabular-nums; }
.kg-tile .l { font-size: 14px; color: #4B5563; margin-top: 2px; }
.kg-tile.rej .n { color: #B42318; } .kg-tile.non .n { color: #1D5FA8; } .kg-tile.warn .n { color: #A15C00; }
.kg-notes { margin-top: 10px; font-size: 14px; color: #4B5563; }
.kg-tablewrap { max-height: 460px; overflow: auto; border: 1px solid #E5E7EB; border-radius: 12px; background: #fff; }
table.kg-table { width: 100%; border-collapse: collapse; font-size: 15.5px; color: #111827; margin: 0 !important; }
table.kg-table th { position: sticky; top: 0; background: #F3F4F6; text-align: left; font-size: 13.5px; text-transform: uppercase; letter-spacing: .05em; color: #4B5563; padding: 10px 14px; border: 0 !important; z-index: 1; }
table.kg-table td { padding: 9px 14px; border: 0 !important; border-top: 1px solid #EEF0F4 !important; font-variant-numeric: tabular-nums; background: #fff; }
table.kg-table tr:hover td { background: #F9FAFB; }
.kg-mini { display: inline-block; vertical-align: middle; width: 90px; height: 8px; background: #EEF0F4; border-radius: 999px; margin-left: 10px; position: relative; overflow: hidden; }
.kg-mini i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; }
.kg-pill { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 14px; font-weight: 650; white-space: nowrap; }
.kg-pill.rej { background: #FDECEA; color: #B42318; } .kg-pill.non { background: #E8F0FB; color: #1D5FA8; }
.kg-muted { color: #9CA3AF; }
.kg-about h3 { font-size: 20px; margin: 18px 0 10px !important; color: #111827; }
.kg-flow { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.kg-flow .c { background: #EEF0FB; color: #1E2A5A; border-radius: 8px; padding: 8px 12px; font-size: 14.5px; font-weight: 600; }
.kg-flow .a { color: #9CA3AF; font-weight: 700; }
.kg-metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
@media (max-width: 720px) { .kg-metrics { grid-template-columns: 1fr; } }
.kg-mcard { border: 1px solid #E5E7EB; border-radius: 12px; padding: 16px 18px; background: #fff; }
.kg-mcard .n { font-size: 30px; font-weight: 800; color: #1E2A5A; font-variant-numeric: tabular-nums; }
.kg-mcard .l { font-size: 14px; color: #4B5563; }
.kg-mcard .c { font-size: 12.5px; color: #9CA3AF; margin-top: 2px; }
.kg-about ul { font-size: 15.5px; line-height: 1.6; color: #374151; }
.kg-about code { font-size: 13px; }
"""

FORCE_LIGHT_JS = """
() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    window.location.replace(u.href);
  }
}
"""


def header_html():
    badges = [f"{META['n_probesets']:,} probesets", "XGBoost + Optuna (Bayesian)",
              f"GSE98320 · {META['n_train']} train / {META['n_test']} test",
              f"Test ROC-AUC {META['test_metrics']['roc_auc']:.3f}"]
    return f"""
<div class="kg-header">
  <div class="kg-kicker">Thesis demonstration · {esc(INSTITUTION)}</div>
  <h1>Kidney Graft Rejection Predictor</h1>
  <div class="kg-title">{esc(THESIS_TITLE)}</div>
  <div class="kg-authors">{' · '.join(esc(a) for a in AUTHORS)}</div>
  <div class="kg-badges">{''.join(f'<span class="kg-badge">{esc(b)}</span>' for b in badges)}</div>
</div>"""


NOTICE_HTML = """
<div class="kg-notice"><b>Research prototype — not for clinical use.</b>
This application demonstrates a thesis model. It accepts kidney biopsy gene expression profiles measured on the
Affymetrix PrimeView platform (GPL15207) and normalised the same way as the GSE98320 series matrix. Profiles from
another platform or normalisation will produce meaningless scores, and the model is trained on SMOTE-balanced data,
so its output ranks risk rather than estimating a calibrated probability.</div>"""


def step_html(n, title, desc=""):
    return (f'<div class="kg-step"><div class="kg-num">{n}</div><div><div class="kg-h">{esc(title)}</div>'
            f'{f"<div class=kg-d>{esc(desc)}</div>" if desc else ""}</div></div>')


def empty_card(msg):
    return f'<div class="kg-card kg-empty"><div class="kg-empty-t">No prediction yet</div><div>{esc(msg)}</div></div>'


def verdict_card(p, specimen, d, coverage=None, extra=""):
    p = float(p)
    is_rej = p >= THRESHOLD
    cls = "rej" if is_rej else "non"
    label = "Graft Rejection" if is_rej else "Non-rejection"
    n_rej, n_used = panel_summary(d)
    cov = f"{coverage:.1%}" if coverage is not None else "100%"
    warn = ""
    if coverage is not None and coverage < 0.95:
        warn = ('<div class="kg-warn"><b>Caution:</b> this file covers only '
                f'{cov} of the probesets the model was trained on. Missing values were treated as missing data.</div>')
    return f"""
<div class="kg-card kg-verdict {cls}">
  <div class="kg-vtop">
    <div>
      <div class="kg-eyebrow">Prediction · {esc(specimen)}</div>
      <div class="kg-vlabel">{label}</div>
      <div class="kg-vsub">Model score is {'at or above' if is_rej else 'below'} the pre-specified threshold τ = {THRESHOLD:.2f}</div>
    </div>
    <div class="kg-score">
      <div class="kg-score-num">{p:.3f}</div>
      <div class="kg-score-cap">Model score · rejection risk</div>
    </div>
  </div>
  <div class="kg-meter">
    <div class="kg-meter-fill" style="width:{max(0.0, min(100.0, p * 100)):.1f}%"></div>
    <div class="kg-meter-thr" style="left:calc({THRESHOLD * 100:.1f}% - 1.5px)"><span>τ = {THRESHOLD:.2f}</span></div>
  </div>
  <div class="kg-scale"><span>0.0 · Non-rejection</span><span>Rejection · 1.0</span></div>
  <div class="kg-facts">
    <div class="kg-fact"><div class="n">{n_rej} of {n_used}</div><div class="l">Top-20 probesets closer to the training rejection average</div></div>
    <div class="kg-fact"><div class="n">{cov}</div><div class="l">Model probesets present in this profile</div></div>
    <div class="kg-fact"><div class="n">{META['n_probesets_used']:,}</div><div class="l">Probesets the final model actually splits on</div></div>
  </div>
  {warn}{extra}
  <div class="kg-foot">Threshold, model and data dictionary are fixed from the final notebook run; this application performs no training.</div>
</div>"""


def tiles_html(n, n_rej, coverage, notes):
    notes_html = "".join(f"<div>· {esc(x)}</div>" for x in notes)
    return f"""
<div class="kg-tiles">
  <div class="kg-tile"><div class="n">{n}</div><div class="l">Biopsies scored</div></div>
  <div class="kg-tile rej"><div class="n">{n_rej}</div><div class="l">Predicted graft rejection</div></div>
  <div class="kg-tile non"><div class="n">{n - n_rej}</div><div class="l">Predicted non-rejection</div></div>
  <div class="kg-tile warn"><div class="n">{coverage:.0%}</div><div class="l">Probeset coverage of the file</div></div>
</div>{f'<div class="kg-notes">{notes_html}</div>' if notes else ''}"""


def table_html(ids, probs):
    rows = []
    for k, (sid, p) in enumerate(zip(ids, probs), 1):
        rej = p >= THRESHOLD
        cls, color = ("rej", REJ_COLOR) if rej else ("non", NON_COLOR)
        rows.append(f"<tr><td class='kg-muted'>{k}</td><td><b>{esc(sid)}</b></td>"
                    f"<td>{p:.3f}<span class='kg-mini'><i style='width:{p * 100:.1f}%;background:{color}'></i></span></td>"
                    f"<td><span class='kg-pill {cls}'>{'Graft Rejection' if rej else 'Non-rejection'}</span></td></tr>")
    return ("<div class='kg-tablewrap'><table class='kg-table'><thead><tr><th>#</th><th>Specimen</th>"
            "<th>Model score</th><th>Prediction</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def panel_html(d):
    rows = []
    for r in d.itertuples():
        if r.closer_to == "not in file":
            val, pill = "<span class='kg-muted'>not in file</span>", "<span class='kg-muted'>—</span>"
        else:
            val = f"{r.value:.2f}"
            cls = "rej" if r.closer_to == "rejection" else "non"
            pill = f"<span class='kg-pill {cls}'>{esc(r.closer_to)}</span>"
        rows.append(f"<tr><td class='kg-muted'>{r.rank}</td><td><b>{esc(r.gene_symbol)}</b></td>"
                    f"<td class='kg-muted'>{esc(r.probeset)}</td><td>{val}</td>"
                    f"<td>{r.mean_log2_rejection:.2f}</td><td>{r.mean_log2_non_rejection:.2f}</td><td>{pill}</td></tr>")
    return ("<div class='kg-tablewrap'><table class='kg-table'><thead><tr><th>#</th><th>Gene</th><th>Probeset</th>"
            "<th>This biopsy</th><th>Train avg · rejection</th><th>Train avg · non-rejection</th><th>Closer to</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def about_html():
    t, c = META["test_metrics"], META["cv_metrics_tuned"]
    steps = ["GSE98320 biopsies", f"{META['n_train']} train / {META['n_test']} test (locked split)",
             f"All {META['n_probesets']:,} probesets", "SMOTE on training data only",
             "Optuna TPE, 50 trials, 5-fold CV", "Tuned XGBoost", "Held-out test, run once"]
    flow = '<span class="a">→</span>'.join(f'<span class="c">{esc(s)}</span>' for s in steps)
    cards = "".join(
        f'<div class="kg-mcard"><div class="n">{t[k]:.3f}</div><div class="l">{lab}</div>'
        f'<div class="c">cross-validated {c[k]:.3f}</div></div>'
        for k, lab in [("roc_auc", "ROC-AUC"), ("auprc", "AUPRC"), ("f1", "F1"),
                       ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("accuracy", "Accuracy")])
    dict_rows = "".join(
        f"<tr><td class='kg-muted'>{r.rank}</td><td><b>{esc(r.gene_symbol)}</b></td><td class='kg-muted'>{esc(r.probeset)}</td>"
        f"<td>{r.gain:.1f}</td><td>{r.log2_fold_change:+.2f}</td><td>{esc(r.higher_in)}</td></tr>"
        for r in DICT20.itertuples())
    params = ", ".join(f"{k} = {round(v, 5) if isinstance(v, float) else v}" for k, v in META["best_params"].items())
    v = META.get("versions", {})
    return f"""
<div class="kg-about">
  <h3>How the model was built</h3><div class="kg-flow">{flow}</div>
  <h3>Held-out test performance ({META['n_test']} biopsies, scored once)</h3>
  <div class="kg-metrics">{cards}</div>
  <h3>Top-20 probesets by gain (data dictionary)</h3>{
      "<div class='kg-tablewrap'><table class='kg-table'><thead><tr><th>#</th><th>Gene</th><th>Probeset</th>"
      "<th>Gain</th><th>log2 FC</th><th>Higher in</th></tr></thead><tbody>" + dict_rows + "</tbody></table></div>"}
  <h3>Reading the output</h3>
  <ul>
    <li><b>Prediction</b> is Graft Rejection when the model score reaches the pre-specified threshold τ = {THRESHOLD:.2f}.</li>
    <li><b>Model score</b> is the tuned XGBoost output for the rejection class. It ranks biopsies by risk and is not a calibrated probability.</li>
    <li><b>Marker panel</b> compares this biopsy with the training averages of the 20 probesets the model relies on most.
        It describes the input, not the model's internal reasoning, and no SHAP values are used anywhere in this application.</li>
  </ul>
  <h3>Proof of run</h3>
  <ul>
    <li>Final model: <code>final_tuned_xgboost.json</code>, {META['n_probesets_used']:,} probesets used out of {META['n_probesets']:,}, {META.get('n_trees', '—')} trees.</li>
    <li>Best Optuna hyperparameters: <code>{esc(params)}</code></li>
    <li>Training run: {esc(META.get('run_finished_utc', '—'))} · xgboost {esc(v.get('xgboost', '—'))}, scikit-learn {esc(v.get('scikit-learn', '—'))}, numpy {esc(v.get('numpy', '—'))}</li>
    <li>Series matrix SHA-256: <code>{esc(META.get('series_matrix_sha256', '—'))}</code></li>
    <li>Model file SHA-256: <code>{esc(META.get('model_sha256', '—'))}</code></li>
  </ul>
</div>"""


# =====================================================================
# 5. Handlers
# =====================================================================
def _path(f):
    if f is None:
        return None
    return f if isinstance(f, str) else getattr(f, "name", None) or getattr(f, "path", None)


def run_demo(sid):
    if not sid:
        raise gr.Error("Pick a demo biopsy first.")
    i = DEMO_IDS.index(sid)
    x = DEMO_X[i]
    p = float(predict(x.reshape(1, -1))[0])
    d = panel_table(x)
    truth = "rejection" if DEMO_LABEL[i] == 1 else "non-rejection"
    agree = "matches" if (p >= THRESHOLD) == bool(DEMO_LABEL[i]) else "differs from"
    extra = (f'<div class="kg-foot">Held-out test biopsy. Biopsy diagnosis recorded in GEO: <b>{truth}</b> '
             f'— the prediction {agree} it. Score recorded in the notebook run: {DEMO_RECORDED[i]:.3f}.</div>')
    return verdict_card(p, sid, d, None, extra), panel_figure(d, sid), panel_html(d)


def run_upload(file, progress=gr.Progress()):
    path = _path(file)
    if not path:
        raise gr.Error("Upload an expression file first.")
    try:
        progress(0.1, desc="Reading file")
        X, ids, notes, coverage = build_matrix(path)
        progress(0.6, desc=f"Scoring {len(ids)} biopsies")
        probs = predict(X)
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(str(e))

    n_rej = int((probs >= THRESHOLD).sum())
    out = os.path.join(tempfile.mkdtemp(), "graft_rejection_predictions.csv")
    pd.DataFrame({"specimen": ids, "model_score": np.round(probs, 6),
                  "prediction": np.where(probs >= THRESHOLD, "Graft Rejection", "Non-rejection")}
                 ).to_csv(out, index=False)

    state = {"X": X, "ids": ids, "probs": probs, "coverage": coverage}
    card, fig, tbl = _explain_upload(0, state)
    return (tiles_html(len(ids), n_rej, coverage, notes), table_html(ids, probs),
            gr.update(value=out, visible=True), gr.update(choices=ids, value=ids[0], visible=True),
            card, fig, tbl, state)


def _explain_upload(i, state):
    x = state["X"][i]
    d = panel_table(x)
    return (verdict_card(state["probs"][i], state["ids"][i], d, state["coverage"]),
            panel_figure(d, state["ids"][i]), panel_html(d))


def explain_selected(sid, state):
    if not state or sid is None:
        return gr.update(), gr.update(), gr.update()
    return _explain_upload(state["ids"].index(sid), state)


def clear_upload():
    return (None, "", "", gr.update(value=None, visible=False), gr.update(choices=[], value=None, visible=False),
            empty_card("Upload a file and press Score to see results here."), None, "", None)


# =====================================================================
# 6. Interface
# =====================================================================
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate", text_size=gr.themes.sizes.text_lg,
                       radius_size=gr.themes.sizes.radius_md,
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])

_opts = {"title": "Kidney Graft Rejection Predictor", "theme": THEME, "css": CSS, "js": FORCE_LIGHT_JS}
_b = inspect.signature(gr.Blocks.__init__).parameters
_l = inspect.signature(gr.Blocks.launch).parameters
BLOCKS_KW = {k: v for k, v in _opts.items() if k in _b}
LAUNCH_KW = {k: v for k, v in _opts.items() if k not in _b and k in _l}

DEMO_LABELS = [f"{sid} — recorded: {'rejection' if lab else 'non-rejection'}"
               for sid, lab in zip(DEMO_IDS, DEMO_LABEL)]
LABEL_TO_ID = dict(zip(DEMO_LABELS, DEMO_IDS))

with gr.Blocks(**BLOCKS_KW) as demo:
    gr.HTML(header_html())
    gr.HTML(NOTICE_HTML)

    with gr.Tabs():
        with gr.Tab("Demo biopsies"):
            gr.HTML(step_html(1, "Choose a biopsy from the held-out test set",
                              "These profiles are bundled with the application, so a prediction needs no upload."))
            with gr.Row(equal_height=True):
                demo_dd = gr.Dropdown(label="Held-out test biopsy", choices=DEMO_LABELS, value=DEMO_LABELS[0], scale=3)
                demo_btn = gr.Button("Score this biopsy", variant="primary", size="lg", scale=1)
            gr.HTML(step_html(2, "Result"))
            demo_card = gr.HTML(empty_card("Choose a biopsy and press Score."))
            demo_plot = gr.Plot(label="Marker panel: this biopsy against the training averages")
            with gr.Accordion("Marker panel as a table", open=False):
                demo_panel = gr.HTML("")

        with gr.Tab("Score your own file"):
            gr.HTML(step_html(1, "Upload a biopsy expression file",
                              "GEO series matrix (.txt or .txt.gz) or a CSV/TSV of probesets by specimens. "
                              "PrimeView (GPL15207) probeset IDs, log2 values on the GSE98320 scale."))
            with gr.Row():
                up_file = gr.File(label="Expression file", file_types=[".txt", ".gz", ".csv", ".tsv"], scale=3)
                with gr.Column(scale=2):
                    gr.Markdown(f"**Accepted layouts**\n\n"
                                f"- Probesets as rows, specimens as columns (GEO layout)\n"
                                f"- Specimens as rows, probesets as columns (detected automatically)\n"
                                f"- Up to {MAX_SAMPLES} specimens per file; missing probesets are allowed")
            with gr.Row():
                up_btn = gr.Button("Score file", variant="primary", size="lg", scale=3)
                clr_btn = gr.Button("Clear", variant="secondary", size="lg", scale=1)
            gr.HTML(step_html(2, "Results"))
            up_tiles = gr.HTML("")
            up_table = gr.HTML("")
            up_dl = gr.DownloadButton("Download predictions (CSV)", visible=False, variant="secondary")
            gr.HTML(step_html(3, "Inspect one biopsy"))
            up_dd = gr.Dropdown(label="Specimen", choices=[], visible=False)
            up_card = gr.HTML(empty_card("Upload a file and press Score to see results here."))
            up_plot = gr.Plot(label="Marker panel: this biopsy against the training averages")
            with gr.Accordion("Marker panel as a table", open=False):
                up_panel = gr.HTML("")
            up_state = gr.State()

        with gr.Tab("About the model"):
            gr.HTML(about_html())

    demo_btn.click(lambda lab: run_demo(LABEL_TO_ID.get(lab)), inputs=demo_dd,
                   outputs=[demo_card, demo_plot, demo_panel])
    up_btn.click(run_upload, inputs=up_file,
                 outputs=[up_tiles, up_table, up_dl, up_dd, up_card, up_plot, up_panel, up_state])
    up_dd.input(explain_selected, inputs=[up_dd, up_state], outputs=[up_card, up_plot, up_panel])
    clr_btn.click(clear_upload, inputs=None,
                  outputs=[up_file, up_tiles, up_table, up_dl, up_dd, up_card, up_plot, up_panel, up_state])

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0",
                                   server_port=int(os.environ.get("PORT", 7860)),
                                   **LAUNCH_KW)
