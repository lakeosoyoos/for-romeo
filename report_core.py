"""
Unidirectional shortened duplicate report with loss — single direction.
Mirrors ZachRequestShortenedReportWithLoss/lsc_short_report.py (with the
multi-page pagination fixes) but renders one direction only.
"""
import os
import sys
import struct
import base64
import numpy as np
from itertools import combinations
from datetime import datetime
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lsa_event_calculator import parse_sor_with_windows, time_to_dist_m

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOP_N = 25


def load_fiber(filepath):
    parsed = parse_sor_with_windows(filepath)
    events = parsed['events']
    IOR = parsed['IOR']
    with open(filepath, 'rb') as f:
        raw = f.read()
    fxd = raw.find(b'FxdParams\x00', 100)
    ts = struct.unpack_from('<I', raw, fxd + len('FxdParams\x00'))[0] if fxd >= 0 else 0
    first_dist = time_to_dist_m(events[0]['time_of_travel'], IOR) if events else 0
    evt_list = []
    total_splice = 0
    total_fiber_atten = 0
    for i, evt in enumerate(events):
        dist_m = time_to_dist_m(evt['time_of_travel'], IOR)
        dist_km = (dist_m - first_dist) / 1000.0
        splice = evt['splice_loss_fw']
        total_splice += splice
        if i > 0:
            prev_dist = time_to_dist_m(events[i-1]['time_of_travel'], IOR)
            span_km = (dist_m - prev_dist) / 1000.0
            slope_dBkm = evt['slope_raw'] / 1000.0
            total_fiber_atten += slope_dBkm * span_km
        evt_list.append({
            'number': evt['number'],
            'dist_km': dist_km,
            'splice_loss': splice,
            'splice_mdB': round(splice * 1000),
            'reflection': evt['reflection_fw'],
        })
    total_loss = total_splice + total_fiber_atten
    return {'events': evt_list, 'timestamp': ts, 'filesize': len(raw),
            'filename': os.path.basename(filepath),
            'total_splice_dB': total_splice,
            'total_fiber_atten_dB': total_fiber_atten,
            'total_loss_dB': total_loss,
            'total_loss_mdB': round(total_loss * 1000)}


def compare_pairs(fibers):
    fids = sorted(fibers.keys())
    pairs = []
    for a, b in combinations(fids, 2):
        ea = fibers[a]['events']; eb = fibers[b]['events']
        n = min(len(ea), len(eb))
        if n == 0:
            continue
        max_diff = 0
        per_event = []
        for i in range(n):
            d = abs(ea[i]['splice_mdB'] - eb[i]['splice_mdB'])
            max_diff = max(max_diff, d)
            per_event.append({
                'event': ea[i]['number'],
                'dist_km': ea[i]['dist_km'],
                'loss_a': ea[i]['splice_loss'],
                'loss_b': eb[i]['splice_loss'],
                'diff_mdB': d,
            })
        ts_a = fibers[a]['timestamp']; ts_b = fibers[b]['timestamp']
        time_gap = abs(ts_a - ts_b) if ts_a and ts_b else None
        loss_a = fibers[a]['total_loss_mdB']; loss_b = fibers[b]['total_loss_mdB']
        pairs.append({
            'fiber_a': a, 'fiber_b': b, 'max_diff_mdB': max_diff,
            'per_event': per_event, 'timestamp_a': ts_a, 'timestamp_b': ts_b,
            'time_gap_sec': time_gap,
            'total_loss_a': loss_a, 'total_loss_b': loss_b,
            'total_loss_diff': abs(loss_a - loss_b),
        })
    pairs.sort(key=lambda x: x['max_diff_mdB'])
    return pairs


def _histogram_b64(pairs, title_str):
    diffs = [p['max_diff_mdB'] for p in pairs]
    fig, ax = plt.subplots(figsize=(14, 4))
    max_x = max(diffs) + 10
    bins = np.arange(0, max_x, 5)
    ax.hist(diffs, bins=bins, color='#4A90D9', edgecolor='white', linewidth=0.5, alpha=0.85)
    ax.set_xlabel('Max splice diff (mdB)', fontsize=11)
    ax.set_ylabel('Number of pairs', fontsize=11)
    ax.set_title(f'{title_str} — {len(diffs)} pairs', fontsize=13, fontweight='bold')
    stats = (f'Total: {len(diffs)}\n'
             f'Min: {min(diffs):.0f}\n'
             f'Median: {np.median(diffs):.0f}\n'
             f'Mean: {np.mean(diffs):.0f}')
    ax.text(0.98, 0.95, stats, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#ccc', alpha=0.9))
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close()
    return b64


def _rows(pairs, top_n=TOP_N):
    out = ''
    for rank, p in enumerate(pairs[:top_n], 1):
        a, b = p['fiber_a'], p['fiber_b']
        diff = p['max_diff_mdB']
        ts_a_str = datetime.fromtimestamp(p['timestamp_a']).strftime('%m/%d %H:%M:%S') if p['timestamp_a'] else '---'
        ts_b_str = datetime.fromtimestamp(p['timestamp_b']).strftime('%m/%d %H:%M:%S') if p['timestamp_b'] else '---'
        gap = p.get('time_gap_sec')
        if gap is not None:
            if gap < 120:
                gap_str, gap_cls = f'{gap:.0f}s', ' style="color:#C0392B;font-weight:700"'
            elif gap < 3600:
                gap_str, gap_cls = f'{gap/60:.0f}m', ''
            else:
                gap_str, gap_cls = f'{gap/3600:.1f}h', ''
        else:
            gap_str, gap_cls = '---', ''
        evt_cells = ''
        for pe in p['per_event']:
            evt_cells += (f'<td class="center">{pe["loss_a"]:+.3f}</td>'
                          f'<td class="center">{pe["loss_b"]:+.3f}</td>'
                          f'<td class="center" style="font-weight:600">{pe["diff_mdB"]:.0f}</td>')
        loss_a = p.get('total_loss_a', 0)
        loss_b = p.get('total_loss_b', 0)
        loss_diff = p.get('total_loss_diff', 0)
        out += (f'<tr>'
                f'<td class="center">{rank}</td>'
                f'<td class="pair-cell">{a} &#8596; {b}</td>'
                f'<td class="center bold">{diff:.0f}</td>'
                f'<td class="center" style="font-size:8px">{ts_a_str}</td>'
                f'<td class="center" style="font-size:8px">{ts_b_str}</td>'
                f'<td class="center"{gap_cls}>{gap_str}</td>'
                f'{evt_cells}'
                f'<td class="center">{loss_a}</td>'
                f'<td class="center">{loss_b}</td>'
                f'<td class="center" style="font-weight:600">{loss_diff}</td>'
                f'</tr>\n')
    return out


def _evt_headers(pairs):
    h, s = '', ''
    if pairs:
        for pe in pairs[0]['per_event']:
            h += (f'<th colspan="3" style="border-left:2px solid #ddd">'
                  f'Evt #{pe["event"]} ({pe["dist_km"]:.3f} km)</th>')
            s += '<th class="r">Fib 1</th><th class="r">Fib 2</th><th class="r">&#916; mdB</th>'
        h += '<th colspan="3" style="border-left:2px solid #ddd">Total Loss (mdB)</th>'
        s += '<th class="r">Fib 1</th><th class="r">Fib 2</th><th class="r">&#916;</th>'
    return h, s


def build_report(fibers, pairs, route_name, direction_label, fiber_nums):
    generated = datetime.now().strftime('%Y-%m-%d %H:%M')
    n_fibers = len(fiber_nums)
    n_pairs = len(pairs)
    show_chart = n_pairs > 1
    chart_b64 = _histogram_b64(pairs, direction_label) if show_chart else ''
    chart_html = f'<img src="data:image/png;base64,{chart_b64}" class="chart-img" />' if show_chart else ''

    evt_h, evt_s = _evt_headers(pairs)
    diff_rows = _rows(pairs, TOP_N)

    time_sorted = sorted([p for p in pairs if p.get('time_gap_sec') is not None],
                         key=lambda x: x['time_gap_sec'])
    time_rows = _rows(time_sorted, TOP_N)

    loss_sorted = sorted(pairs, key=lambda x: x.get('total_loss_diff', 999))
    loss_rows = _rows(loss_sorted, TOP_N)

    fiber_list = ', '.join(str(f) for f in fiber_nums)

    logo_b64 = ''
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zerodblogo.png')
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            logo_b64 = base64.b64encode(f.read()).decode('ascii')
    logo_html = (f'<div style="text-align:center; margin-bottom:16px;">'
                 f'<img src="data:image/png;base64,{logo_b64}" '
                 f'style="height:60px; margin-left:-30px;" /></div>') if logo_b64 else ''

    min_pair = pairs[0] if pairs else {'max_diff_mdB': 0, 'fiber_a': '-', 'fiber_b': '-'}
    median_val = np.median([p['max_diff_mdB'] for p in pairs]) if pairs else 0

    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{route_name} — Unidirectional Duplicate Report — Fibers {fiber_list}</title>
<style>
@page {{
  size: landscape;
  margin: 10mm 10mm 18mm 10mm;
  @bottom-center {{
    content: "Page " counter(page) " of " counter(pages);
    font-size: 8px; color: #000;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }}
  @bottom-right {{
    content: "\\A9  ZeroDB";
    font-size: 8px; color: #000;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }}
}}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
        color:#2c2c2a; padding:16px; font-size:11px; max-width:1400px; margin:0 auto; }}
h1 {{ font-size:20px; font-weight:500; margin-bottom:2px; }}
h2 {{ font-size:14px; font-weight:500; margin:24px 0 8px; }}
.subtitle {{ font-size:11px; color:#888; margin-bottom:16px; }}
.chart-img {{ width:100%; border-radius:8px; border:1px solid #ddd; margin-bottom:16px; }}
.cards {{ display:flex; gap:10px; margin-bottom:16px; }}
.card {{ flex:1; background:#fff; border:1px solid rgba(0,0,0,.08); border-radius:10px; padding:12px 14px; }}
.card-label {{ font-size:9px; color:#999; margin-bottom:2px; text-transform:uppercase; letter-spacing:.04em; }}
.card-value {{ font-size:22px; font-weight:600; }}
.card-sub {{ font-size:9px; color:#999; margin-top:2px; }}
.table-section {{ page-break-inside:auto; break-inside:auto; }}
.table-section h2 {{ page-break-after:avoid; break-after:avoid; }}
.vote-table {{ width:100%; border-collapse:collapse; font-size:9px;
               font-family:'SF Mono','Courier New',monospace; margin-bottom:4px;
               page-break-inside:auto; break-inside:auto; }}
.vote-table thead {{ display:table-header-group; }}
.vote-table tbody {{ display:table-row-group; }}
.vote-table th {{ background:#f4f3f0; padding:5px 6px; text-align:center;
                  font-weight:600; border:0.5px solid #ddd; font-size:8px; color:#555; }}
.vote-table td {{ padding:4px 6px; border:0.5px solid #ddd; }}
.vote-table tr {{ page-break-inside:avoid; break-inside:avoid; }}
.pair-cell {{ text-align:left !important; font-weight:600; }}
.center {{ text-align:center; }}
.r {{ text-align:right; }}
.bold {{ font-weight:600; }}
.dir-banner {{ background:#2C3E50; color:white; padding:10px 16px; border-radius:8px;
               font-size:14px; font-weight:600; margin:28px 0 12px; }}
</style></head><body>

{logo_html}
<h1>{route_name} — Unidirectional Duplicate Report</h1>
<div class="subtitle">Fibers {fiber_list} &bull; {n_fibers} fibers &bull; {n_pairs} pairs &bull; generated {generated}</div>

<div class="cards">
  <div class="card">
    <div class="card-label">Closest pair</div>
    <div class="card-value">{min_pair["max_diff_mdB"]:.0f} mdB</div>
    <div class="card-sub">{min_pair["fiber_a"]} &#8596; {min_pair["fiber_b"]}</div>
  </div>
  <div class="card">
    <div class="card-label">Median max diff</div>
    <div class="card-value">{median_val:.0f} mdB</div>
    <div class="card-sub">{n_pairs} pairs</div>
  </div>
  <div class="card">
    <div class="card-label">Fibers loaded</div>
    <div class="card-value">{n_fibers}</div>
    <div class="card-sub">from {direction_label}</div>
  </div>
</div>

<div class="dir-banner">Direction: {direction_label}</div>

{chart_html}

<div class="table-section">
<h2>Ranked by Smallest Splice Loss Difference</h2>
<table class="vote-table">
<thead>
<tr><th>#</th><th style="text-align:left">Pair</th><th>Max Diff (mdB)</th><th>Time A</th><th>Time B</th><th>Gap</th>{evt_h}</tr>
<tr><th></th><th style="text-align:left;font-size:7px;color:#888">Fiber 1 &#8596; Fiber 2</th><th></th><th></th><th></th><th></th>{evt_s}</tr>
</thead>
<tbody>
{diff_rows}
</tbody>
</table>
</div>

<div class="table-section">
<h2>Ranked by Shortest Time Gap</h2>
<table class="vote-table">
<thead>
<tr><th>#</th><th style="text-align:left">Pair</th><th>Max Diff (mdB)</th><th>Time A</th><th>Time B</th><th>Gap</th>{evt_h}</tr>
<tr><th></th><th style="text-align:left;font-size:7px;color:#888">Fiber 1 &#8596; Fiber 2</th><th></th><th></th><th></th><th></th>{evt_s}</tr>
</thead>
<tbody>
{time_rows}
</tbody>
</table>
</div>

<div class="table-section">
<h2>Ranked by Smallest Total Loss Difference</h2>
<table class="vote-table">
<thead>
<tr><th>#</th><th style="text-align:left">Pair</th><th>Max Diff (mdB)</th><th>Time A</th><th>Time B</th><th>Gap</th>{evt_h}</tr>
<tr><th></th><th style="text-align:left;font-size:7px;color:#888">Fiber 1 &#8596; Fiber 2</th><th></th><th></th><th></th><th></th>{evt_s}</tr>
</thead>
<tbody>
{loss_rows}
</tbody>
</table>
</div>

</body></html>'''


def find_chrome():
    for p in ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
              '/usr/bin/google-chrome', '/usr/bin/chromium-browser']:
        if os.path.isfile(p):
            return p
    return None


def html_to_pdf(html_path, pdf_path):
    import subprocess
    chrome = find_chrome()
    if not chrome:
        return False, 'Chrome not found'
    r = subprocess.run(
        [chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
         '--run-all-compositor-stages-before-draw',
         '--virtual-time-budget=5000',
         f'--print-to-pdf={os.path.abspath(pdf_path)}',
         '--print-to-pdf-no-header', '--no-pdf-header-footer',
         'file://' + os.path.abspath(html_path)],
        capture_output=True, timeout=180)
    return r.returncode == 0, r.stderr.decode(errors='ignore')[:400]
