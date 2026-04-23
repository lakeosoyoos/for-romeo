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


def _read_cstr(raw, offset):
    end = raw.find(b'\x00', offset)
    if end < 0:
        return '', offset
    try:
        s = raw[offset:end].decode('latin-1', errors='replace').strip()
    except Exception:
        s = ''
    return s, end + 1


def parse_gen_params(filepath):
    """Pull route-level metadata from the SOR GenParams block.

    Returns a dict with whatever could be read (empty strings otherwise):
    cable_id, fiber_id, location_a, location_b, cable_code, operator, comment.
    """
    with open(filepath, 'rb') as f:
        raw = f.read()
    marker = b'GenParams\x00'
    i = raw.find(marker, 50)
    if i < 0:
        return {}
    p = i + len(marker)
    # Telcordia SR-4731: language code (2 bytes) then strings in order.
    # Skip language code bytes (2). Some firmwares also have a 2-byte index;
    # we are lenient and just skip the next 2 bytes unconditionally.
    p += 2
    out = {}
    for key in ('cable_id', 'fiber_id', 'fiber_type_code',
                'wavelength_code', 'location_a', 'location_b',
                'cable_code', 'build_condition'):
        if key in ('fiber_type_code', 'wavelength_code'):
            if p + 2 <= len(raw):
                out[key] = struct.unpack_from('<H', raw, p)[0]
                p += 2
            continue
        s, p = _read_cstr(raw, p)
        out[key] = s
    # two 4-byte offsets (user offset + distance)
    p += 8
    for key in ('operator', 'comment'):
        s, p = _read_cstr(raw, p)
        out[key] = s
    return out


def _find_key(tree, targets_lower):
    """Recursively walk a JSON tree and return the first string value whose
    key (case-insensitive) matches any of `targets_lower`. Returns '' if not
    found."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            if isinstance(k, str) and k.lower() in targets_lower:
                if isinstance(v, (str, int, float)) and str(v).strip():
                    return str(v).strip()
            found = _find_key(v, targets_lower)
            if found:
                return found
    elif isinstance(tree, list):
        for v in tree:
            found = _find_key(v, targets_lower)
            if found:
                return found
    return ''


def parse_gen_params_json(filepath):
    """Extract route/direction metadata from an EXFO FastReporter JSON in the
    same shape parse_gen_params() returns for SOR files."""
    import json as _json
    try:
        with open(filepath) as fh:
            data = _json.load(fh)
    except Exception:
        return {}

    loc_a = _find_key(data, {'locationa', 'originatinglocation', 'locationfrom',
                              'locationstart'})
    loc_b = _find_key(data, {'locationb', 'terminatinglocation', 'locationto',
                              'locationend'})
    cable_id = _find_key(data, {'cableid', 'cable'})
    fiber_id = _find_key(data, {'fiberid', 'fibernumber', 'fiber'})
    cable_code = _find_key(data, {'cablecode'})
    operator = _find_key(data, {'operator', 'user', 'technician'})
    comment = _find_key(data, {'comment', 'comments', 'notes'})

    # Wavelength code (short int matching SOR convention) — pull from the
    # JSON, round to nearest nm.
    wl_str = _find_key(data, {'wavelength', 'nominalwavelength'})
    try:
        wl_nm = int(round(float(wl_str))) if wl_str else 0
    except ValueError:
        wl_nm = 0

    return {
        'cable_id': cable_id,
        'fiber_id': fiber_id,
        'fiber_type_code': 0,
        'wavelength_code': wl_nm,
        'location_a': loc_a,
        'location_b': loc_b,
        'cable_code': cable_code,
        'build_condition': '',
        'operator': operator,
        'comment': comment,
    }


def load_fiber_json(filepath):
    """Parse an EXFO FastReporter JSON and return the same dict shape
    load_fiber() returns for SOR files."""
    from json_reader import parse_otdr_json
    parsed = parse_otdr_json(filepath)
    events = parsed.get('events', [])

    # Fiber-start distance (first surviving event after launch trim).
    first_dist_km = events[0]['dist_km'] if events else 0.0

    # Try to recover an acquisition timestamp from anywhere in the JSON.
    import json as _json
    try:
        with open(filepath) as fh:
            raw_json = _json.load(fh)
    except Exception:
        raw_json = {}
    ts_str = _find_key(raw_json, {'acquisitiondatetime', 'datetime',
                                   'measurementdatetime', 'timestamp'})
    ts = 0
    if ts_str:
        try:
            # ISO-ish: "2026-04-21T14:32:05" or with trailing Z / offset.
            from datetime import datetime as _dt
            for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ',
                        '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f'):
                try:
                    ts = int(_dt.strptime(ts_str.split('+')[0].split('.')[0],
                                          '%Y-%m-%dT%H:%M:%S').timestamp())
                    break
                except ValueError:
                    continue
        except Exception:
            ts = 0

    evt_list = []
    total_splice = 0.0
    total_fiber_atten = 0.0
    prev_km = None
    for i, ev in enumerate(events):
        dist_km = ev['dist_km'] - first_dist_km
        splice = float(ev.get('splice_loss') or 0.0)
        total_splice += splice
        if prev_km is not None:
            span_km = ev['dist_km'] - prev_km
            slope_dBkm = float(ev.get('slope') or 0.0)  # already dB/km
            total_fiber_atten += slope_dBkm * span_km
        prev_km = ev['dist_km']
        evt_list.append({
            'number': ev.get('number', i),
            'dist_km': dist_km,
            'splice_loss': splice,
            'splice_mdB': round(splice * 1000),
            'reflection': float(ev.get('reflection') or 0.0),
        })

    total_loss = total_splice + total_fiber_atten
    filesize = os.path.getsize(filepath)
    return {
        'events': evt_list,
        'timestamp': ts,
        'filesize': filesize,
        'filename': os.path.basename(filepath),
        'total_splice_dB': total_splice,
        'total_fiber_atten_dB': total_fiber_atten,
        'total_loss_dB': total_loss,
        'total_loss_mdB': round(total_loss * 1000),
        'gen_params': parse_gen_params_json(filepath),
    }


def parse_gen_params_any(filepath):
    """Dispatch to the right metadata extractor based on file extension."""
    if filepath.lower().endswith('.json'):
        return parse_gen_params_json(filepath)
    return parse_gen_params(filepath)


def load_fiber_any(filepath):
    """Dispatch to the right loader based on file extension."""
    if filepath.lower().endswith('.json'):
        return load_fiber_json(filepath)
    return load_fiber(filepath)


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
    gp = parse_gen_params(filepath)
    return {'events': evt_list, 'timestamp': ts, 'filesize': len(raw),
            'filename': os.path.basename(filepath),
            'total_splice_dB': total_splice,
            'total_fiber_atten_dB': total_fiber_atten,
            'total_loss_dB': total_loss,
            'total_loss_mdB': round(total_loss * 1000),
            'gen_params': gp}


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


# How many event columns (3 sub-cells each) fit on one landscape page with
# the fixed left-side columns. 5 event groups leaves safe margin for the
# total-loss columns on the last chunk.
EVENTS_PER_CHUNK = 5


RED_STYLE = 'color:#C0392B;font-weight:700'


def _rows(pairs, top_n, event_start, event_end, include_total, highlight=None):
    """Render rows for a given slice of events.

    event_start / event_end: half-open indices into each pair's per_event list.
    include_total:  whether to append the 3 total-loss columns on the right.
    highlight: which ranking column to paint red across every row. One of
        None, 'diff' (Max Diff), 'gap' (Time Gap), or 'total' (Total Loss Δ).
    """
    diff_style = RED_STYLE if highlight == 'diff' else 'font-weight:600'
    total_style = RED_STYLE if highlight == 'total' else 'font-weight:600'

    out = ''
    for rank, p in enumerate(pairs[:top_n], 1):
        a, b = p['fiber_a'], p['fiber_b']
        diff = p['max_diff_mdB']
        ts_a_str = datetime.fromtimestamp(p['timestamp_a']).strftime('%m/%d %H:%M:%S') if p['timestamp_a'] else '---'
        ts_b_str = datetime.fromtimestamp(p['timestamp_b']).strftime('%m/%d %H:%M:%S') if p['timestamp_b'] else '---'
        gap = p.get('time_gap_sec')
        if gap is not None:
            if gap < 120:
                gap_str = f'{gap:.0f}s'
            elif gap < 3600:
                gap_str = f'{gap/60:.0f}m'
            else:
                gap_str = f'{gap/3600:.1f}h'
        else:
            gap_str = '---'
        gap_style = RED_STYLE if highlight == 'gap' else ''
        gap_attr = f' style="{gap_style}"' if gap_style else ''

        evt_cells = ''
        for pe in p['per_event'][event_start:event_end]:
            evt_cells += (f'<td class="center">{pe["loss_a"]:+.3f}</td>'
                          f'<td class="center">{pe["loss_b"]:+.3f}</td>'
                          f'<td class="center" style="font-weight:600">{pe["diff_mdB"]:.0f}</td>')
        total_cells = ''
        if include_total:
            loss_a = p.get('total_loss_a', 0)
            loss_b = p.get('total_loss_b', 0)
            loss_diff = p.get('total_loss_diff', 0)
            total_cells = (f'<td class="center">{loss_a}</td>'
                           f'<td class="center">{loss_b}</td>'
                           f'<td class="center" style="{total_style}">{loss_diff}</td>')
        out += (f'<tr>'
                f'<td class="center">{rank}</td>'
                f'<td class="pair-cell">{a} &#8596; {b}</td>'
                f'<td class="center" style="{diff_style}">{diff:.0f}</td>'
                f'<td class="center" style="font-size:8px">{ts_a_str}</td>'
                f'<td class="center" style="font-size:8px">{ts_b_str}</td>'
                f'<td class="center"{gap_attr}>{gap_str}</td>'
                f'{evt_cells}'
                f'{total_cells}'
                f'</tr>\n')
    return out


def _evt_headers(pairs, event_start, event_end, include_total):
    h, s = '', ''
    if pairs:
        for pe in pairs[0]['per_event'][event_start:event_end]:
            h += (f'<th colspan="3" style="border-left:2px solid #ddd">'
                  f'Evt #{pe["event"]} ({pe["dist_km"]:.3f} km)</th>')
            s += '<th class="r">Fib 1</th><th class="r">Fib 2</th><th class="r">&#916; mdB</th>'
        if include_total:
            h += '<th colspan="3" style="border-left:2px solid #ddd">Total Loss (mdB)</th>'
            s += '<th class="r">Fib 1</th><th class="r">Fib 2</th><th class="r">&#916;</th>'
    return h, s


def _chunked_tables(pairs, title, total_events, force_break_first=False, highlight=None):
    """Build one `<div class='table-section'>` per event chunk. First chunk
    uses `title`; subsequent chunks append " (cont.)". Chunks after the first
    always start on a new page; the first chunk starts a new page only if
    `force_break_first` is True (used between rankings and between directions
    so we don't rely on empty spacer divs that cause double breaks)."""
    if total_events == 0:
        return ''
    blocks = []
    starts = list(range(0, total_events, EVENTS_PER_CHUNK))
    for i, s in enumerate(starts):
        e = min(s + EVENTS_PER_CHUNK, total_events)
        is_last = (i == len(starts) - 1)
        chunk_title = title if i == 0 else f'{title} (cont. events {s+1}&ndash;{e})'
        evt_h, evt_s = _evt_headers(pairs, s, e, include_total=is_last)
        rows_html = _rows(pairs, TOP_N, s, e, include_total=is_last, highlight=highlight)
        needs_break = (i > 0) or force_break_first
        page_break = 'style="page-break-before:always; break-before:page;"' if needs_break else ''
        blocks.append(f'''
<div class="table-section" {page_break}>
<h2>{chunk_title}</h2>
<table class="vote-table">
<thead>
<tr><th>#</th><th style="text-align:left">Pair</th><th>Max Diff (mdB)</th><th>Time A</th><th>Time B</th><th>Gap</th>{evt_h}</tr>
<tr><th></th><th style="text-align:left;font-size:7px;color:#888">Fiber 1 &#8596; Fiber 2</th><th></th><th></th><th></th><th></th>{evt_s}</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>''')
    return ''.join(blocks)


def build_direction_section(pairs, direction_label, section_index=0):
    """Build the HTML fragment for ONE direction: histogram + three chunked
    ranking tables. Direction context is carried in each ranking's h2 title
    so we don't need a full-width banner (which could leave an orphan page
    break on short directions). For sections after the first, the very
    first chunk is forced onto a new page."""
    n_pairs = len(pairs)
    show_chart = n_pairs > 1
    chart_b64 = _histogram_b64(pairs, direction_label) if show_chart else ''
    chart_html = (f'<img src="data:image/png;base64,{chart_b64}" class="chart-img" '
                  f'style="page-break-before:always; break-before:page;" />'
                  if show_chart and section_index > 0 else
                  f'<img src="data:image/png;base64,{chart_b64}" class="chart-img" />'
                  if show_chart else '')

    total_events = len(pairs[0]['per_event']) if pairs else 0

    def t(label):  # prepend direction to each ranking title
        return f'{direction_label} &mdash; {label}'

    # If there is no chart for this non-first section, force the first table
    # onto a new page instead (covers the edge case of 0/1 pair direction).
    force_first = (section_index > 0) and (not show_chart)

    diff_tables = _chunked_tables(
        pairs, t('Ranked by Smallest Splice Loss Difference'), total_events,
        force_break_first=force_first, highlight='diff')
    time_sorted = sorted([p for p in pairs if p.get('time_gap_sec') is not None],
                         key=lambda x: x['time_gap_sec'])
    time_tables = _chunked_tables(
        time_sorted, t('Ranked by Shortest Time Gap'), total_events,
        force_break_first=True, highlight='gap')
    loss_sorted = sorted(pairs, key=lambda x: x.get('total_loss_diff', 999))
    loss_tables = _chunked_tables(
        loss_sorted, t('Ranked by Smallest Total Loss Difference'), total_events,
        force_break_first=True, highlight='total')

    return f'''
{chart_html}

{diff_tables}

{time_tables}

{loss_tables}
'''


def _logo_html():
    logo_b64 = ''
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zerodblogo.png')
    if os.path.exists(logo_path):
        with open(logo_path, 'rb') as f:
            logo_b64 = base64.b64encode(f.read()).decode('ascii')
    if not logo_b64:
        return ''
    return (f'<div style="text-align:center; margin-bottom:16px;">'
            f'<img src="data:image/png;base64,{logo_b64}" '
            f'style="height:60px; margin-left:-30px;" /></div>')


_PAGE_CSS = '''
@page {
  size: letter landscape;
  margin: 10mm 10mm 18mm 10mm;
  @bottom-center {
    content: "Page " counter(page) " of " counter(pages);
    font-size: 8px; color: #000;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  @bottom-right {
    content: "\\A9  ZeroDB";
    font-size: 8px; color: #000;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
}
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       color:#2c2c2a; padding:0; font-size:11px; margin:0; }
@media screen {
  body { padding:16px; max-width:1400px; margin:0 auto; }
}
h1 { font-size:20px; font-weight:500; margin-bottom:2px; }
h2 { font-size:14px; font-weight:500; margin:24px 0 8px; }
.subtitle { font-size:11px; color:#888; margin-bottom:16px; }
.chart-img { width:100%; border-radius:8px; border:1px solid #ddd; margin-bottom:16px; }
.cards { display:flex; gap:10px; margin-bottom:16px; flex-wrap:wrap; }
.card { flex:1 1 180px; background:#fff; border:1px solid rgba(0,0,0,.08);
        border-radius:10px; padding:12px 14px; }
.card-label { font-size:9px; color:#999; margin-bottom:2px; text-transform:uppercase; letter-spacing:.04em; }
.card-value { font-size:22px; font-weight:600; }
.card-sub { font-size:9px; color:#999; margin-top:2px; }
.table-section { page-break-inside:auto; break-inside:auto; }
.table-section h2 { page-break-after:avoid; break-after:avoid; }
.vote-table { width:100%; border-collapse:collapse; font-size:9px;
              font-family:'SF Mono','Courier New',monospace; margin-bottom:4px;
              page-break-inside:auto; break-inside:auto; }
.vote-table thead { display:table-header-group; }
.vote-table tbody { display:table-row-group; }
.vote-table th { background:#f4f3f0; padding:5px 6px; text-align:center;
                 font-weight:600; border:0.5px solid #ddd; font-size:8px; color:#555; }
.vote-table td { padding:4px 6px; border:0.5px solid #ddd; }
.vote-table tr { page-break-inside:avoid; break-inside:avoid; }
.pair-cell { text-align:left !important; font-weight:600; }
.center { text-align:center; }
.r { text-align:right; }
.bold { font-weight:600; }
.dir-banner { background:#2C3E50; color:white; padding:10px 16px; border-radius:8px;
              font-size:14px; font-weight:600; margin:28px 0 12px; }
'''


def build_combined_report(route_name, directions):
    """Build a single HTML report covering multiple directions.

    `directions` is a list of dicts, each with keys:
        label     — direction label string (e.g. 'DNW → RCH')
        pairs     — output of compare_pairs() for that direction
        fiber_nums— sorted list of fiber numbers for that direction
    """
    generated = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Summary cards: one pair of cards per direction (closest + median).
    cards_html = ''
    for d in directions:
        pairs = d['pairs']
        label = d['label']
        if pairs:
            mp = pairs[0]
            med = np.median([p['max_diff_mdB'] for p in pairs])
            cards_html += (
                f'<div class="card">'
                f'<div class="card-label">{label} — closest pair</div>'
                f'<div class="card-value">{mp["max_diff_mdB"]:.0f} mdB</div>'
                f'<div class="card-sub">{mp["fiber_a"]} &#8596; {mp["fiber_b"]}</div>'
                f'</div>'
                f'<div class="card">'
                f'<div class="card-label">{label} — median</div>'
                f'<div class="card-value">{med:.0f} mdB</div>'
                f'<div class="card-sub">{len(pairs)} pairs</div>'
                f'</div>'
            )

    # Subtitle gives the combined fiber count per direction.
    subtitle_bits = []
    for d in directions:
        subtitle_bits.append(
            f'{d["label"]}: {len(d["fiber_nums"])} fibers / {len(d["pairs"])} pairs'
        )
    subtitle = ' &nbsp;&bull;&nbsp; '.join(subtitle_bits)

    sections = ''.join(
        build_direction_section(d['pairs'], d['label'], section_index=i)
        for i, d in enumerate(directions)
    )

    return f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{route_name} — Unidirectional Duplicate Report</title>
<style>{_PAGE_CSS}</style></head><body>

{_logo_html()}
<h1>{route_name} — Unidirectional Duplicate Report</h1>
<div class="subtitle">{subtitle} &nbsp;&bull;&nbsp; generated {generated}</div>

<div class="cards">{cards_html}</div>

{sections}

</body></html>'''


def build_report(fibers, pairs, route_name, direction_label, fiber_nums):
    """Backwards-compatible single-direction wrapper (kept for any external
    callers). New code should use build_combined_report."""
    return build_combined_report(route_name, [{
        'label': direction_label,
        'pairs': pairs,
        'fiber_nums': fiber_nums,
    }])


def html_to_pdf_bytes(html_str, base_url=None):
    """Render an HTML string to a PDF byte blob using WeasyPrint.

    Works identically on macOS and on Streamlit Cloud (Debian) — no browser
    required. `base_url` only matters if the HTML references local files by
    relative path; this report embeds everything as base64, so it's optional.
    """
    from weasyprint import HTML
    return HTML(string=html_str, base_url=base_url).write_pdf()


def html_to_pdf(html_path, pdf_path):
    """Backwards-compatible file-to-file wrapper."""
    with open(html_path, 'r', encoding='utf-8') as fh:
        html = fh.read()
    base = os.path.dirname(os.path.abspath(html_path))
    pdf_bytes = html_to_pdf_bytes(html, base_url=base)
    with open(pdf_path, 'wb') as fh:
        fh.write(pdf_bytes)
    return True, ''
